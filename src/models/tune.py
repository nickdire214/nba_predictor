import os
import json
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# ─────────────────────────────────────────────
# Same STAT_CONFIGS as train.py
# ─────────────────────────────────────────────

BASE_FEATURES = [
    "DAYS_REST",
    "IS_BACK_TO_BACK",
    "IS_WELL_RESTED",
    "IS_HOME",
    "OPP_PTS_allowed_avg_L10",
    "KEY_TEAMMATE_OUT",
    "KEY_OPP_OUT",
    "MIN_avg_L5",
    "MIN_avg_L10",
]

STAT_CONFIGS = {
    "PTS": {
        "target": "PTS",
        "features": BASE_FEATURES + [
            "PTS_avg_L5", "PTS_avg_L10", "PTS_avg_L20",
            "FGA_avg_L5", "FGA_avg_L10",
            "FG_PCT_avg_L5", "FG_PCT_avg_L10",
            "FG3A_avg_L5", "FG3_PCT_avg_L5",
            "FTA_avg_L5", "FT_PCT_avg_L5",
            "PLUS_MINUS_avg_L5",
            "PFD_per36_avg_L10",
            "HIGH_FT_DRAW",
        ],
        "model_file": "pts_model_tuned.json",
        "baseline_mae": 4.569,
        "baseline_r2":  0.508,
    },
    "REB": {
        "target": "REB",
        "features": BASE_FEATURES + [
            "REB_avg_L5", "REB_avg_L10", "REB_avg_L20",
            "OREB_avg_L5", "OREB_avg_L10",
            "DREB_avg_L5", "DREB_avg_L10",
            "FGA_avg_L5",
            "PLUS_MINUS_avg_L5",
            "PF_per36_avg_L10",
            "FOUL_TROUBLE_RISK",
        ],
        "model_file": "reb_model_tuned.json",
        "baseline_mae": 1.858,
        "baseline_r2":  0.436,
    },
    "AST": {
        "target": "AST",
        "features": BASE_FEATURES + [
            "AST_avg_L5", "AST_avg_L10", "AST_avg_L20",
            "TOV_avg_L5", "TOV_avg_L10",
            "PTS_avg_L5",
            "FGA_avg_L5",
            "PLUS_MINUS_avg_L5",
            "PF_per36_avg_L10",
            "FOUL_TROUBLE_RISK",
        ],
        "model_file": "ast_model_tuned.json",
        "baseline_mae": 1.336,
        "baseline_r2":  0.510,
    },
}

PARAM_DIST = {
    "n_estimators":     [200, 400, 600, 800],
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [3, 4, 5, 6, 7],
    "subsample":        [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7],
}


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    path = os.path.join("data", "features", "player_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Feature matrix not found. Run engineer.py first.")
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"Loaded feature matrix: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def split_data(df: pd.DataFrame):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    print(f"Train: {len(train_df):,} rows  "
          f"({train_df['GAME_DATE'].min().date()} -> {train_df['GAME_DATE'].max().date()})")
    print(f"Test:  {len(test_df):,} rows  "
          f"({test_df['GAME_DATE'].min().date()} -> {test_df['GAME_DATE'].max().date()})")
    return train_df, test_df


def prepare_xy(df: pd.DataFrame, features: list, target: str):
    available = [f for f in features if f in df.columns]
    subset = df[available + [target]].dropna()
    return subset[available], subset[target]


# ─────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────

def tune_stat(stat: str, cfg: dict, train_df: pd.DataFrame, test_df: pd.DataFrame):
    print()
    print("=" * 60)
    print(f"  Tuning {stat}")
    print("=" * 60)

    X_train, y_train = prepare_xy(train_df, cfg["features"], cfg["target"])
    X_test,  y_test  = prepare_xy(test_df,  cfg["features"], cfg["target"])

    print(f"  Train X: {X_train.shape}  |  Test X: {X_test.shape}")

    base_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DIST,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=False,
    )

    t0 = time.time()
    print(f"  Running RandomizedSearchCV (n_iter=20, TimeSeriesSplit n_splits=5)...")
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Search done in {elapsed:.1f}s")

    best_params = search.best_params_
    best_cv_mae = -search.best_score_

    print(f"\n  Best CV MAE: {best_cv_mae:.3f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")

    # Train final model on full train set with best params
    print(f"\n  Training final {stat} model with best params...")
    final_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    preds = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # Compare to baseline
    baseline_mae = cfg["baseline_mae"]
    baseline_r2  = cfg["baseline_r2"]
    mae_delta = mae - baseline_mae
    r2_delta  = r2 - baseline_r2
    mae_sign  = "-" if mae_delta < 0 else "+"
    r2_sign   = "+" if r2_delta  > 0 else "-"

    print(f"\n  Test set results:")
    print(f"    {'':12}  {'MAE':>8}  {'R2':>8}")
    print(f"    {'-'*30}")
    print(f"    {'Baseline':<12}  {baseline_mae:>8.3f}  {baseline_r2:>8.3f}")
    print(f"    {'Tuned':<12}  {mae:>8.3f}  {r2:>8.3f}")
    print(f"    {'Delta':<12}  {mae_sign}{abs(mae_delta):>7.3f}  {r2_sign}{abs(r2_delta):>7.3f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", cfg["model_file"])
    final_model.save_model(model_path)

    metrics = {
        "MAE": round(mae, 3),
        "R2":  round(r2, 3),
        "best_params": best_params,
        "best_cv_mae": round(best_cv_mae, 3),
    }
    metrics_path = model_path.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved tuned model to {model_path}")

    return {"mae": mae, "r2": r2, "baseline_mae": baseline_mae, "baseline_r2": baseline_r2}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    full_df = load_features()
    train_df, test_df = split_data(full_df)

    summary = {}
    for stat, cfg in STAT_CONFIGS.items():
        summary[stat] = tune_stat(stat, cfg, train_df, test_df)

    print()
    print("=" * 60)
    print("  Summary: baseline vs tuned")
    print("=" * 60)
    header = f"{'Stat':<6}  {'Baseline MAE':>12}  {'Tuned MAE':>10}  {'Delta':>8}  {'Baseline R2':>12}  {'Tuned R2':>9}  {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for stat, m in summary.items():
        mae_delta = m["mae"] - m["baseline_mae"]
        r2_delta  = m["r2"]  - m["baseline_r2"]
        mae_sign  = "-" if mae_delta < 0 else "+"
        r2_sign   = "+" if r2_delta  > 0 else "-"
        print(
            f"{stat:<6}  {m['baseline_mae']:>12.3f}  {m['mae']:>10.3f}  "
            f"{mae_sign}{abs(mae_delta):>7.3f}  "
            f"{m['baseline_r2']:>12.3f}  {m['r2']:>9.3f}  "
            f"{r2_sign}{abs(r2_delta):>7.3f}"
        )
