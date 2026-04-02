import pandas as pd
import numpy as np
import os
import json
import unicodedata
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


# ─────────────────────────────────────────────
# Feature sets per target stat
# ─────────────────────────────────────────────

BASE_FEATURES = [
    "DAYS_REST",
    "IS_BACK_TO_BACK",
    "IS_WELL_RESTED",
    "IS_HOME",
    "OPP_PTS_allowed_avg_L10",
    "OPP_PTS_vs_POSITION_L10",
    "KEY_TEAMMATE_OUT",
    "KEY_OPP_OUT",
    "MIN_avg_L5",
    "MIN_avg_L10",
    "TEAM_PACE_L10",
    "OPP_PACE_L10",
    "COMBINED_PACE",
    "GAMES_SINCE_RETURN",
    "IS_RETURNING",
    "RETURN_GAME_NUMBER",
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
            "PFD_per36_avg_L10",   # fouls drawn — free throw volume signal
            "HIGH_FT_DRAW",
        ],
        "model_file": "pts_model.json",
    },
    "REB": {
        "target": "REB",
        "features": BASE_FEATURES + [
            "REB_avg_L5", "REB_avg_L10", "REB_avg_L20",
            "OREB_avg_L5", "OREB_avg_L10",           # now fixed
            "DREB_avg_L5", "DREB_avg_L10",           # now fixed
            "FGA_avg_L5",
            "PLUS_MINUS_avg_L5",
            "PF_per36_avg_L10",    # foul trouble = less time on glass
            "FOUL_TROUBLE_RISK",
        ],
        "model_file": "reb_model.json",
    },
    "AST": {
        "target": "AST",
        "features": BASE_FEATURES + [
            "AST_avg_L5", "AST_avg_L10", "AST_avg_L20",
            "TOV_avg_L5", "TOV_avg_L10",
            "PTS_avg_L5",
            "FGA_avg_L5",
            "PLUS_MINUS_avg_L5",
            "PF_per36_avg_L10",    # foul trouble limits playmaking opportunities
            "FOUL_TROUBLE_RISK",
        ],
        "model_file": "ast_model.json",
    },
}



# ─────────────────────────────────────────────
# Data loading and splitting
# ─────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    path = os.path.join("data", "features", "player_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Feature matrix not found. Run engineer.py first.")
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    logger.info(f"Loaded feature matrix: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """
    Time-aware split — train on first 80% of season,
    test on most recent 20%. Never shuffle.
    """
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    logger.info(f"Train: {train_df['GAME_DATE'].min().date()} -> {train_df['GAME_DATE'].max().date()}")
    logger.info(f"Test:  {test_df['GAME_DATE'].min().date()} -> {test_df['GAME_DATE'].max().date()}")

    return train_df, test_df


def prepare_xy(df: pd.DataFrame, features: list, target: str):
    """Extract X and y, dropping rows where any feature or target is NaN."""
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        logger.warning(f"Missing columns for {target}: {missing}")

    subset = df[available + [target]].dropna()
    X = subset[available]
    y = subset[target]
    return X, y


# ─────────────────────────────────────────────
# Training and evaluation
# ─────────────────────────────────────────────

def train_model(X_train, y_train, target: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    logger.info(f"Training {target} model...")
    model.fit(X_train, y_train)
    logger.info(f"{target} training complete.")
    return model


def evaluate_model(model, X_test, y_test, target: str):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2   = r2_score(y_test, preds)

    metrics = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}
    logger.info(f"{target} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
    return metrics, preds


def save_model(model, metrics: dict, model_file: str, target: str):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_file)
    metrics_path = os.path.join("models", model_file.replace(".json", "_metrics.json"))

    model.save_model(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"{target} model saved to {model_path}")


def train_quantile_models(X_train, y_train, stat: str):
    """
    Train lower (10th pct) and upper (90th pct) quantile models alongside
    the mean model to produce prediction intervals.
    Uses the same hyperparameters as the mean model.
    """
    base_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    lower = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        **base_params,
    )
    upper = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        **base_params,
    )

    logger.info(f"Training {stat} quantile models (p10 / p90)...")
    lower.fit(X_train, y_train)
    upper.fit(X_train, y_train)
    logger.info(f"{stat} quantile training complete.")

    os.makedirs("models", exist_ok=True)
    stat_lower = stat.lower()
    lower_path = os.path.join("models", f"{stat_lower}_model_lower.json")
    upper_path = os.path.join("models", f"{stat_lower}_model_upper.json")
    lower.save_model(lower_path)
    upper.save_model(upper_path)
    logger.info(f"Saved {stat} quantile models -> {lower_path}, {upper_path}")

    return lower, upper


def show_feature_importance(model, feature_names: list, target: str):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print(f"\n--- Top 10 features for {target} ---")
    print(importance.head(10).to_string())


# ─────────────────────────────────────────────
# Main — train all three models
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_features()
    train_df, test_df = split_data(df)

    all_metrics = {}

    for stat, config in STAT_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"  Training {stat} model")
        print(f"{'='*50}")

        X_train, y_train = prepare_xy(train_df, config["features"], config["target"])
        X_test, y_test   = prepare_xy(test_df,  config["features"], config["target"])

        model = train_model(X_train, y_train, stat)
        metrics, preds = evaluate_model(model, X_test, y_test, stat)
        save_model(model, metrics, config["model_file"], stat)
        show_feature_importance(model, X_train.columns.tolist(), stat)

        train_quantile_models(X_train, y_train, stat)

        all_metrics[stat] = metrics

        # Sample predictions
        results = test_df.loc[y_test.index, ["PLAYER_NAME", "GAME_DATE", stat]].copy()
        results[f"{stat}_PRED"] = preds.round(1)
        results["ERROR"] = (results[f"{stat}_PRED"] - results[stat]).abs().round(1)
        results["PLAYER_NAME"] = results["PLAYER_NAME"].apply(
            lambda n: unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
        )
        print(f"\n--- Sample {stat} predictions ---")
        print(results[["PLAYER_NAME", "GAME_DATE", stat,
                        f"{stat}_PRED", "ERROR"]].head(10).to_string())

    # Summary
    print(f"\n{'='*50}")
    print("  Model summary")
    print(f"{'='*50}")
    for stat, m in all_metrics.items():
        print(f"{stat}  ->  MAE: {m['MAE']}  RMSE: {m['RMSE']}  R2: {m['R2']}")