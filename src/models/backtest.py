import os
import sys
import unicodedata
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

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
        "model_file": "pts_model.json",
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
            "PF_per36_avg_L10",
            "FOUL_TROUBLE_RISK",
        ],
        "model_file": "ast_model.json",
    },
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ascii_name(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")


def load_features() -> pd.DataFrame:
    path = os.path.join("data", "features", "player_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Feature matrix not found. Run engineer.py first.")
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"Loaded feature matrix: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def split_test(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last 20% chronologically — same split as train.py."""
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    test_df = df.iloc[split_idx:].copy()
    print(f"Test set: {len(test_df):,} rows  "
          f"({test_df['GAME_DATE'].min().date()} -> {test_df['GAME_DATE'].max().date()})")
    return test_df


def load_model(model_file: str) -> xgb.XGBRegressor:
    path = os.path.join("models", model_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def assign_player_tier(df: pd.DataFrame, full_df: pd.DataFrame) -> pd.Series:
    """
    Tier based on each player's season-average PTS across all games (full dataset),
    so tier reflects full-season role rather than just test-set sample.
    """
    season_avg = full_df.groupby("PLAYER_ID")["PTS"].mean()
    player_tiers = season_avg.apply(
        lambda avg: "star" if avg >= 20 else ("starter" if avg >= 12 else "role")
    )
    return df["PLAYER_ID"].map(player_tiers).fillna("role")


def flag_outliers(results: pd.DataFrame, full_df: pd.DataFrame, z: float = 3.0) -> pd.Series:
    """
    Flag rows where the actual value for ANY of PTS, REB, AST is more than
    `z` standard deviations from that player's season mean (computed on the
    full dataset for stability).  Returns a boolean Series aligned to results.
    """
    outlier_mask = pd.Series(False, index=results.index)
    for stat in ["PTS", "REB", "AST"]:
        player_stats = full_df.groupby("PLAYER_ID")[stat].agg(["mean", "std"])
        player_stats.columns = ["mean", "std"]
        # players with only 1 game have std=NaN — treat as non-outlier
        player_stats["std"] = player_stats["std"].fillna(0)

        merged = results[["PLAYER_ID", stat]].join(
            player_stats, on="PLAYER_ID"
        )
        z_scores = (merged[stat] - merged["mean"]).abs() / merged["std"].replace(0, float("nan"))
        outlier_mask |= z_scores > z

    return outlier_mask


def assign_rest_bucket(days_rest: pd.Series) -> pd.Series:
    def bucket(d):
        if d <= 1:
            return "back-to-back"
        elif d == 2:
            return "1-day-rest"
        else:
            return "2plus-days"
    return days_rest.apply(bucket)


def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_breakdown(label: str, group_col: str, results_df: pd.DataFrame):
    print(f"\n--- MAE by {label} ---")
    header = f"{'Group':<20}  {'N':>6}  {'PTS MAE':>8}  {'REB MAE':>8}  {'AST MAE':>8}"
    print(header)
    print("-" * len(header))
    for group, grp in results_df.groupby(group_col):
        n = len(grp)
        pts_mae = mean_absolute_error(grp["PTS"], grp["PTS_PRED"]) if n > 0 else float("nan")
        reb_mae = mean_absolute_error(grp["REB"], grp["REB_PRED"]) if n > 0 else float("nan")
        ast_mae = mean_absolute_error(grp["AST"], grp["AST_PRED"]) if n > 0 else float("nan")
        print(f"{str(group):<20}  {n:>6,}  {pts_mae:>8.3f}  {reb_mae:>8.3f}  {ast_mae:>8.3f}")


def print_worst(stat: str, results_df: pd.DataFrame, n: int = 10):
    col_pred = f"{stat}_PRED"
    col_err  = f"{stat}_ERROR"
    worst = results_df.nlargest(n, col_err)[
        ["PLAYER_NAME", "GAME_DATE", stat, col_pred, col_err]
    ].copy()
    worst["PLAYER_NAME"] = worst["PLAYER_NAME"].apply(ascii_name)
    worst["GAME_DATE"] = worst["GAME_DATE"].dt.strftime("%Y-%m-%d")
    worst[stat]      = worst[stat].round(1)
    worst[col_pred]  = worst[col_pred].round(1)
    worst[col_err]   = worst[col_err].round(1)

    print(f"\n--- Top 10 worst {stat} predictions ---")
    header = f"{'Player':<25}  {'Date':<12}  {'Actual':>7}  {'Pred':>7}  {'Error':>7}"
    print(header)
    print("-" * len(header))
    for _, row in worst.iterrows():
        print(
            f"{row['PLAYER_NAME']:<25}  {row['GAME_DATE']:<12}  "
            f"{row[stat]:>7.1f}  {row[col_pred]:>7.1f}  {row[col_err]:>7.1f}"
        )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    full_df = load_features()
    test_df = split_test(full_df)

    # Load all three models
    models = {stat: load_model(cfg["model_file"]) for stat, cfg in STAT_CONFIGS.items()}

    # Run predictions for each stat and collect into results
    stat_preds = {}
    for stat, cfg in STAT_CONFIGS.items():
        features = cfg["features"]
        target   = cfg["target"]
        available = [f for f in features if f in test_df.columns]
        subset = test_df[available + [target]].dropna()
        idx = subset.index

        X = subset[available]
        y = subset[target]
        preds = models[stat].predict(X)

        stat_preds[stat] = pd.Series(preds, index=idx, name=f"{stat}_PRED")

    # Build combined results on the intersection of all three valid-index sets
    common_idx = stat_preds["PTS"].index
    for s in ["REB", "AST"]:
        common_idx = common_idx.intersection(stat_preds[s].index)

    results = test_df.loc[common_idx, [
        "PLAYER_ID", "PLAYER_NAME", "GAME_DATE",
        "PTS", "REB", "AST",
        "DAYS_REST", "IS_BACK_TO_BACK",
    ]].copy()

    for stat in ["PTS", "REB", "AST"]:
        results[f"{stat}_PRED"]  = stat_preds[stat].loc[common_idx].values
        results[f"{stat}_ERROR"] = (results[f"{stat}_PRED"] - results[stat]).abs()

    results["PLAYER_TIER"] = assign_player_tier(results, full_df)
    results["REST_BUCKET"] = assign_rest_bucket(results["DAYS_REST"])
    results["MONTH"] = results["GAME_DATE"].dt.strftime("%Y-%m")
    results["OUTLIER_FLAG"] = flag_outliers(results, full_df)

    n_total    = len(results)
    n_outliers = results["OUTLIER_FLAG"].sum()
    n_clean    = n_total - n_outliers
    clean      = results[~results["OUTLIER_FLAG"]]

    # ── Overall metrics ───────────────────────────────────────────────
    print_section("Overall metrics")

    print(f"\nOutlier threshold: 3 standard deviations from per-player season mean")
    print(f"Flagged {n_outliers:,} rows as outliers out of {n_total:,} "
          f"({100 * n_outliers / n_total:.1f}%)")

    print(f"\n  With outliers (N={n_total:,})")
    header = f"  {'Stat':<6}  {'MAE':>8}  {'R2':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stat in ["PTS", "REB", "AST"]:
        mae = mean_absolute_error(results[stat], results[f"{stat}_PRED"])
        r2  = r2_score(results[stat], results[f"{stat}_PRED"])
        print(f"  {stat:<6}  {mae:>8.3f}  {r2:>8.3f}")

    print(f"\n  Without outliers (N={n_clean:,})")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stat in ["PTS", "REB", "AST"]:
        mae = mean_absolute_error(clean[stat], clean[f"{stat}_PRED"])
        r2  = r2_score(clean[stat], clean[f"{stat}_PRED"])
        print(f"  {stat:<6}  {mae:>8.3f}  {r2:>8.3f}")

    # ── Breakdowns (outliers excluded) ───────────────────────────────
    print_section("MAE breakdowns (outliers excluded)")
    print_breakdown("player tier", "PLAYER_TIER", clean)
    print_breakdown("rest situation", "REST_BUCKET", clean)
    print_breakdown("month", "MONTH", clean)

    # ── Worst predictions (outliers excluded) ─────────────────────────
    print_section("Worst predictions (outliers excluded)")
    for stat in ["PTS", "REB", "AST"]:
        print_worst(stat, clean)

    # ── Save CSV ──────────────────────────────────────────────────────
    os.makedirs(os.path.join("data", "predictions"), exist_ok=True)
    out_path = os.path.join("data", "predictions", "backtest_results.csv")

    csv_cols = [
        "PLAYER_NAME", "GAME_DATE",
        "PTS", "PTS_PRED", "PTS_ERROR",
        "REB", "REB_PRED", "REB_ERROR",
        "AST", "AST_PRED", "AST_ERROR",
        "PLAYER_TIER", "DAYS_REST", "MONTH", "OUTLIER_FLAG",
    ]
    out_df = results[csv_cols].copy()
    out_df["GAME_DATE"] = out_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
    for col in ["PTS_PRED", "PTS_ERROR", "REB_PRED", "REB_ERROR", "AST_PRED", "AST_ERROR"]:
        out_df[col] = out_df[col].round(2)
    out_df.to_csv(out_path, index=False)

    print()
    print(f"Saved {len(out_df):,} rows to {out_path}")
