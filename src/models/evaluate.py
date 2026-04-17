import os
import sys
import unicodedata
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from nba_api.stats.endpoints import playergamelogs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").strip()


def _fmt(val) -> str:
    try:
        return f"{float(val):.1f}"
    except Exception:
        return "-"


def _mae(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return s.abs().mean()


def _bias(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return s.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(date_str: str) -> pd.DataFrame:
    path = os.path.join("data", "predictions", f"predictions_{date_str}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions file: {path}")
    df = pd.read_csv(path)
    df["PLAYER_NAME_ASCII"] = df["PLAYER_NAME"].apply(_ascii)
    logger.info(f"Loaded {len(df)} predictions from {path}")
    return df


def fetch_actuals(date_str: str) -> pd.DataFrame:
    """
    Pull player game logs for a specific date using playergamelogs.
    Tries Regular Season first; falls back to Playoffs if empty.
    """
    logger.info(f"Fetching actual box scores for {date_str} ...")
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_fmt = dt.strftime("%m/%d/%Y")

    for season_type in ("Regular Season", "PlayIn", "Playoffs"):
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable="2025-26",
                season_type_nullable=season_type,
                date_from_nullable=date_fmt,
                date_to_nullable=date_fmt,
            )
            df = logs.get_data_frames()[0]
        except Exception as e:
            logger.error(f"Failed to fetch actuals ({season_type}): {e}")
            raise

        if not df.empty:
            df["PLAYER_NAME_ASCII"] = df["PLAYER_NAME"].apply(_ascii)
            logger.info(f"Fetched {len(df)} rows for {date_str} ({season_type})")
            return df

        logger.info(f"No data for {date_str} ({season_type}), trying next...")

    logger.warning(f"No box score data returned for {date_str} (Regular Season or Playoffs).")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_df(preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join predictions onto actuals keyed on ASCII player name.
    Actuals may have duplicate rows if a player played twice (rare); keep first.
    """
    act = actuals[["PLAYER_NAME_ASCII", "PTS", "REB", "AST"]].drop_duplicates(
        subset="PLAYER_NAME_ASCII", keep="first"
    ).rename(columns={"PTS": "PTS_ACTUAL", "REB": "REB_ACTUAL", "AST": "AST_ACTUAL"})

    merged = preds.merge(act, on="PLAYER_NAME_ASCII", how="left")

    # Error columns (pred - actual; positive = predicted too high)
    for stat in ("PTS", "REB", "AST"):
        pred_col   = f"{stat}_PRED"
        actual_col = f"{stat}_ACTUAL"
        merged[f"{stat}_ERROR"] = merged[pred_col] - merged[actual_col]

    # Vegas error
    merged["VEGAS_ERROR"] = merged["VEGAS_PTS"] - merged["PTS_ACTUAL"]

    # Whether the model beat Vegas on PTS (smaller absolute error)
    has_both = merged["VEGAS_PTS"].notna() & merged["PTS_ACTUAL"].notna()
    merged["MODEL_BEAT_VEGAS"] = None
    merged.loc[has_both, "MODEL_BEAT_VEGAS"] = (
        merged.loc[has_both, "PTS_ERROR"].abs()
        < merged.loc[has_both, "VEGAS_ERROR"].abs()
    )

    # WAS_FLAGGED: |PTS_DIFF| > 3.0
    try:
        merged["WAS_FLAGGED"] = merged["PTS_DIFF"].apply(
            lambda v: not pd.isna(v) and abs(float(v)) > 3.0
        )
    except Exception:
        merged["WAS_FLAGGED"] = False

    return merged


def save_evaluation(df: pd.DataFrame, date_str: str):
    os.makedirs(os.path.join("data", "predictions"), exist_ok=True)
    out_path = os.path.join("data", "predictions", f"evaluation_{date_str}.csv")
    cols = [
        "PLAYER_NAME",
        "PTS_PRED", "PTS_ACTUAL", "PTS_ERROR",
        "VEGAS_PTS", "VEGAS_ERROR",
        "REB_PRED", "REB_ACTUAL", "REB_ERROR",
        "AST_PRED", "AST_ACTUAL", "AST_ERROR",
        "WAS_FLAGGED", "MODEL_BEAT_VEGAS",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.to_csv(out_path, index=False)
    logger.info(f"Saved evaluation to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────────────────

OUTLIER_THRESHOLD = 15.0


def _sign(val: float) -> str:
    return "+" if val >= 0 else ""


def print_report(df: pd.DataFrame, date_str: str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_label = dt.strftime("%B %d, %Y")

    matched   = df["PTS_ACTUAL"].notna()
    matched_df = df[matched].copy()
    matched_df["ABS_PTS_ERROR"] = matched_df["PTS_ERROR"].abs()

    # Outlier split
    is_outlier = matched_df["ABS_PTS_ERROR"] > OUTLIER_THRESHOLD
    outlier_df = matched_df[is_outlier].sort_values("ABS_PTS_ERROR", ascending=False)
    clean_df   = matched_df[~is_outlier]

    n_pred      = len(df)
    n_matched   = len(matched_df)
    n_outliers  = len(outlier_df)
    n_vegas     = (df["VEGAS_PTS"].notna() & matched).sum()

    # Full-set metrics (all matched players)
    pts_mae_full  = _mae(matched_df["PTS_ERROR"])
    pts_bias_full = _bias(matched_df["PTS_ERROR"])

    # Clean-set metrics
    has_vegas_clean = clean_df["VEGAS_PTS"].notna()
    pts_mae_clean   = _mae(clean_df["PTS_ERROR"])
    reb_mae_clean   = _mae(clean_df["REB_ERROR"])
    ast_mae_clean   = _mae(clean_df["AST_ERROR"])
    pts_bias_clean  = _bias(clean_df["PTS_ERROR"])
    reb_bias_clean  = _bias(clean_df["REB_ERROR"])
    ast_bias_clean  = _bias(clean_df["AST_ERROR"])

    model_mae_clean = _mae(clean_df.loc[has_vegas_clean, "PTS_ERROR"])
    vegas_mae_clean = _mae(clean_df.loc[has_vegas_clean, "VEGAS_ERROR"])
    n_vegas_clean   = has_vegas_clean.sum()
    n_beat_clean    = (clean_df.loc[has_vegas_clean, "MODEL_BEAT_VEGAS"] == True).sum()

    # Flagged breakdown (clean set)
    flagged_clean = clean_df["WAS_FLAGGED"]
    n_flagged     = (df["WAS_FLAGGED"] & matched).sum()
    flag_model    = (flagged_clean & (clean_df["PTS_ERROR"].abs() < clean_df["VEGAS_ERROR"].abs())).sum()
    flag_vegas    = (flagged_clean & (clean_df["PTS_ERROR"].abs() >= clean_df["VEGAS_ERROR"].abs())).sum()

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 52)
    print(f"  Post-Game Evaluation: {date_label}")
    print("=" * 52)
    print(f"Players predicted:           {n_pred}")
    print(f"Players with actual results: {n_matched}")
    print(f"Players with Vegas lines:    {n_vegas}")

    # ── Full results ──────────────────────────────────────────────────────────
    print("\n--- Full results (all players) ---")
    print(f"  PTS MAE:  {pts_mae_full:.2f}")
    print(f"  PTS bias: {_sign(pts_bias_full)}{pts_bias_full:.2f}")
    print(f"  Players excluded as outliers: {n_outliers} (error > {OUTLIER_THRESHOLD:.0f} pts)")

    # ── Excluded outliers list ────────────────────────────────────────────────
    if not outlier_df.empty:
        print("\nExcluded outliers (abs PTS error > 15):")
        for _, row in outlier_df.iterrows():
            err = row["PTS_ERROR"]
            err_str = f"{_sign(err)}{err:.1f}"
            name = _ascii(row["PLAYER_NAME"])
            print(f"  {name:<22}  predicted {_fmt(row['PTS_PRED']):>5}  "
                  f"actual {_fmt(row['PTS_ACTUAL']):>5}  error {err_str:>6}")

    # ── Clean results ─────────────────────────────────────────────────────────
    print("\n--- Clean results (outliers removed) ---")
    print(f"  PTS MAE:  {pts_mae_clean:.2f}")
    print(f"  PTS bias: {_sign(pts_bias_clean)}{pts_bias_clean:.2f}")
    print(f"  REB MAE:  {reb_mae_clean:.2f}")
    print(f"  REB bias: {_sign(reb_bias_clean)}{reb_bias_clean:.2f}")
    print(f"  AST MAE:  {ast_mae_clean:.2f}")
    print(f"  AST bias: {_sign(ast_bias_clean)}{ast_bias_clean:.2f}")

    print("\n  Model vs Vegas (clean set):")
    print(f"    Model PTS MAE: {model_mae_clean:.2f}")
    print(f"    Vegas PTS MAE: {vegas_mae_clean:.2f}")
    pct = (n_beat_clean / n_vegas_clean * 100) if n_vegas_clean > 0 else 0.0
    print(f"    Model beat Vegas: {n_beat_clean}/{n_vegas_clean} ({pct:.0f}%)")

    # ── Top 10 best / worst (clean set) ───────────────────────────────────────
    best  = clean_df.nsmallest(10, "ABS_PTS_ERROR")
    worst = clean_df.nlargest(10, "ABS_PTS_ERROR")

    print("\nTop 10 best predictions (smallest PTS error, clean set):")
    print(f"  {'Player':<28} {'Predicted':>9} {'Actual':>7} {'Error':>7}")
    print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*7}")
    for _, row in best.iterrows():
        err = row["PTS_ERROR"]
        print(f"  {_ascii(row['PLAYER_NAME']):<28} {_fmt(row['PTS_PRED']):>9} "
              f"{_fmt(row['PTS_ACTUAL']):>7} {_sign(err)}{err:.1f}".rstrip())

    print("\nTop 10 worst predictions (largest PTS error, clean set):")
    print(f"  {'Player':<28} {'Predicted':>9} {'Actual':>7} {'Error':>7}")
    print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*7}")
    for _, row in worst.iterrows():
        err = row["PTS_ERROR"]
        print(f"  {_ascii(row['PLAYER_NAME']):<28} {_fmt(row['PTS_PRED']):>9} "
              f"{_fmt(row['PTS_ACTUAL']):>7} {_sign(err)}{err:.1f}".rstrip())

    # ── Vegas disagreements ───────────────────────────────────────────────────
    print("\nVegas disagreements result:")
    print(f"  Flagged players (* diff > 3.0): {n_flagged}")
    if n_flagged > 0:
        print(f"  Model was closer: {flag_model}/{n_flagged}")
        print(f"  Vegas was closer: {flag_vegas}/{n_flagged}")
    else:
        print("  No flagged players with actual results.")

    print("")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/models/evaluate.py YYYY-MM-DD")
        sys.exit(1)

    date_arg = sys.argv[1].strip()
    try:
        datetime.strptime(date_arg, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {date_arg}  (expected YYYY-MM-DD)")
        sys.exit(1)

    preds   = load_predictions(date_arg)
    actuals = fetch_actuals(date_arg)

    if actuals.empty:
        print(f"No actual box scores found for {date_arg}. Cannot evaluate.")
        sys.exit(1)

    eval_df  = build_eval_df(preds, actuals)
    out_path = save_evaluation(eval_df, date_arg)
    print_report(eval_df, date_arg)
    print(f"Full results saved -> {out_path}")
