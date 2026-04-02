import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.train import (
    load_features,
    split_data,
    prepare_xy,
    save_model,
    STAT_CONFIGS,
)


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

def build_models() -> dict:
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=5,
            random_seed=42,
            verbose=0,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ]),
    }


# ─────────────────────────────────────────────
# Train and evaluate one model / stat
# ─────────────────────────────────────────────

def run_one(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, preds)), 3),
        "R2":  round(float(r2_score(y_test, preds)), 3),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_features()
    train_df, test_df = split_data(df)

    # results[model_name][stat] = {"MAE": ..., "R2": ...}
    results: dict[str, dict] = {}

    for model_name in ["XGBoost", "LightGBM", "CatBoost", "Ridge"]:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")
        results[model_name] = {}

        for stat, config in STAT_CONFIGS.items():
            X_train, y_train = prepare_xy(train_df, config["features"], config["target"])
            X_test,  y_test  = prepare_xy(test_df,  config["features"], config["target"])

            model = build_models()[model_name]
            metrics = run_one(model, X_train, y_train, X_test, y_test)
            results[model_name][stat] = metrics
            logger.info(f"  {model_name} {stat} -> MAE: {metrics['MAE']}  R2: {metrics['R2']}")
            print(f"  {stat} -> MAE: {metrics['MAE']}  R2: {metrics['R2']}")

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  Model Comparison")
    print(f"{'='*75}")
    print(
        f"{'Model':<12}  {'PTS MAE':>7}  {'PTS R2':>6}  "
        f"{'REB MAE':>7}  {'REB R2':>6}  "
        f"{'AST MAE':>7}  {'AST R2':>6}"
    )
    print(f"{'-'*75}")

    for name in ["XGBoost", "LightGBM", "CatBoost", "Ridge"]:
        r = results[name]
        print(
            f"{name:<12}  {r['PTS']['MAE']:>7.3f}  {r['PTS']['R2']:>6.3f}  "
            f"{r['REB']['MAE']:>7.3f}  {r['REB']['R2']:>6.3f}  "
            f"{r['AST']['MAE']:>7.3f}  {r['AST']['R2']:>6.3f}"
        )

    print(f"{'='*75}")

    # ── Determine winner on PTS MAE ───────────────────────────────────────
    xgb_pts_mae = results["XGBoost"]["PTS"]["MAE"]
    best_name   = min(results, key=lambda n: results[n]["PTS"]["MAE"])
    best_mae    = results[best_name]["PTS"]["MAE"]

    print(f"\n  XGBoost PTS MAE : {xgb_pts_mae:.3f}")
    print(f"  Best model      : {best_name} (PTS MAE {best_mae:.3f})")

    if best_name != "XGBoost" and best_mae < xgb_pts_mae:
        print(f"\n  {best_name} beats XGBoost by "
              f"{xgb_pts_mae - best_mae:.3f} MAE -- saving as production models.")

        for stat, config in STAT_CONFIGS.items():
            X_train, y_train = prepare_xy(train_df, config["features"], config["target"])
            X_test,  y_test  = prepare_xy(test_df,  config["features"], config["target"])
            winner = build_models()[best_name]
            metrics = run_one(winner, X_train, y_train, X_test, y_test)

            # For sklearn-compatible models wrap predict inside a thin XGBRegressor shell
            # so predict.py can load them with model.load_model(). Instead, save via
            # xgb native format only for XGBoost; for others use joblib alongside a note.
            import joblib
            stat_lower = stat.lower()
            joblib_path = os.path.join("models", f"{stat_lower}_model_{best_name.lower()}.pkl")
            joblib.dump(winner, joblib_path)
            logger.info(f"  Saved {stat} {best_name} -> {joblib_path}")
            print(f"  Saved {stat} {best_name} model -> {joblib_path}")

        print(f"\n  NOTE: production predict.py still uses XGBoost .json models.")
        print(f"  To switch, retrain with {best_name} in train.py and update load_models().")
    else:
        print(f"\n  XGBoost remains the best model -- no production model change.")
