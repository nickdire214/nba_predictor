"""
Compare Ridge baseline vs Ridge + player embeddings.

Usage:
    venv/Scripts/python src/models/embedding_compare.py
"""

import os
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.train import load_features, split_data, prepare_xy, STAT_CONFIGS
from src.features.engineer import add_player_embeddings

EMB_COLS = [f"EMB_{i}" for i in range(16)]


def run_ridge(X_train, y_train, X_test, y_test) -> dict:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, preds)), 3),
        "R2":  round(float(r2_score(y_test, preds)), 3),
    }


if __name__ == "__main__":
    embeddings_path = os.path.join("data", "features", "player_embeddings.csv")
    if not os.path.exists(embeddings_path):
        print("ERROR: player_embeddings.csv not found.")
        print("Run first: venv/Scripts/python src/models/embeddings.py")
        sys.exit(1)

    # Load and join embeddings onto feature matrix
    logger.info("Loading feature matrix and joining embeddings...")
    df = load_features()
    df = add_player_embeddings(df)
    train_df, test_df = split_data(df)

    results = {"Ridge (baseline)": {}, "Ridge + Embeddings": {}}

    for stat, config in STAT_CONFIGS.items():
        base_features = config["features"]
        emb_features  = base_features + [c for c in EMB_COLS if c in df.columns]

        X_tr_base, y_tr = prepare_xy(train_df, base_features, config["target"])
        X_te_base, y_te = prepare_xy(test_df,  base_features, config["target"])

        X_tr_emb, _  = prepare_xy(train_df, emb_features, config["target"])
        X_te_emb, _  = prepare_xy(test_df,  emb_features, config["target"])

        # Align indices so y is consistent
        common_tr = X_tr_base.index.intersection(X_tr_emb.index)
        common_te = X_te_base.index.intersection(X_te_emb.index)

        m_base = run_ridge(X_tr_base.loc[common_tr], y_tr.loc[common_tr],
                           X_te_base.loc[common_te], y_te.loc[common_te])
        m_emb  = run_ridge(X_tr_emb.loc[common_tr],  y_tr.loc[common_tr],
                           X_te_emb.loc[common_te],  y_te.loc[common_te])

        results["Ridge (baseline)"][stat] = m_base
        results["Ridge + Embeddings"][stat] = m_emb

        logger.info(f"  {stat} baseline    -> MAE: {m_base['MAE']}  R2: {m_base['R2']}")
        logger.info(f"  {stat} + embeddings -> MAE: {m_emb['MAE']}  R2: {m_emb['R2']}")

    # ── Comparison table ──────────────────────────────────────────────────
    W = 78
    print(f"\n{'='*W}")
    print("  Ridge Baseline vs Ridge + Player Embeddings")
    print(f"{'='*W}")
    print(
        f"{'Model':<22}  {'PTS MAE':>7}  {'PTS R2':>6}  "
        f"{'REB MAE':>7}  {'REB R2':>6}  "
        f"{'AST MAE':>7}  {'AST R2':>6}"
    )
    print(f"{'-'*W}")

    for name in ("Ridge (baseline)", "Ridge + Embeddings"):
        r = results[name]
        print(
            f"{name:<22}  {r['PTS']['MAE']:>7.3f}  {r['PTS']['R2']:>6.3f}  "
            f"{r['REB']['MAE']:>7.3f}  {r['REB']['R2']:>6.3f}  "
            f"{r['AST']['MAE']:>7.3f}  {r['AST']['R2']:>6.3f}"
        )

    print(f"{'='*W}")

    base = results["Ridge (baseline)"]
    emb  = results["Ridge + Embeddings"]
    pts_delta = base["PTS"]["MAE"] - emb["PTS"]["MAE"]
    if pts_delta > 0:
        print(f"\n  Embeddings improve PTS MAE by {pts_delta:.3f} -- consider adding to production.")
    elif pts_delta < 0:
        print(f"\n  Embeddings hurt PTS MAE by {abs(pts_delta):.3f} -- baseline Ridge remains better.")
    else:
        print(f"\n  No PTS MAE difference -- embeddings add no signal.")

    print()
