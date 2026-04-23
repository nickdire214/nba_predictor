"""
Player embedding autoencoder.

Learns a 16-dimensional representation for each player from their
career-average stat profile vector. The encoder half is then used
to produce the embedding saved to data/features/player_embeddings.csv.

Usage:
    venv/Scripts/python src/models/embeddings.py
"""

import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

PROFILE_COLS = [
    "PTS_avg_L20", "REB_avg_L20", "AST_avg_L20",
    "FGA_avg_L20", "FG3A_avg_L20", "FTA_avg_L20",
    "MIN_avg_L20", "TOV_avg_L20", "STL_avg_L20",
    "BLK_avg_L20", "OREB_avg_L20", "DREB_avg_L20",
    "PF_per36_avg_L10", "PFD_per36_avg_L10",
    "TEAM_PACE_L10", "COMBINED_PACE",
]

INPUT_DIM  = 16
HIDDEN_DIM = 32
EMB_DIM    = 16
EPOCHS     = 100
LR         = 0.001
BATCH_SIZE = 256
RANDOM_SEED = 42


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").strip()


# ─────────────────────────────────────────────
# Autoencoder architecture
# ─────────────────────────────────────────────

class PlayerAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, emb_dim: int = EMB_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


# ─────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────

def build_player_profiles(features_path: str = "data/features/player_features.csv") -> pd.DataFrame:
    """
    Compute one profile vector per player by averaging all their game rows.
    Returns a DataFrame with PLAYER_NAME_ASCII + PROFILE_COLS.
    """
    logger.info("Loading feature matrix...")
    df = pd.read_csv(features_path).copy()
    df["PLAYER_NAME_ASCII"] = df["PLAYER_NAME"].apply(_ascii)

    available = [c for c in PROFILE_COLS if c in df.columns]
    missing = set(PROFILE_COLS) - set(available)
    if missing:
        logger.warning(f"Missing profile columns: {missing}")

    # Mean across all game rows per player — career average profile
    profile = (
        df.groupby("PLAYER_NAME_ASCII")[available]
        .mean()
        .reset_index()
    )

    # Also keep one canonical PLAYER_NAME (last seen)
    name_map = df.groupby("PLAYER_NAME_ASCII")["PLAYER_NAME"].last()
    profile["PLAYER_NAME"] = profile["PLAYER_NAME_ASCII"].map(name_map)

    # Fill any remaining NaN with column mean
    for col in available:
        profile[col] = profile[col].fillna(profile[col].mean())

    logger.info(f"Built profiles for {len(profile)} players.")
    return profile


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_autoencoder(profiles: pd.DataFrame) -> PlayerAutoencoder:
    torch.manual_seed(RANDOM_SEED)

    data_cols = [c for c in PROFILE_COLS if c in profiles.columns]
    X = profiles[data_cols].values.astype(np.float32)

    # Normalize each feature to zero mean / unit std
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    dataset = TensorDataset(torch.tensor(X_norm))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PlayerAutoencoder(input_dim=len(data_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    logger.info(f"Training autoencoder: {len(profiles)} players, {EPOCHS} epochs...")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(X_norm)
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{EPOCHS}  loss: {avg_loss:.6f}")

    logger.info("Training complete.")

    # Store normalization params on model for later use
    model._norm_mean = mean
    model._norm_std  = std
    return model


# ─────────────────────────────────────────────
# Extract embeddings
# ─────────────────────────────────────────────

def extract_embeddings(model: PlayerAutoencoder, profiles: pd.DataFrame) -> pd.DataFrame:
    data_cols = [c for c in PROFILE_COLS if c in profiles.columns]
    X = profiles[data_cols].values.astype(np.float32)
    X_norm = (X - model._norm_mean) / model._norm_std

    model.eval()
    with torch.no_grad():
        emb = model.encode(torch.tensor(X_norm)).numpy()

    emb_cols = [f"EMB_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "PLAYER_NAME", profiles["PLAYER_NAME"].values)
    emb_df.insert(1, "PLAYER_NAME_ASCII", profiles["PLAYER_NAME_ASCII"].values)

    logger.info(f"Extracted {emb.shape[1]}-dim embeddings for {len(emb_df)} players.")
    return emb_df


# ─────────────────────────────────────────────
# Nearest-neighbor analysis
# ─────────────────────────────────────────────

def cosine_similarity_matrix(emb_matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    normed = emb_matrix / norms
    return normed @ normed.T


def print_nearest_neighbors(emb_df: pd.DataFrame, query_names: list, top_k: int = 5):
    emb_cols = [c for c in emb_df.columns if c.startswith("EMB_")]
    matrix = emb_df[emb_cols].values.astype(np.float32)
    sim = cosine_similarity_matrix(matrix)
    name_col = emb_df["PLAYER_NAME_ASCII"].values

    print(f"\n{'='*60}")
    print("  Player embedding nearest neighbors")
    print(f"{'='*60}")

    for query in query_names:
        query_ascii = _ascii(query)
        matches = np.where(name_col == query_ascii)[0]
        if len(matches) == 0:
            # try partial match
            matches = [i for i, n in enumerate(name_col) if query_ascii.lower() in n.lower()]
        if len(matches) == 0:
            print(f"\n  '{query}' not found in embeddings.")
            continue

        idx = matches[0]
        found_name = _ascii(emb_df.iloc[idx]["PLAYER_NAME"])
        sims = sim[idx]
        # exclude self
        ranked = np.argsort(sims)[::-1]
        ranked = [r for r in ranked if r != idx][:top_k]

        print(f"\n  {found_name} -- nearest neighbors:")
        for rank, r in enumerate(ranked, 1):
            neighbor = _ascii(emb_df.iloc[r]["PLAYER_NAME"])
            print(f"    {rank}. {neighbor:<28}  sim={sims[r]:.4f}")


def print_top_similar_pairs(emb_df: pd.DataFrame, top_k: int = 10):
    emb_cols = [c for c in emb_df.columns if c.startswith("EMB_")]
    matrix = emb_df[emb_cols].values.astype(np.float32)
    sim = cosine_similarity_matrix(matrix)
    n = len(sim)
    names = emb_df["PLAYER_NAME"].values

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim[i, j], names[i], names[j]))

    pairs.sort(reverse=True)

    print(f"\n{'='*60}")
    print(f"  Top {top_k} most similar player pairs (cosine similarity)")
    print(f"{'='*60}")
    print(f"  {'Player A':<28} {'Player B':<28} {'Sim':>6}")
    print(f"  {'-'*28} {'-'*28} {'-'*6}")
    for score, a, b in pairs[:top_k]:
        a_str = _ascii(str(a))[:27]
        b_str = _ascii(str(b))[:27]
        print(f"  {a_str:<28} {b_str:<28} {score:.4f}")


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

def save_embeddings(emb_df: pd.DataFrame,
                    out_path: str = None) -> str:
    """
    Save the embedding DataFrame to CSV.
    Only PLAYER_NAME and EMB_0..EMB_N columns are written.
    Returns the path written.
    """
    if out_path is None:
        out_path = os.path.join("data", "features", "player_embeddings.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_cols = ["PLAYER_NAME"] + [c for c in emb_df.columns if c.startswith("EMB_")]
    emb_df[save_cols].to_csv(out_path, index=False)
    logger.info(f"Saved embeddings to {out_path}  ({len(emb_df)} players)")
    return out_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    profiles  = build_player_profiles()
    model     = train_autoencoder(profiles)
    emb_df    = extract_embeddings(model, profiles)

    out_path = save_embeddings(emb_df)
    print(f"\nSaved embeddings -> {out_path}  ({emb_df.shape[0]} players, 16 dims)")

    # Analysis
    print_top_similar_pairs(emb_df, top_k=10)

    # Load feature matrix to find the lowest PTS_avg_L20 role player
    feat_df = pd.read_csv("data/features/player_features.csv")
    role_player_name = (
        feat_df.groupby("PLAYER_NAME")["PTS_avg_L20"]
        .mean()
        .nsmallest(5)
        .index[0]
    )

    print_nearest_neighbors(emb_df, [
        "Nikola Jokic",
        "Stephen Curry",
        role_player_name,
    ])
