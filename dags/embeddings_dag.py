import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "nba_predictor",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def regenerate_embeddings():
    """
    Rebuild player embeddings from the current feature matrix.
    Imports are deferred to runtime so torch is not required at DAG parse time.
    """
    sys.path.insert(0, "/opt/airflow")

    from loguru import logger
    from src.models.embeddings import (
        build_player_profiles,
        train_autoencoder,
        extract_embeddings,
        save_embeddings,
    )

    logger.info("Regenerating player embeddings...")

    profiles = build_player_profiles(
        features_path="/opt/airflow/data/features/player_features.csv"
    )
    model   = train_autoencoder(profiles)
    emb_df  = extract_embeddings(model, profiles)
    save_embeddings(
        emb_df,
        out_path="/opt/airflow/data/features/player_embeddings.csv",
    )

    logger.info(f"Embeddings regenerated: {len(emb_df)} players")


with DAG(
    dag_id="nba_embeddings_weekly",
    description="Regenerate player embedding vectors every Sunday at 6am UTC",
    default_args=default_args,
    start_date=datetime(2025, 10, 1),
    schedule_interval="0 6 * * 0",
    catchup=False,
    is_paused_upon_creation=True,
    tags=["nba", "embeddings"],
) as dag:

    task_regenerate = PythonOperator(
        task_id="regenerate_embeddings",
        python_callable=regenerate_embeddings,
    )
