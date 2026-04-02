from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Make sure Airflow can find our src modules inside the container
sys.path.insert(0, "/opt/airflow")

from src.ingestion.nba_stats import fetch_player_gamelogs, fetch_team_gamelogs, save_raw

# Default arguments applied to every task in the DAG
default_args = {
    "owner": "nba_predictor",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def ingest_player_logs():
    df = fetch_player_gamelogs(season="2025-26", season_type="Regular Season")
    if not df.empty:
        save_raw(df, filename_prefix="player_gamelogs", season="2025-26")


def ingest_team_logs():
    df = fetch_team_gamelogs(season="2025-26", season_type="Regular Season")
    if not df.empty:
        save_raw(df, filename_prefix="team_gamelogs", season="2025-26")


with DAG(
    dag_id="nba_ingest_daily",
    description="Fetch NBA player and team game logs daily",
    default_args=default_args,
    start_date=datetime(2025, 10, 1),
    schedule_interval="0 9 * * *",
    catchup=False,
    is_paused_upon_creation=True,
    tags=["nba", "ingestion"],
) as dag:

    task_player_logs = PythonOperator(
        task_id="ingest_player_logs",
        python_callable=ingest_player_logs,
    )

    task_team_logs = PythonOperator(
        task_id="ingest_team_logs",
        python_callable=ingest_team_logs,
    )

    # Run player logs first, then team logs
    task_player_logs >> task_team_logs