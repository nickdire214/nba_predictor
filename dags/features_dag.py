import glob
import os
from datetime import date, datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
import sys

sys.path.insert(0, "/opt/airflow")

from src.features.engineer import build_player_features

default_args = {
    "owner": "nba_predictor",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def ingest_file_ready() -> bool:
    """
    Return True if any player_gamelogs_2025_26_*.csv in data/raw/
    was written today (mtime == today's date).
    """
    pattern = "/opt/airflow/data/raw/player_gamelogs_2025_26_*.csv"
    today   = date.today()
    for path in glob.glob(pattern):
        if date.fromtimestamp(os.path.getmtime(path)) == today:
            return True
    return False


with DAG(
    dag_id="nba_features_daily",
    description="Engineer features from raw NBA game logs",
    default_args=default_args,
    start_date=datetime(2025, 10, 1),
    schedule_interval="0 10 * * *",
    catchup=False,
    is_paused_upon_creation=True,
    tags=["nba", "features"],
) as dag:

    wait_for_ingest = PythonSensor(
        task_id="wait_for_ingest",
        python_callable=ingest_file_ready,
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    task_build_features = PythonOperator(
        task_id="build_player_features",
        python_callable=build_player_features,
    )

    wait_for_ingest >> task_build_features