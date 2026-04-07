import os
from datetime import date, datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
import sys

sys.path.insert(0, "/opt/airflow")

from src.models.predict import (
    load_models,
    build_prediction_input,
    filter_available_players,
    generate_predictions,
    apply_calibration,
    load_quantile_models,
    merge_vegas_lines,
    save_predictions,
)

default_args = {
    "owner": "nba_predictor",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def features_file_ready() -> bool:
    """
    Return True if data/features/player_features.csv was written today.
    """
    path  = "/opt/airflow/data/features/player_features.csv"
    today = date.today()
    if not os.path.exists(path):
        return False
    return date.fromtimestamp(os.path.getmtime(path)) == today


def run_predictions():
    """Full prediction pipeline task."""
    game_date = datetime.now().strftime("%Y-%m-%d")
    season    = "2025-26"

    models   = load_models()
    q_models = load_quantile_models()
    player_df, playing_team_ids = build_prediction_input(game_date=game_date)

    if player_df.empty:
        print(f"No games found for {game_date} - skipping predictions.")
        return

    player_df = filter_available_players(
        player_df,
        team_ids=playing_team_ids,
        season=season,
    )

    if player_df.empty:
        print("No available players after filtering.")
        return

    results = generate_predictions(models, player_df, q_models=q_models)
    results = apply_calibration(results)
    results = merge_vegas_lines(results)
    save_predictions(results, game_date)
    print(f"Saved predictions for {game_date} - {len(results)} players.")


with DAG(
    dag_id="nba_predict_daily",
    description="Generate NBA player stat predictions for today's games",
    default_args=default_args,
    start_date=datetime(2025, 10, 1),
    schedule_interval="0 11 * * *",
    catchup=False,
    is_paused_upon_creation=True,
    tags=["nba", "predictions"],
) as dag:

    wait_for_features = PythonSensor(
        task_id="wait_for_features",
        python_callable=features_file_ready,
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    task_run_predictions = PythonOperator(
        task_id="run_predictions",
        python_callable=run_predictions,
    )

    wait_for_features >> task_run_predictions