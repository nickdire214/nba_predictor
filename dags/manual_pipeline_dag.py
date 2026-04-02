from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.insert(0, "/opt/airflow")

from src.ingestion.nba_stats import fetch_player_gamelogs, fetch_team_gamelogs, save_raw
from src.features.engineer import build_player_features
from src.models.predict import (
    load_models,
    build_prediction_input,
    filter_available_players,
    generate_predictions,
    save_predictions,
)

default_args = {
    "owner": "nba_predictor",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def ingest_player_and_team_logs():
    season = "2025-26"
    player_df = fetch_player_gamelogs(season=season, season_type="Regular Season")
    if not player_df.empty:
        save_raw(player_df, filename_prefix="player_gamelogs", season=season)

    team_df = fetch_team_gamelogs(season=season, season_type="Regular Season")
    if not team_df.empty:
        save_raw(team_df, filename_prefix="team_gamelogs", season=season)


def run_predictions():
    game_date = datetime.now().strftime("%Y-%m-%d")
    season = "2025-26"

    models = load_models()
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

    results = generate_predictions(models, player_df)
    save_predictions(results, game_date)
    print(f"Saved predictions for {game_date} - {len(results)} players.")


with DAG(
    dag_id="nba_manual_pipeline",
    description="Full ingest -> features -> predict pipeline, manually triggered only",
    default_args=default_args,
    start_date=datetime(2025, 10, 1),
    schedule_interval=None,
    catchup=False,
    tags=["nba", "manual"],
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_player_and_team_logs",
        python_callable=ingest_player_and_team_logs,
    )

    def _build_features():
        build_player_features()

    task_features = PythonOperator(
        task_id="build_features",
        python_callable=_build_features,
    )

    task_predict = PythonOperator(
        task_id="run_predictions",
        python_callable=run_predictions,
    )

    task_ingest >> task_features >> task_predict
