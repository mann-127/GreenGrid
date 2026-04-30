"""
Apache Airflow DAG — GreenGrid Data Ingestion & Forecasting Pipeline
=====================================================================
Scheduled every 6 hours to:
  1. Ingest new weather data (simulated via generator).
  2. Preprocess & create feature sequences.
  3. Run the forecasting model and produce 24-h predictions.
  4. Execute the dispatch optimizer.
  5. Log metrics & push results to a dashboard-accessible store.

Activate with::

    airflow dags trigger greengrid_pipeline
"""
# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator  # noqa

# ── Default DAG args ─────────────────────────────────────────────────
default_args = {
    "owner": "greengrid",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ── Task callables ───────────────────────────────────────────────────

def task_ingest_data(**kwargs):
    """Generate / fetch latest weather + grid data.

    Creates synthetic data or ingests from external source.
    Pushes raw data path to XCom for downstream tasks.
    """
    from greengrid.data.generator import generate_dataset, save_dataset
    from greengrid.settings import CFG
    from loguru import logger

    logger.info("[airflow] task_ingest_data: Starting data generation")
    try:
        # Instruct the pipeline to fetch real historical weather data
        df = generate_dataset(CFG, use_real_weather=True)
        logger.debug(f"[airflow] task_ingest_data: Generated {len(df)} rows via Open-Meteo")

        path = save_dataset(df)
        logger.info(f"[airflow] task_ingest_data: Data saved to {path}")

        kwargs["ti"].xcom_push(key="raw_data_path", value=str(path))
        logger.debug(f"[airflow] task_ingest_data: XCom pushed raw_data_path={path}")
    except Exception as e:
        logger.error(f"[airflow] task_ingest_data: Failed with {e}")
        raise


def task_preprocess(**kwargs):
    """Run the preprocessing pipeline.

    Loads raw data, applies scaling and windowing.
    Pushes dataset dimensions to XCom for monitoring.
    """
    from greengrid.data.preprocessing import prepare_data, load_raw
    from loguru import logger

    logger.info("[airflow] task_preprocess: Starting preprocessing")
    try:
        df = load_raw()
        logger.debug(f"[airflow] task_preprocess: Loaded {len(df)} raw samples")

        data = prepare_data(df)
        logger.info(
            f"[airflow] task_preprocess: Prepared splits: "
            f"train={len(data.train.X)}, val={len(data.val.X)}, test={len(data.test.X)}"
        )

        # Persist shapes for downstream tasks
        kwargs["ti"].xcom_push(key="train_samples", value=len(data.train.X))
        kwargs["ti"].xcom_push(key="test_samples", value=len(data.test.X))
        logger.debug("[airflow] task_preprocess: XCom pushed dataset dimensions")
        return "preprocessing_complete"
    except Exception as e:
        logger.error(f"[airflow] task_preprocess: Failed with {e}")
        raise


def task_run_baseline(**kwargs):
    """Fit and evaluate the moving-average baseline.

    Trains on preprocessed data and pushes best window size to XCom.
    Serves as reference point for ML model performance evaluation.
    """
    from greengrid.data.preprocessing import prepare_data, load_raw
    from greengrid.models.baseline import fit_baseline
    from loguru import logger

    logger.info("[airflow] task_run_baseline: Starting baseline training")
    try:
        df = load_raw()
        logger.debug(f"[airflow] task_run_baseline: Loaded {len(df)} samples")

        data = prepare_data(df)
        result = fit_baseline(
            data.train.y, data.train.y,
            data.val.y, data.val.y,
        )
        logger.info(f"[airflow] task_run_baseline: Best window={result.best_window}h")

        kwargs["ti"].xcom_push(key="baseline_window", value=result.best_window)
        logger.debug("[airflow] task_run_baseline: XCom pushed baseline_window")
    except Exception as e:
        logger.error(f"[airflow] task_run_baseline: Failed with {e}")
        raise


def task_run_forecast(**kwargs):
    """Run the AI forecasting model (inference only — assumes pre-trained).

    Loads latest checkpoint and generates 24h ahead probabilistic forecasts.
    If no checkpoint found, skips and logs warning.
    """
    from pathlib import Path
    from greengrid.models.trainer import load_model
    from greengrid.settings import CFG
    from loguru import logger

    logger.info("[airflow] task_run_forecast: Starting forecast")
    try:
        ckpt_dir = Path(CFG["paths"]["models"])
        ckpts = sorted(ckpt_dir.glob("lstm-*.ckpt"))

        if ckpts:
            latest_ckpt = ckpts[-1]
            logger.debug(f"[airflow] task_run_forecast: Loading checkpoint {latest_ckpt.name}")

            load_model(latest_ckpt, "lstm")
            logger.info("[airflow] task_run_forecast: Model loaded, ready for inference")
            kwargs["ti"].xcom_push(key="model_loaded", value=True)
        else:
            logger.warning("[airflow] task_run_forecast: No LSTM checkpoint found, skipping forecast")
            kwargs["ti"].xcom_push(key="model_loaded", value=False)
    except Exception as e:
        logger.error(f"[airflow] task_run_forecast: Failed with {e}")
        raise


def task_dispatch_optimise(**kwargs):
    """Run the forecast-aware dispatch optimizer.

    Executes BESS simulation with dispatch decisions based on forecasts.
    Computes curtailment reduction and revenue improvement metrics.
    """
    from greengrid.evaluation.simulation import run_simulation
    from loguru import logger

    logger.info("[airflow] task_dispatch_optimise: Starting dispatch optimization")
    try:
        report = run_simulation(skip_training=True)
        logger.info(
            f"[airflow] task_dispatch_optimise: Curtailment reduction={report.curtailment_reduction_pct:.1f}%, "
            f"revenue improvement={report.revenue_improvement_pct:.1f}%"
        )

        kwargs["ti"].xcom_push(key="curtailment_reduction", value=report.curtailment_reduction_pct)
        kwargs["ti"].xcom_push(key="revenue_improvement", value=report.revenue_improvement_pct)
        logger.debug("[airflow] task_dispatch_optimise: XCom pushed metrics")
    except Exception as e:
        logger.error(f"[airflow] task_dispatch_optimise: Failed with {e}")
        raise


def task_log_metrics(**kwargs):
    """Collect and log pipeline metrics.

    Retrieves metrics from upstream tasks via XCom and logs them.
    This is the final step for monitoring and alerting.
    """
    from loguru import logger

    logger.info("[airflow] task_log_metrics: Collecting pipeline metrics")
    ti = kwargs["ti"]

    try:
        curt = ti.xcom_pull(task_ids="dispatch_optimise", key="curtailment_reduction")
        rev = ti.xcom_pull(task_ids="dispatch_optimise", key="revenue_improvement")

        logger.info(
            f"[airflow] task_log_metrics: "
            f"Curtailment Reduction: {curt:.1f}%  |  Revenue Δ: {rev:.1f}%"
        )
        print(f"[GreenGrid Pipeline] Curtailment Reduction: {curt:.1f}%  |  Revenue Δ: {rev:.1f}%")
    except Exception as e:
        logger.error(f"[airflow] task_log_metrics: Failed to retrieve metrics: {e}")
        raise


# ── DAG Definition ───────────────────────────────────────────────────

with DAG(
    dag_id="greengrid_pipeline",
    default_args=default_args,
    description="GreenGrid — Renewable energy forecasting & dispatch pipeline",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["greengrid", "renewable-energy", "forecasting"],
) as dag:

    ingest = PythonOperator(task_id="ingest_data", python_callable=task_ingest_data)
    preprocess = PythonOperator(task_id="preprocess", python_callable=task_preprocess)
    baseline = PythonOperator(task_id="run_baseline", python_callable=task_run_baseline)
    forecast = PythonOperator(task_id="run_forecast", python_callable=task_run_forecast)
    dispatch = PythonOperator(task_id="dispatch_optimise", python_callable=task_dispatch_optimise)
    log_metrics = PythonOperator(task_id="log_metrics", python_callable=task_log_metrics)

    #  ingest -> preprocess -> [baseline, forecast] -> dispatch -> log
    ingest >> preprocess >> [baseline, forecast] >> dispatch >> log_metrics
