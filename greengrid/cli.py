"""
GreenGrid CLI
==============
Unified command-line interface for all GreenGrid operations.

Usage::

    greengrid generate-data
    greengrid preprocess
    greengrid train --model lstm
    greengrid train --model tft
    greengrid simulate
    greengrid dashboard
"""

import click
from loguru import logger


@click.group()
@click.version_option(version="1.0.0", prog_name="GreenGrid")
def main():
    """GreenGrid — AI-Driven Smart Grid Controller"""
    pass


@main.command("generate-data")
@click.option("--output", default=None, help="Output directory for raw data.")
@click.option(
    "--real-weather",
    is_flag=True,
    help="Fetch real historical weather from Open-Meteo instead of synthetic.",
)
def generate_data(output, real_weather):
    """Generate synthetic NREL-style weather & power data, or fetch real data.

    Creates a multi-year dataset with realistic wind speeds, solar irradiance,
    and energy production profiles. Useful for initial testing and demos.

    Args:
        output: Optional output directory. Defaults to config path.
        real_weather: If True, fetches historical data from Open-Meteo API.
    """
    from greengrid.data.generator import (
        fetch_real_weather,
        generate_dataset,
        save_dataset,
    )
    from greengrid.settings import CFG

    logger.info(f"[CLI] Starting data generation (real_weather={real_weather})...")
    try:
        if real_weather:
            # Fetch real weather from Open-Meteo API
            dg = CFG["data_generation"]
            tr = dg["time_range"]
            wind_loc = dg["wind"]["location"]
            df = fetch_real_weather(
                tr["start"],
                tr["end"],
                wind_loc["latitude"],
                wind_loc["longitude"],
            )
            logger.info(f"[CLI] Fetched real weather data: {len(df)} rows")
        else:
            # Generate synthetic data
            df = generate_dataset(CFG)
            logger.debug(f"[CLI] Generated {len(df)} rows")

        path = save_dataset(df, output)
        logger.info(f"[CLI] Data saved to {path}")
        click.echo(f"Data saved to {path}")
    except Exception as e:
        logger.error(f"[CLI] Data generation failed: {e}")
        raise


@main.command("preprocess")
@click.option("--input", "input_dir", default=None, help="Raw data directory.")
def preprocess(input_dir):
    """Run the preprocessing pipeline (scaling + windowing).

    Transforms raw hourly data into train/val/test sequences ready for model training.

    Args:
        input_dir: Optional input data directory. Defaults to config path.

    Raises:
        FileNotFoundError: If raw data not found.
    """
    from greengrid.data.preprocessing import load_raw, prepare_data

    logger.info("[CLI] Starting preprocessing...")
    try:
        df = load_raw(input_dir)
        logger.debug(f"[CLI] Loaded {len(df)} raw samples")

        data = prepare_data(df)
        logger.debug(
            "[CLI] Splits: "
            f"train={len(data.train.X)}, "
            f"val={len(data.val.X)}, "
            f"test={len(data.test.X)}"
        )
        logger.info(f"[CLI] Preprocessing complete with {len(data.feature_columns)} features")

        click.echo(
            f"Preprocessing complete: "
            f"train={len(data.train.X):,}  val={len(data.val.X):,}  "
            f"test={len(data.test.X):,}"
        )
    except Exception as e:
        logger.error(f"[CLI] Preprocessing failed: {e}")
        raise


@main.command("train")
@click.option("--model", type=click.Choice(["lstm", "tft"]), default="lstm")
def train(model):
    """Train a forecasting model.

    Trains either LSTM or TFT on preprocessed data with PyTorch Lightning.
    Checkpoints are saved to models/checkpoints/.

    Args:
        model: Model choice — 'lstm' or 'tft'.

    Raises:
        ValueError: If model type not recognized.
    """
    from greengrid.data.preprocessing import load_raw, prepare_data
    from greengrid.models.trainer import train_lstm, train_tft

    logger.info(f"[CLI] Starting {model.upper()} training...")
    try:
        df = load_raw()
        logger.debug(f"[CLI] Loaded {len(df)} samples")

        data = prepare_data(df)
        logger.debug(f"[CLI] Prepared data: train={len(data.train.X)}, val={len(data.val.X)}")

        if model == "lstm":
            logger.info("[CLI] Training LSTM model")
            train_lstm(data)
        else:
            logger.info("[CLI] Training TFT model")
            train_tft(data)

        logger.info(f"[CLI] {model.upper()} training complete")
        click.echo(f"{model.upper()} training complete.")
    except Exception as e:
        logger.error(f"[CLI] {model.upper()} training failed: {e}")
        raise


@main.command("simulate")
@click.option("--skip-training", is_flag=True, help="Use baseline only (fast mode).")
@click.option(
    "--model",
    type=click.Choice(["lstm", "tft"]),
    default="lstm",
    help="Model type to load.",
)
def simulate(skip_training, model):
    """Run the full simulation & evaluation pipeline.

    Executes end-to-end: baseline -> forecast -> dispatch -> metrics.
    Reports curtailment reduction, revenue improvement, and success status.

    Args:
        skip_training: If True, skip AI models and use baseline only.
        model: Model type to load (lstm or tft).
    """
    from pathlib import Path

    from greengrid.evaluation.simulation import run_simulation
    from greengrid.models.trainer import load_model

    logger.info(f"[CLI] Starting simulation (skip_training={skip_training}, model={model})...")
    try:
        # Load trained model if not skipping
        trained_model = None
        if not skip_training:
            checkpoint_dir = Path("models/checkpoints")
            if checkpoint_dir.exists():
                # Find best checkpoint for the specified model type
                checkpoints = list(checkpoint_dir.glob(f"{model}-epoch=*.ckpt"))
                if checkpoints:

                    def extract_val_loss(p: Path) -> float:
                        try:
                            loss_str = p.stem.split("val_loss=")[1]
                            loss_value = loss_str.split("-")[0] if "-" in loss_str else loss_str
                            return float(loss_value)
                        except (IndexError, ValueError):
                            return float("inf")

                    best_ckpt = min(checkpoints, key=extract_val_loss)
                    logger.info(f"[CLI] Loading checkpoint: {best_ckpt.name}")
                    trained_model = load_model(best_ckpt, model_type=model)
                else:
                    logger.warning(f"[CLI] No {model} checkpoints found in {checkpoint_dir}")
            else:
                logger.warning(f"[CLI] Checkpoint directory not found: {checkpoint_dir}")

        report = run_simulation(model=trained_model, model_type=model, skip_training=skip_training)
        logger.info(
            "[CLI] Simulation complete: "
            f"curtailment_reduction={report.curtailment_reduction_pct:.1f}%, "
            f"revenue_improvement={report.revenue_improvement_pct:.1f}%, "
            f"target_met={report.target_met}"
        )

        click.echo(f"\n{'═' * 50}")
        click.echo(f"  Curtailment Reduction: {report.curtailment_reduction_pct:.1f}%")
        click.echo(f"  Revenue Improvement:   {report.revenue_improvement_pct:.1f}%")
        click.echo(f"  Target Met:            {'YES' if report.target_met else 'NO'}")
        click.echo(f"{'═' * 50}")
    except Exception as e:
        logger.error(f"[CLI] Simulation failed: {e}")
        raise


@main.command("dashboard")
@click.option("--port", default=8501, help="Streamlit port.")
def dashboard(port):
    """Launch the Streamlit dashboard.

    Starts the interactive Streamlit web app for visualization and monitoring.
    Use Ctrl+C to stop.

    Args:
        port: Port number for the web server.
    """
    import subprocess
    import sys
    from pathlib import Path

    logger.info(f"[CLI] Starting Streamlit dashboard on port {port}...")
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    logger.debug(f"[CLI] Dashboard app: {app_path}")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(port),
            ],
            check=False,  # Don't raise on exit
        )
    except Exception as e:
        logger.error(f"[CLI] Dashboard launch failed: {e}")
        raise


if __name__ == "__main__":
    main()
