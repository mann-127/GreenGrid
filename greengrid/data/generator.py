"""
Synthetic Weather & Power Data Generator
=========================================
Generates NREL-style hourly weather + renewable-energy production data for
a simulated wind farm and solar farm.  The physics models are simplified but
capture the key statistical properties (diurnal cycles, seasonal trends,
auto-correlation, turbine power curves, etc.) that downstream ML models must
learn.

Usage::

    python -m greengrid.data.generator          # writes to data/raw/
    greengrid generate-data --years 3           # via CLI
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from greengrid.settings import CFG
from greengrid.utils import ensure_dir, hour_to_sincos, month_to_sincos, set_seed

# AR(1) coefficient for wind speed temporal smoothing.
# φ=0.85 gives ~6-hour autocorrelation, matching NREL wind mast observations.
_WIND_AR1_PHI = 0.85

# Diurnal modulation amplitude for wind: ±25% variation peak-to-trough.
_WIND_DIURNAL_AMP = 0.25

# Poisson rate for gust events (~3% of hours experience a gust).
_WIND_GUST_RATE = 0.03

# Day-of-year offset for spring equinox in the simplified solar declination formula.
# Day 81 ≈ March 22, when solar declination crosses 0°.
_SOLAR_EQUINOX_DOY = 81


# ── Real Weather Fetching ──────────────────────────────────────────────
def fetch_real_weather(start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch real historical weather data from the Open-Meteo Archive API."""
    logger.info(f"[generator] Fetching real weather data for {lat}, {lon} from {start_date} to {end_date}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "cloud_cover",
        ],
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Unexpected Open-Meteo response structure: {list(data.keys())}")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "temperature_c": data["hourly"]["temperature_2m"],
            "humidity_pct": data["hourly"]["relative_humidity_2m"],
            "pressure_hpa": data["hourly"]["surface_pressure"],
            "wind_speed_ms": data["hourly"]["wind_speed_10m"],
            "wind_direction_deg": data["hourly"]["wind_direction_10m"],
            "ghi_wm2": data["hourly"]["shortwave_radiation"],
            "cloud_cover_pct": data["hourly"]["cloud_cover"],
        }
    )

    return df.ffill()


# ── Wind-speed physics ─────────────────────────────────────────────────
def _generate_wind_speed(timestamps: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """
    Simulate realistic hourly wind speeds (m/s) using a physics-informed model.

    Physics model:
      • Weibull base distribution (shape k ≈ 2, scale λ seasonal)
      • Diurnal modulation (wind picks up afternoon, calms at night)
      • Seasonal modulation (windier in winter / spring)
      • Auto-regressive AR(1) smoothing for temporal coherence
      • Random gusts (Poisson events with uniform magnitude)

    Args:
        timestamps: DatetimeIndex with hourly frequency.
        seed: Random seed for reproducibility.

    Returns:
        Wind speeds in m/s, clipped to [0, 35] range. Shape (n_hours,).
    """
    logger.debug(f"[generator] Generating wind speeds for {len(timestamps)} hours (seed={seed})")
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    hours = timestamps.hour.values
    months = timestamps.month.values

    # Seasonal scale: stronger in winter (Dec-Feb) & spring
    seasonal_scale = 8.0 + 3.0 * np.cos(2 * np.pi * (months - 1) / 12.0)
    logger.debug(f"[generator] Seasonal scale range: [{seasonal_scale.min():.2f}, {seasonal_scale.max():.2f}] m/s")

    # Diurnal pattern: peak around 14:00, trough around 04:00
    diurnal = 1.0 + _WIND_DIURNAL_AMP * np.sin(2 * np.pi * (hours - 6) / 24.0)

    # Base Weibull samples
    weibull_shape = 2.0
    raw = rng.weibull(weibull_shape, size=n) * seasonal_scale * diurnal

    # AR(1) smoothing for temporal coherence
    smoothed = np.empty(n)
    smoothed[0] = raw[0]
    for i in range(1, n):
        smoothed[i] = _WIND_AR1_PHI * smoothed[i - 1] + (1 - _WIND_AR1_PHI) * raw[i]

    # Add gust events (Poisson-like)
    gusts = rng.poisson(_WIND_GUST_RATE, size=n) * rng.uniform(3, 8, size=n)
    wind = np.clip(smoothed + gusts, 0, 35)

    logger.debug(
        f"[generator] Wind speed stats: mean={wind.mean():.2f}, std={wind.std():.2f}, "
        f"min={wind.min():.2f}, max={wind.max():.2f} m/s"
    )
    return wind.astype(np.float32)


def _wind_power_curve(
    wind_speed: np.ndarray,
    rated_mw: float,
    cut_in: float,
    rated_speed: float,
    cut_out: float,
) -> np.ndarray:
    """Apply IEC 61400-1 standard wind turbine power curve.

    The curve has three regions:
      • Region I (v < cut_in): No power output
      • Region II (cut_in ≤ v < rated): Cubic ramp from 0 to rated power
      • Region III (rated ≤ v ≤ cut_out): Constant rated power
      • Beyond cut_out: Turbine shuts down

    Args:
        wind_speed: Wind speeds in m/s. Shape (n_hours,).
        rated_mw: Rated power of turbine in MW.
        cut_in: Cut-in wind speed (m/s).
        rated_speed: Wind speed at rated power (m/s).
        cut_out: Cut-out wind speed (m/s).

    Returns:
        Power output in MW. Shape (n_hours,).
    """
    if rated_speed <= cut_in:
        raise ValueError(f"rated_speed ({rated_speed} m/s) must be greater than cut_in ({cut_in} m/s)")
    power = np.zeros_like(wind_speed)
    # Region II — cubic ramp (physics: power proportional to wind_speed^3)
    mask_ramp = (wind_speed >= cut_in) & (wind_speed < rated_speed)
    power[mask_ramp] = rated_mw * ((wind_speed[mask_ramp] - cut_in) / (rated_speed - cut_in)) ** 3
    # Region III — rated (governor controls blade pitch)
    mask_rated = (wind_speed >= rated_speed) & (wind_speed <= cut_out)
    power[mask_rated] = rated_mw

    logger.debug(
        f"[generator] Wind power curve: {power.sum():.1f} MWh total, capacity factor={(power.mean() / rated_mw):.1%}"
    )
    return power.astype(np.float32)


# ── Solar irradiance physics ──────────────────────────────────────────
def _generate_solar_irradiance(timestamps: pd.DatetimeIndex, latitude: float, seed: int) -> np.ndarray:
    """
    Approximate Global Horizontal Irradiance (GHI) in W/m² using a
    clear-sky cosine model with cloud-cover perturbation.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(timestamps)
    hours = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values

    # Solar declination (simplified)
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - _SOLAR_EQUINOX_DOY)))
    # Hour angle
    hour_angle = 15.0 * (hours - 12.0)
    # Solar elevation (degrees)
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)
    sin_elev = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    sin_elev = np.clip(sin_elev, 0, 1)

    # Clear-sky GHI
    ghi_clear = 1000.0 * sin_elev  # peak ≈ 1000 W/m²

    # Cloud cover (correlated random walk, 0-1)
    cloud = np.empty(n)
    cloud[0] = rng.uniform(0.1, 0.5)
    for i in range(1, n):
        cloud[i] = np.clip(0.9 * cloud[i - 1] + 0.1 * rng.beta(2, 5), 0, 1)

    ghi = ghi_clear * (1 - 0.75 * cloud)
    # Add small sensor noise
    ghi += rng.normal(0, 5, size=n)
    ghi = np.clip(ghi, 0, 1200).astype(np.float32)

    return ghi, cloud.astype(np.float32)


def _solar_power(
    ghi: np.ndarray,
    capacity_mw: float,
    efficiency: float,
    temperature: np.ndarray,
    temp_coeff: float,
) -> np.ndarray:
    """Convert GHI to solar power using a simple efficiency model."""
    # Reference: 1000 W/m2 = rated capacity at 25 C
    temp_factor = 1 + temp_coeff * (temperature - 25.0)
    power = capacity_mw * (ghi / 1000.0) * efficiency / 0.20 * temp_factor
    return np.clip(power, 0, capacity_mw).astype(np.float32)


# ── Other weather variables ───────────────────────────────────────────
def _generate_temperature(timestamps: pd.DatetimeIndex, lat: float, seed: int) -> np.ndarray:
    """Seasonal + diurnal temperature (°C)."""
    rng = np.random.default_rng(seed + 2)
    n = len(timestamps)
    hours = timestamps.hour.values
    doy = timestamps.dayofyear.values

    seasonal = 15.0 - 15.0 * np.cos(2 * np.pi * (doy - 30) / 365.0)  # warm summer
    diurnal = 5.0 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    noise = rng.normal(0, 2, size=n)
    temp = seasonal + diurnal + noise

    # Latitude adjustment: higher lat => colder baseline
    temp -= max(0, (lat - 35)) * 0.5

    return temp.astype(np.float32)


def _generate_humidity(temperature: np.ndarray, seed: int) -> np.ndarray:
    """Relative humidity (%) — inversely correlated with temperature."""
    rng = np.random.default_rng(seed + 3)
    base = 70 - 0.5 * temperature + rng.normal(0, 5, size=len(temperature))
    return np.clip(base, 10, 100).astype(np.float32)


def _generate_pressure(timestamps: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """Barometric pressure (hPa) — slow random walk around 1013."""
    rng = np.random.default_rng(seed + 4)
    n = len(timestamps)
    p = np.empty(n)
    p[0] = 1013.0
    for i in range(1, n):
        p[i] = 0.995 * p[i - 1] + 0.005 * 1013.0 + rng.normal(0, 0.3)
    return p.astype(np.float32)


def _generate_wind_direction(timestamps: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """Wind direction (degrees) — slow random walk."""
    rng = np.random.default_rng(seed + 5)
    n = len(timestamps)
    d = np.empty(n)
    d[0] = rng.uniform(0, 360)
    for i in range(1, n):
        d[i] = (d[i - 1] + rng.normal(0, 8)) % 360
    return d.astype(np.float32)


# ── Electricity price ────────────────────────────────────────────────
def _generate_electricity_price(timestamps: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """Simulate wholesale electricity spot price ($/MWh)."""
    rng = np.random.default_rng(seed + 6)
    n = len(timestamps)
    hours = timestamps.hour.values
    peak_mask = np.isin(hours, CFG["optimizer"]["peak_hours"])
    base = np.where(peak_mask, 95.0, 40.0)
    noise = rng.normal(0, 8, size=n)
    return np.clip(base + noise, 5, 250).astype(np.float32)


# ── Grid demand ──────────────────────────────────────────────────────
def _generate_grid_demand(timestamps: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """Simulated regional grid demand (MW)."""
    rng = np.random.default_rng(seed + 7)
    n = len(timestamps)
    hours = timestamps.hour.values
    months = timestamps.month.values
    # Diurnal: peaks at 9 AM and 7 PM
    diurnal = 500 + 200 * np.sin(2 * np.pi * (hours - 3) / 24)
    # Seasonal: higher in summer (AC) and winter (heating)
    seasonal = 100 * np.cos(2 * np.pi * (months - 7) / 6)
    noise = rng.normal(0, 30, size=n)
    return np.clip(diurnal + seasonal + noise, 200, 1200).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════


def generate_dataset(cfg: dict | None = None) -> pd.DataFrame:
    """
    Generate a complete synthetic dataset and return a ``DataFrame``.

    Columns
    -------
    timestamp, wind_speed_ms, wind_direction_deg, temperature_c,
    humidity_pct, pressure_hpa, ghi_wm2, cloud_cover_pct,
    wind_power_mw, solar_power_mw, total_renewable_mw,
    electricity_price_mwh, grid_demand_mw,
    hour_sin, hour_cos, month_sin, month_cos
    """
    if cfg is None:
        cfg = CFG

    seed = cfg["project"]["seed"]
    set_seed(seed)
    dg = cfg["data_generation"]
    tr = dg["time_range"]

    timestamps = pd.date_range(start=tr["start"], end=tr["end"], freq=tr["freq"])
    logger.info(f"Generating {len(timestamps):,} hourly records ({tr['start']} to {tr['end']})")

    # ── Weather features ──────────────────────────────────────────────
    if dg.get("use_real_weather", False):
        lat = dg["wind"]["location"]["latitude"]
        lon = dg["wind"]["location"]["longitude"]
        weather_df = fetch_real_weather(tr["start"], tr["end"], lat, lon)
        timestamps = pd.DatetimeIndex(weather_df["timestamp"])
        wind_speed = weather_df["wind_speed_ms"].to_numpy()
        wind_dir = weather_df["wind_direction_deg"].to_numpy()
        temperature = weather_df["temperature_c"].to_numpy()
        humidity = weather_df["humidity_pct"].to_numpy()
        pressure = weather_df["pressure_hpa"].to_numpy()
        ghi = weather_df["ghi_wm2"].to_numpy()
        cloud_cover = weather_df["cloud_cover_pct"].to_numpy() / 100.0
    else:
        wind_speed = _generate_wind_speed(timestamps, seed)
        wind_dir = _generate_wind_direction(timestamps, seed)
        temperature = _generate_temperature(timestamps, dg["wind"]["location"]["latitude"], seed)
        humidity = _generate_humidity(temperature, seed)
        pressure = _generate_pressure(timestamps, seed)
        ghi, cloud_cover = _generate_solar_irradiance(timestamps, dg["solar"]["location"]["latitude"], seed)

    # ── Power production ──────────────────────────────────────────────
    wc = dg["wind"]
    single_turbine_power = _wind_power_curve(
        wind_speed,
        wc["rated_capacity_mw"],
        wc["cut_in_speed_ms"],
        wc["rated_speed_ms"],
        wc["cut_out_speed_ms"],
    )
    wind_power = single_turbine_power * wc["num_turbines"]

    sc = dg["solar"]
    solar_power = _solar_power(
        ghi,
        sc["capacity_mw"],
        sc["panel_efficiency"],
        temperature,
        sc["temperature_coeff"],
    )

    total_renewable = wind_power + solar_power

    # ── Market / demand ───────────────────────────────────────────────
    price = _generate_electricity_price(timestamps, seed)
    demand = _generate_grid_demand(timestamps, seed)

    # ── Cyclical time features ────────────────────────────────────────
    h_sin, h_cos = hour_to_sincos(timestamps.hour.values.astype(float))
    m_sin, m_cos = month_to_sincos(timestamps.month.values.astype(float))

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_dir,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
            "ghi_wm2": ghi,
            "cloud_cover_pct": cloud_cover * 100,
            "wind_power_mw": wind_power,
            "solar_power_mw": solar_power,
            "total_renewable_mw": total_renewable,
            "electricity_price_mwh": price,
            "grid_demand_mw": demand,
            "hour_sin": h_sin,
            "hour_cos": h_cos,
            "month_sin": m_sin,
            "month_cos": m_cos,
        }
    )

    logger.info(
        f"Wind power range: {wind_power.min():.1f} - {wind_power.max():.1f} MW  |  "
        f"Solar power range: {solar_power.min():.1f} - {solar_power.max():.1f} MW"
    )
    return df


def save_dataset(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist the dataset as Parquet + CSV."""
    out_dir = ensure_dir(path or CFG["paths"]["raw_data"])
    parquet_path = out_dir / "greengrid_raw.parquet"
    csv_path = out_dir / "greengrid_raw.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw data to {parquet_path}  ({len(df):,} rows)")
    return parquet_path


# ── CLI entry-point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GreenGrid synthetic data")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()
    df = generate_dataset()
    save_dataset(df, args.output)
