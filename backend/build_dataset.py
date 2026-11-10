"""
GajRaksha - Transferable Dataset Builder
=======================================
Transforms cleaned elephant GPS tracking data into a generalized ML dataset
designed to transfer from Kruger training data to India deployment.

Ecological Transferability Design:
1. Excludes absolute geographic coordinates (latitude, longitude) as model features.
2. Excludes calendar 'month' (hemispheric seasonal inversion).
3. Excludes 'past_presence_count' (grid-cell spatial history specific to Kruger).
4. Includes transferable ecological drivers:
   - hour_of_day: Local hour of day (0-23)
   - is_night: Binary flag (1 if between solar sunset and sunrise, 0 if day)
   - rainfall_last_7_days: Rolling 7-day cumulative precipitation (mm)
   - rainfall_last_30_days: Rolling 30-day cumulative precipitation (mm)
   - speed_kmh: Movement velocity (km/h)
   - step_distance_km: Hourly step displacement (km)
   - heading_change_deg: Turning angle / direction change between steps
   - dist_to_nearest_village_km: Distance to nearest community boundary
5. Target: elephant_presence_risk_near_villages (1 if <= 5 km of village, 0 otherwise)
"""

import os
import sys
import math
import json
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
TRACKS_PARQUET = DATA_DIR / "clean_tracks.parquet"
VILLAGES_FILE = DATA_DIR / "raw" / "villages.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "elephant_presence_dataset.parquet"
OUTPUT_CSV = PROCESSED_DIR / "elephant_presence_dataset.csv"
WEATHER_CACHE = CACHE_DIR / "kruger_rainfall_daily.json"

# Strict Transferable Feature List (No lat, No lon, No month, No cell history)
FEATURE_COLS = [
    "hour_of_day",
    "is_night",
    "rainfall_last_7_days",
    "rainfall_last_30_days",
    "speed_kmh",
    "step_distance_km",
    "heading_change_deg",
    "dist_to_nearest_village_km"
]

TARGET_COL = "elephant_presence_risk_near_villages"


def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def compute_bearing(lon1, lat1, lon2, lat2):
    """Calculates compass heading / bearing from point 1 to point 2 in degrees (0-360)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def calculate_solar_sunrise_sunset(lat: float, lon: float, dt: datetime) -> tuple:
    """
    Computes exact astronomical sunrise and sunset hours (local time)
    for any location and date using standard NOAA solar zenith equations.
    """
    day_of_year = dt.timetuple().tm_yday
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (dt.hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma) - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma) + 0.00148 * math.sin(3.0 * gamma)
    )

    zenith = math.radians(90.833)  # Official solar zenith including atmospheric refraction
    lat_rad = math.radians(lat)
    cos_ha = (math.cos(zenith) / (math.cos(lat_rad) * math.cos(decl)) - math.tan(lat_rad) * math.tan(decl))

    if cos_ha >= 1.0:
        return 12.0, 12.0  # Polar night
    elif cos_ha <= -1.0:
        return 0.0, 24.0   # Polar day

    ha = math.degrees(math.acos(cos_ha))
    tz_offset_min = dt.utcoffset().total_seconds() / 60.0 if dt.utcoffset() else 0.0
    s_noon = 720.0 - 4.0 * lon - eqtime + tz_offset_min
    sunrise_hour = (s_noon - ha * 4.0) / 60.0
    sunset_hour = (s_noon + ha * 4.0) / 60.0
    return sunrise_hour, sunset_hour


def load_or_fetch_rainfall(start_date: str = "2007-07-01", end_date: str = "2009-08-31") -> pd.DataFrame:
    """
    Loads daily historical rainfall from cache or fetches from Open-Meteo ERA5 archive.
    Caches the response in data/cache to respect the zero-re-download rule.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if WEATHER_CACHE.exists():
        with open(WEATHER_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("Fetching historical rainfall data from Open-Meteo archive...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude=-24.71&longitude=31.63"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=precipitation_sum&timezone=Africa%2FJohannesburg"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "GajRaksha/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(WEATHER_CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"✓ Cached historical rainfall to: {WEATHER_CACHE.name}")

    df_rain = pd.DataFrame({
        "date_str": data["daily"]["time"],
        "precipitation_mm": data["daily"]["precipitation_sum"]
    })
    df_rain["precipitation_mm"] = df_rain["precipitation_mm"].fillna(0.0)
    df_rain["date"] = pd.to_datetime(df_rain["date_str"]).dt.date

    # Compute rolling 7-day and 30-day cumulative sums
    df_rain["rainfall_last_7_days"] = df_rain["precipitation_mm"].rolling(7, min_periods=1).sum()
    df_rain["rainfall_last_30_days"] = df_rain["precipitation_mm"].rolling(30, min_periods=1).sum()

    return df_rain.set_index("date")


def build_dataset():
    print("=" * 65)
    print("🐘 GAJRAKSHA - TRANSFERABLE DATASET BUILDER")
    print("=" * 65)

    # 1. Load clean tracks and villages
    if not TRACKS_PARQUET.exists():
        raise FileNotFoundError(f"Clean tracks file missing: {TRACKS_PARQUET}")
    if not VILLAGES_FILE.exists():
        raise FileNotFoundError(f"Villages reference file missing: {VILLAGES_FILE}")

    print(f"Loading cleaned GPS tracks from: {TRACKS_PARQUET.name}...")
    df = pd.read_parquet(TRACKS_PARQUET)
    print(f"Loading village locations from: {VILLAGES_FILE.name}...")
    villages = pd.read_csv(VILLAGES_FILE)

    print(f"Loaded {len(df):,} GPS fixes across {df['individual_local_identifier'].nunique()} elephants.")
    print(f"Loaded {len(villages)} border villages.")

    # 2. Calculate Distance to Nearest Village
    print("\nComputing distance to nearest village for all GPS fixes...")
    v_lats = villages["location_lat"].values
    v_lons = villages["location_long"].values
    v_names = villages["name"].values
    v_ids = villages["village_id"].values
    v_buffers = villages["buffer_radius_km"].values

    min_dists = []
    nearest_names = []
    nearest_ids = []
    presence_risks = []

    track_lats = df["location_lat"].values
    track_lons = df["location_long"].values

    for i in range(len(df)):
        dists = haversine_np(track_lons[i], track_lats[i], v_lons, v_lats)
        closest_idx = np.argmin(dists)
        closest_dist = dists[closest_idx]
        buffer_thresh = v_buffers[closest_idx]

        min_dists.append(round(float(closest_dist), 3))
        nearest_names.append(v_names[closest_idx])
        nearest_ids.append(v_ids[closest_idx])
        
        # Target: 1 if elephant is within community buffer radius (5.0 km), 0 otherwise
        presence_risks.append(1 if closest_dist <= buffer_thresh else 0)

    df["dist_to_nearest_village_km"] = min_dists
    df["nearest_village_name"] = nearest_names
    df["nearest_village_id"] = nearest_ids
    df[TARGET_COL] = presence_risks

    # 3. Compute Local Time & Solar Day/Night
    print("Computing local hour_of_day and solar sunrise/sunset is_night...")
    # hour_of_day (local 0 to 23)
    df["hour_of_day"] = df["timestamp"].dt.hour

    # is_night: Calculated using exact solar sunrise and sunset for each date and location
    # Precompute for unique (date, round_lat, round_lon) to execute in milliseconds
    dates = df["timestamp"].dt.date.values
    unique_dates = sorted(list(set(dates)))
    center_lat = float(df["location_lat"].mean())
    center_lon = float(df["location_long"].mean())

    # Map date to (sunrise_hour, sunset_hour)
    sun_times = {}
    sample_ts = df["timestamp"].iloc[0]
    for d in unique_dates:
        dt_noon = datetime(d.year, d.month, d.day, 12, 0, tzinfo=sample_ts.tzinfo)
        sr, ss = calculate_solar_sunrise_sunset(center_lat, center_lon, dt_noon)
        sun_times[d] = (sr, ss)

    is_night_list = []
    for i in range(len(df)):
        d = dates[i]
        hour_frac = df["timestamp"].iloc[i].hour + df["timestamp"].iloc[i].minute / 60.0
        sr, ss = sun_times[d]
        # Night: before sunrise or after sunset
        is_night_list.append(1 if (hour_frac < sr or hour_frac > ss) else 0)

    df["is_night"] = is_night_list

    # 4. Integrate Historical Rainfall (7-day and 30-day cumulative sums)
    print("Merging rolling 7-day and 30-day rainfall...")
    rain_df = load_or_fetch_rainfall()
    
    df["date_only"] = df["timestamp"].dt.date
    df = df.merge(
        rain_df[["rainfall_last_7_days", "rainfall_last_30_days"]],
        left_on="date_only",
        right_index=True,
        how="left"
    )
    df["rainfall_last_7_days"] = df["rainfall_last_7_days"].fillna(0.0).round(2)
    df["rainfall_last_30_days"] = df["rainfall_last_30_days"].fillna(0.0).round(2)
    df = df.drop(columns=["date_only"])

    # 5. Compute Movement Dynamics (Speed, Step Displacement, Heading Change)
    print("Calculating step displacement, movement speed, and heading turn angle...")
    df = df.sort_values(by=["individual_local_identifier", "timestamp"]).reset_index(drop=True)

    step_distances = []
    speeds = []
    heading_changes = []

    for animal_id, group in df.groupby("individual_local_identifier"):
        glats = group["location_lat"].values
        glons = group["location_long"].values
        gtimes = group["timestamp"].values

        bearings = []
        for j in range(len(group)):
            if j == 0:
                step_distances.append(0.0)
                speeds.append(0.0)
                bearings.append(0.0)
                heading_changes.append(0.0)
                continue

            # Haversine step displacement
            dist_km = haversine_np(glons[j - 1], glats[j - 1], glons[j], glats[j])
            dt_hours = (pd.to_datetime(gtimes[j]) - pd.to_datetime(gtimes[j - 1])).total_seconds() / 3600.0

            speed = dist_km / dt_hours if dt_hours > 0 else 0.0
            bearing = compute_bearing(glons[j - 1], glats[j - 1], glons[j], glats[j])

            # Turning angle change
            prev_bearing = bearings[-1]
            diff = abs(bearing - prev_bearing)
            turn = 360.0 - diff if diff > 180.0 else diff

            step_distances.append(round(float(dist_km), 3))
            speeds.append(round(float(min(speed, 25.0)), 3))
            bearings.append(bearing)
            heading_changes.append(round(float(turn), 1))

    df["step_distance_km"] = step_distances
    df["speed_kmh"] = speeds
    df["heading_change_deg"] = heading_changes

    # 6. Evaluation Split Columns (Rule: Animal split & Time split, never random)
    print("Setting up chronological time-split indicator (75% train / 25% test)...")
    cutoff_idx = int(len(df) * 0.75)
    time_sorted = df.sort_values("timestamp")
    time_cutoff = time_sorted.iloc[cutoff_idx]["timestamp"]
    df["split_time_test"] = (df["timestamp"] >= time_cutoff).astype(int)

    # 7. Save Processed Datasets
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving processed dataset to: {OUTPUT_PARQUET.relative_to(BASE_DIR)}...")
    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    
    # Also save a sample CSV for easy inspection
    csv_sample_path = PROCESSED_DIR / "elephant_presence_dataset_sample.csv"
    df.head(1000).to_csv(csv_sample_path, index=False)
    print(f"✓ Saved Parquet dataset and preview CSV.")

    # 8. Report Column List and Class Balance
    total_fixes = len(df)
    pos_count = (df[TARGET_COL] == 1).sum()
    neg_count = (df[TARGET_COL] == 0).sum()
    pos_pct = (pos_count / total_fixes) * 100.0
    neg_pct = (neg_count / total_fixes) * 100.0

    print("\n" + "=" * 65)
    print("NEW FEATURE & COLUMN LIST (STRICTLY TRANSFERABLE):")
    print("=" * 65)
    print(f"Model Feature Columns ({len(FEATURE_COLS)} features):")
    for idx, col in enumerate(FEATURE_COLS, 1):
        print(f"  {idx}. {col:<28} [dtype: {df[col].dtype}]")

    print("\nExcluded Features (As per transferability rules):")
    print("  ✗ latitude / longitude      (Spatial absolute coordinates - non-transferable)")
    print("  ✗ month                     (Hemispheric season inversion - non-transferable)")
    print("  ✗ past_presence_count       (Kruger-specific cell history - non-transferable)")

    print(f"\nTarget Variable:")
    print(f"  ★ {TARGET_COL:<28} [dtype: {df[TARGET_COL].dtype}]")

    print("\nMetadata Columns (Kept for map display, validation splits, and API):")
    metadata_cols = ["timestamp", "location_lat", "location_long", "individual_local_identifier", "nearest_village_name", "split_time_test"]
    for col in metadata_cols:
        print(f"  • {col}")

    print("\n" + "=" * 65)
    print("TARGET CLASS BALANCE:")
    print("=" * 65)
    print(f"Total Records:                     {total_fixes:,} hourly fixes")
    print(f"Class 0 (Normal / Outside Buffer):  {neg_count:,} ({neg_pct:.2f}%)")
    print(f"Class 1 (High Presence Risk Zone): {pos_count:,} ({pos_pct:.2f}%)")
    print(f"Presence Risk Ratio:               1 in {int(round(total_fixes / pos_count))} fixes")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    build_dataset()
