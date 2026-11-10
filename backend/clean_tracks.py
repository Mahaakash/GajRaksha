"""
GajRaksha - Elephant GPS Tracking Data Cleaning Pipeline
=========================================================
Cleans raw elephant GPS tracking data:
1. Drops missing and (0, 0) coordinates.
2. Drops duplicate timestamps per animal.
3. Converts UTC timestamps to local timezone based on region setting
   (Default for Kruger: Africa/Johannesburg; configurable for India: Asia/Kolkata).
4. Resamples to 1 fix per hour per animal.
5. Removes unrealistic GPS jumps faster than 15 km/h.
6. Saves cleaned dataset to data/clean_tracks.parquet.
7. Draws all tracks on a map image at data/track_map.png.
"""

import os
import sys
import math
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless image generation
import matplotlib.pyplot as plt

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_FILE = DATA_DIR / "raw" / "movebank_events.csv"
VILLAGES_FILE = DATA_DIR / "raw" / "villages.csv"
OUTPUT_PARQUET = DATA_DIR / "clean_tracks.parquet"
OUTPUT_MAP = DATA_DIR / "track_map.png"

# Region to timezone mapping (supports both Kruger training and India transfer)
REGION_TIMEZONES = {
    "kruger": "Africa/Johannesburg",
    "south_africa": "Africa/Johannesburg",
    "india": "Asia/Kolkata",
    "assam": "Asia/Kolkata",
    "karnataka": "Asia/Kolkata",
    "west_bengal": "Asia/Kolkata",
}


def load_env_config(env_path: Path) -> dict:
    """Reads REGION and TIMEZONE from .env if present."""
    config = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip().strip("'\"")
    return config


def get_configured_timezone(cli_region: str = None, cli_timezone: str = None) -> tuple:
    """
    Determines the active region and timezone from CLI arguments, .env, or defaults.
    Default for Kruger elephant tracks: Africa/Johannesburg.
    """
    env_cfg = load_env_config(BASE_DIR / ".env")
    
    region = (cli_region or env_cfg.get("REGION") or "kruger").lower()
    
    if cli_timezone:
        timezone = cli_timezone
    elif "TIMEZONE" in env_cfg:
        timezone = env_cfg["TIMEZONE"]
    else:
        timezone = REGION_TIMEZONES.get(region, "Africa/Johannesburg")

    return region, timezone


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in kilometers
    using the Haversine formula.
    """
    r = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def remove_speed_jumps(df_animal: pd.DataFrame, max_speed_kmh: float = 15.0) -> pd.DataFrame:
    """
    Filters out GPS anomaly points where speed from the last valid point exceeds max_speed_kmh.
    Over hourly intervals, sustained speeds > 15 km/h represent GPS jump noise.
    """
    if len(df_animal) <= 1:
        return df_animal

    valid_indices = []
    last_idx = None

    for idx, row in df_animal.iterrows():
        if last_idx is None:
            valid_indices.append(idx)
            last_idx = idx
            continue

        prev_row = df_animal.loc[last_idx]
        dt_hours = (row["timestamp"] - prev_row["timestamp"]).total_seconds() / 3600.0

        if dt_hours <= 0:
            continue

        dist_km = haversine_distance(
            prev_row["location_long"], prev_row["location_lat"],
            row["location_long"], row["location_lat"]
        )
        speed = dist_km / dt_hours

        if speed <= max_speed_kmh:
            valid_indices.append(idx)
            last_idx = idx

    return df_animal.loc[valid_indices].copy()


def clean_tracking_data(raw_csv_path: Path, target_tz: str) -> pd.DataFrame:
    """Performs the full cleaning and filtering pipeline on raw GPS tracks."""
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")

    print(f"Reading raw GPS fixes from: {raw_csv_path.name}")
    df = pd.read_csv(raw_csv_path)
    initial_count = len(df)
    print(f"  Initial row count: {initial_count:,}")

    # 1. Drop missing and (0,0) coordinates
    df = df.dropna(subset=["location_lat", "location_long", "timestamp", "individual_local_identifier"])
    df = df[(df["location_lat"] != 0) & (df["location_long"] != 0)]
    after_coords_count = len(df)
    dropped_coords = initial_count - after_coords_count
    print(f"  Dropped {dropped_coords:,} invalid/zero coordinates. Remaining: {after_coords_count:,}")

    # Convert coordinates to numeric
    df["location_lat"] = pd.to_numeric(df["location_lat"], errors="coerce")
    df["location_long"] = pd.to_numeric(df["location_long"], errors="coerce")
    df = df.dropna(subset=["location_lat", "location_long"])

    # 2. Convert timestamps to datetime and apply target local timezone
    print(f"  Converting timestamps (UTC -> {target_tz})...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(target_tz)

    # 3. Drop duplicate timestamps per animal
    df = df.sort_values(by=["individual_local_identifier", "timestamp"]).reset_index(drop=True)
    before_dups = len(df)
    df = df.drop_duplicates(subset=["individual_local_identifier", "timestamp"], keep="first")
    print(f"  Dropped {before_dups - len(df):,} duplicate timestamps per animal.")

    # 4. Resample to 1 fix per hour per animal using local hour floor
    print("  Resampling to 1 fix per hour per animal (in local hours)...")
    df["hour_bin"] = df["timestamp"].dt.floor("h")
    before_resample = len(df)
    # Keep the first recorded fix in each hourly bin for each animal
    df = df.drop_duplicates(subset=["individual_local_identifier", "hour_bin"], keep="first")
    df = df.drop(columns=["hour_bin"]).reset_index(drop=True)
    print(f"  Hourly resampling reduced fixes from {before_resample:,} to {len(df):,}")

    # 5. Remove speed jumps faster than 15 km/h
    print("  Filtering out GPS speed jumps (> 15 km/h)...")
    cleaned_animals = []
    total_jumps_removed = 0

    for animal_id, group in df.groupby("individual_local_identifier"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        count_before = len(group)
        filtered_group = remove_speed_jumps(group, max_speed_kmh=15.0)
        removed = count_before - len(filtered_group)
        total_jumps_removed += removed
        cleaned_animals.append(filtered_group)

    df_clean = pd.concat(cleaned_animals, ignore_index=True)
    print(f"  Removed {total_jumps_removed:,} jump points exceeding 15 km/h.")
    print(f"  Final cleaned row count: {len(df_clean):,}")

    return df_clean


def plot_track_map(df: pd.DataFrame, output_image_path: Path, villages_csv_path: Path = None):
    """Draws all cleaned elephant tracks on a 2D map with distinct colors and village markers."""
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1)

    # Palette for different elephants
    animals = df["individual_local_identifier"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(animals), 10)))

    for i, animal_id in enumerate(animals):
        animal_df = df[df["individual_local_identifier"] == animal_id].sort_values("timestamp")
        ax.plot(
            animal_df["location_long"],
            animal_df["location_lat"],
            label=f"Elephant {animal_id} ({len(animal_df):,} fixes)",
            color=colors[i % len(colors)],
            alpha=0.6,
            linewidth=1.2,
            marker="o",
            markersize=1.5
        )

    # Plot villages if available
    if villages_csv_path and villages_csv_path.exists():
        villages = pd.read_csv(villages_csv_path)
        ax.scatter(
            villages["location_long"],
            villages["location_lat"],
            color="#d90429",
            marker="^",
            s=80,
            edgecolors="black",
            linewidths=1.2,
            label="Border Villages",
            zorder=5
        )
        for _, v in villages.iterrows():
            ax.annotate(
                v["name"],
                (v["location_long"], v["location_lat"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                fontweight="bold",
                color="#2b2d42"
            )

    ax.set_title("GajRaksha - Cleaned Elephant GPS Tracks & Border Villages", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°S)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"✓ Saved track visualization map to: {output_image_path.name}")


def show_hourly_distribution_comparison(raw_csv_path: Path, df_clean: pd.DataFrame, old_tz: str, new_tz: str):
    """
    Prints a clear comparison of fixes per hour of day before (Asia/Kolkata)
    and after (Africa/Johannesburg), showing how local night-time hours line up.
    """
    # Raw data timestamps in old timezone (Asia/Kolkata)
    df_raw = pd.read_csv(raw_csv_path)
    ts_old = pd.to_datetime(df_raw["timestamp"], utc=True).dt.tz_convert(old_tz)
    hours_old = ts_old.dt.hour.value_counts()

    # Clean data timestamps in new timezone (Africa/Johannesburg)
    hours_new = df_clean["timestamp"].dt.hour.value_counts()

    # Calculate average fixes per hour across the date span (~730 days)
    span_days = max(1, (df_clean["timestamp"].max() - df_clean["timestamp"].min()).days)

    print("\n" + "=" * 76)
    print("HOURLY FIX DISTRIBUTION COMPARISON (BEFORE vs AFTER TIMEZONE CHANGE)")
    print(f"Before: {old_tz} (Shifted +3h30m)  |  After: {new_tz} (True Kruger Local Time)")
    print("=" * 76)
    print(f"{'Hour (0-23)':<12} {'Period':<12} {'Raw Fixes ('+old_tz+')':<25} {'Clean Fixes ('+new_tz+')':<25}")
    print("-" * 76)

    night_fixes_old = 0
    night_fixes_new = 0

    for h in range(24):
        cnt_old = hours_old.get(h, 0)
        cnt_new = hours_new.get(h, 0)
        
        # Local night in Kruger is ~18:00 (6 PM) to 06:00 (6 AM)
        is_night_hour = (h >= 18 or h < 6)
        period = "🌙 Night" if is_night_hour else "☀️ Day"

        if is_night_hour:
            night_fixes_old += cnt_old
            night_fixes_new += cnt_new

        print(f"{h:02d}:00        {period:<12} {cnt_old:<25,d} {cnt_new:<25,d}")

    print("-" * 76)
    print(f"Total Night Fixes (18:00 - 06:00 Local Kruger Time):")
    print(f"  • Using {old_tz} (Misaligned):      {night_fixes_old:,} fixes")
    print(f"  • Using {new_tz} (Aligned to Kruger): {night_fixes_new:,} fixes")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clean raw elephant GPS data.")
    parser.add_argument("--region", type=str, default=None, help="Target region (kruger or india).")
    parser.add_argument("--timezone", type=str, default=None, help="Explicit timezone (e.g. Africa/Johannesburg).")
    args = parser.parse_args()

    print("=" * 60)
    print("🐘 GAJRAKSHA - GPS TRACK CLEANING PIPELINE")
    print("=" * 60)

    # 1. Determine region and timezone
    region, target_tz = get_configured_timezone(args.region, args.timezone)
    print(f"Active Region:            {region.upper()}")
    print(f"Configured Local Timezone: {target_tz}")

    # 2. Run cleaning pipeline with new timezone
    df_clean = clean_tracking_data(RAW_FILE, target_tz)

    # 3. Save to Parquet
    print(f"\nSaving cleaned tracks to: {OUTPUT_PARQUET.relative_to(BASE_DIR)}...")
    df_clean.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    print("✓ Parquet file successfully saved.")

    # 4. Generate Track Map
    print(f"\nRendering track map at: {OUTPUT_MAP.relative_to(BASE_DIR)}...")
    plot_track_map(df_clean, OUTPUT_MAP, VILLAGES_FILE)

    # 5. Display Before/After Hourly Comparison (Asia/Kolkata vs Africa/Johannesburg)
    show_hourly_distribution_comparison(RAW_FILE, df_clean, old_tz="Asia/Kolkata", new_tz=target_tz)

    # 6. Print summary
    raw_df = pd.read_csv(RAW_FILE)
    print("=" * 60)
    print("FINAL SUMMARY:")
    print("=" * 60)
    print(f"Before (Raw Fixes):    {len(raw_df):,} rows")
    print(f"After (Cleaned Fixes): {len(df_clean):,} rows")
    print(f"Total Points Filtered: {len(raw_df) - len(df_clean):,} rows")
    print(f"Number of Elephants:   {df_clean['individual_local_identifier'].nunique()}")
    print(f"Region / Timezone:     {region.upper()} ({target_tz})")
    print(f"Date Range ({target_tz}):")
    print(f"  Start: {df_clean['timestamp'].min()}")
    print(f"  End:   {df_clean['timestamp'].max()}")
    print(f"Clean Data Output:     {OUTPUT_PARQUET.relative_to(BASE_DIR)}")
    print(f"Map Image Output:      {OUTPUT_MAP.relative_to(BASE_DIR)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
