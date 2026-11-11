"""
GajRaksha - Training Dataset Builder: Used vs. Available Design
==============================================================
Transforms raw elephant GPS tracking fixes (presence points) into a
balanced machine learning training dataset using the "Used vs Available"
(Resource Selection Function) framework.

=============================================================================
"USED VS AVAILABLE" DESIGN EXPLANATION (IN SIMPLE WORDS):
-----------------------------------------------------------------------------
1. What does the collar record?
   The GPS collar only logs where an elephant actually traveled. These are
   "USED" locations (Positives, label = 1).
   However, GPS collars never record "true absences" — we cannot tell where an
   elephant purposefully avoided going simply because the collar never records
   non-events.

2. How do we create a comparison for machine learning?
   To train a model to distinguish high-risk from low-risk areas, we need
   counterexamples. We generate "AVAILABLE" points (Negatives, label = 0):
   - For every 1 GPS fix (positive), we sample 5 background grid cells from
     the same landscape.
   - We assign each background cell a random timestamp within that specific
     animal's active tracking period.

3. What does this teach the model?
   By comparing where the elephant chose to be (USED) against all the places
   and times it could have been (AVAILABLE), the algorithm learns what
   environmental factors (distance to water, distance to forest edge, proximity
   to villages, crop density, hour of the day, night vs day, and rainfall)
   drive elephant presence and village interface risk.
=============================================================================

Rules applied:
- 1:5 ratio (5 background negatives for every positive GPS fix).
- No month, past_presence_count, latitude, or longitude as features (ensures
  transferability between ecosystems without memorizing spatial coordinates).
- Cached rainfall from local daily repository.
- Output saved to data/real_dataset.csv.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

TRACKS_FILE = DATA_DIR / "clean_tracks.parquet"
GRID_FILE = DATA_DIR / "grid_features_kruger.parquet"
RAINFALL_FILE = CACHE_DIR / "kruger_rainfall_daily.json"
OUTPUT_DATASET = DATA_DIR / "real_dataset.csv"


def load_cached_rainfall() -> dict:
    """
    Loads daily precipitation history from data/cache and pre-calculates
    rolling 7-day and 30-day cumulative rainfall for instant lookup.
    """
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(f"Rainfall cache not found at: {RAINFALL_FILE}")

    with open(RAINFALL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    daily = data.get("daily", {})
    df_rain = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "precip": daily["precipitation_sum"]
    }).sort_values("date").reset_index(drop=True)

    # Calculate rolling 7-day and 30-day sums
    df_rain["rain_7d"] = df_rain["precip"].rolling(window=7, min_periods=1).sum()
    df_rain["rain_30d"] = df_rain["precip"].rolling(window=30, min_periods=1).sum()

    # Fast O(1) dictionary lookup: 'YYYY-MM-DD' -> (rain_7d, rain_30d)
    rain_lookup = {}
    for _, row in df_rain.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d")
        rain_lookup[d_str] = (round(float(row["rain_7d"]), 2), round(float(row["rain_30d"]), 2))

    return rain_lookup


def build_dataset(neg_ratio: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """
    Constructs the training dataset combining Positives (Used GPS fixes)
    and Negatives (Available landscape background cells).
    """
    np.random.seed(random_seed)

    print("=" * 70)
    print("BUILDING TRAINING DATASET (USED VS AVAILABLE DESIGN)")
    print("=" * 70)

    # 1. Load clean tracking data and Kruger grid features
    print(f"Loading GPS tracks from: {TRACKS_FILE.name}...")
    df_tracks = pd.read_parquet(TRACKS_FILE)
    n_pos = len(df_tracks)
    print(f"✓ Loaded {n_pos:,} positive GPS fixes across {df_tracks['individual_local_identifier'].nunique()} animals.")

    print(f"Loading landscape grid from: {GRID_FILE.name}...")
    df_grid = pd.read_parquet(GRID_FILE)
    n_grid = len(df_grid)
    print(f"✓ Loaded {n_grid:,} Kruger landscape grid cells.")

    # Load rainfall lookup
    print("Loading cached rainfall history (CHIRPS / NASA POWER)...")
    rain_lookup = load_cached_rainfall()
    print(f"✓ Rainfall lookup ready ({len(rain_lookup)} daily records).")

    # -------------------------------------------------------------------------
    # STEP A: Process POSITIVES ("USED" locations where elephants were tracked)
    # -------------------------------------------------------------------------
    print("\n[A] Processing Positives (label = 1)...")

    # Map each GPS fix to its nearest spatial grid cell using KDTree
    grid_coords = df_grid[["location_lon", "location_lat"]].values
    track_coords = df_tracks[["location_long", "location_lat"]].values
    kdtree = cKDTree(grid_coords)
    _, nearest_cell_indices = kdtree.query(track_coords)

    # Extract environmental features from the matched grid cells
    matched_grid_cells = df_grid.iloc[nearest_cell_indices].reset_index(drop=True)

    # Extract local temporal features
    ts_pos = pd.to_datetime(df_tracks["timestamp"])
    # Ensure local time in Africa/Johannesburg (UTC+2)
    if ts_pos.dt.tz is None:
        ts_pos = ts_pos.dt.tz_localize("Africa/Johannesburg")
    else:
        ts_pos = ts_pos.dt.tz_convert("Africa/Johannesburg")

    hours_pos = ts_pos.dt.hour.values
    # Nighttime rule: night between 18:00 (6 PM) and 06:00 (6 AM)
    is_night_pos = ((hours_pos >= 18) | (hours_pos < 6)).astype(int)

    # Lookup rolling rainfall
    date_strs_pos = ts_pos.dt.strftime("%Y-%m-%d").values
    rain_7d_pos = np.array([rain_lookup.get(d, (0.0, 0.0))[0] for d in date_strs_pos])
    rain_30d_pos = np.array([rain_lookup.get(d, (0.0, 0.0))[1] for d in date_strs_pos])

    df_pos = pd.DataFrame({
        "timestamp": ts_pos.dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "animal_id": df_tracks["individual_local_identifier"].values,
        "cell_id": matched_grid_cells["cell_id"].values,
        "hour_of_day": hours_pos,
        "is_night": is_night_pos,
        "rainfall_last_7_days": rain_7d_pos,
        "rainfall_last_30_days": rain_30d_pos,
        "distance_to_forest_edge_km": matched_grid_cells["distance_to_forest_edge_km"].values,
        "distance_to_water_km": matched_grid_cells["distance_to_water_km"].values,
        "distance_to_village_km": matched_grid_cells["distance_to_village_km"].values,
        "crop_density": matched_grid_cells["crop_density"].values,
        "presence": np.ones(n_pos, dtype=int)  # Positives: Used = 1
    })
    print(f"✓ Created {len(df_pos):,} positive rows.")

    # -------------------------------------------------------------------------
    # STEP B: Generate NEGATIVES ("AVAILABLE" background landscape samples)
    # -------------------------------------------------------------------------
    print(f"\n[B] Sampling Negatives (label = 0, ratio 1:{neg_ratio})...")
    # For each animal, sample background cells from the grid at random timestamps
    # within that specific animal's collar active tracking window.
    neg_list = []

    for animal, group in df_tracks.groupby("individual_local_identifier"):
        n_animal_pos = len(group)
        n_animal_neg = n_animal_pos * neg_ratio

        animal_ts = pd.to_datetime(group["timestamp"])
        t_min = animal_ts.min().timestamp()
        t_max = animal_ts.max().timestamp()

        # 1. Random background grid cells across the Kruger landscape
        rand_cell_idx = np.random.randint(0, n_grid, size=n_animal_neg)
        sampled_cells = df_grid.iloc[rand_cell_idx].reset_index(drop=True)

        # 2. Random timestamps within this animal's observation period
        rand_sec = np.random.uniform(t_min, t_max, size=n_animal_neg)
        rand_ts = pd.to_datetime(rand_sec, unit="s", utc=True).tz_convert("Africa/Johannesburg")

        rand_hours = rand_ts.hour.values
        rand_is_night = ((rand_hours >= 18) | (rand_hours < 6)).astype(int)

        rand_date_strs = rand_ts.strftime("%Y-%m-%d").values
        rand_rain_7d = np.array([rain_lookup.get(d, (0.0, 0.0))[0] for d in rand_date_strs])
        rand_rain_30d = np.array([rain_lookup.get(d, (0.0, 0.0))[1] for d in rand_date_strs])

        df_animal_neg = pd.DataFrame({
            "timestamp": rand_ts.strftime("%Y-%m-%d %H:%M:%S").values,
            "animal_id": animal,
            "cell_id": sampled_cells["cell_id"].values,
            "hour_of_day": rand_hours,
            "is_night": rand_is_night,
            "rainfall_last_7_days": rand_rain_7d,
            "rainfall_last_30_days": rand_rain_30d,
            "distance_to_forest_edge_km": sampled_cells["distance_to_forest_edge_km"].values,
            "distance_to_water_km": sampled_cells["distance_to_water_km"].values,
            "distance_to_village_km": sampled_cells["distance_to_village_km"].values,
            "crop_density": sampled_cells["crop_density"].values,
            "presence": np.zeros(n_animal_neg, dtype=int)  # Negatives: Available = 0
        })
        neg_list.append(df_animal_neg)
        print(f"  • Animal {animal}: {n_animal_pos:,} positives -> generated {n_animal_neg:,} random background negatives.")

    df_neg = pd.concat(neg_list, ignore_index=True)
    print(f"✓ Total negative rows generated: {len(df_neg):,}")

    # -------------------------------------------------------------------------
    # STEP C: Combine, Shuffle & Save
    # -------------------------------------------------------------------------
    print("\n[C] Combining and shuffling full dataset...")
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Save to data/real_dataset.csv
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(OUTPUT_DATASET, index=False)
    print(f"✓ Saved final dataset to: {OUTPUT_DATASET}")

    # -------------------------------------------------------------------------
    # STEP D: Display Class Balance & Summary
    # -------------------------------------------------------------------------
    total_rows = len(df_combined)
    pos_count = int((df_combined["presence"] == 1).sum())
    neg_count = int((df_combined["presence"] == 0).sum())

    print("\n" + "=" * 70)
    print("DATASET CLASS BALANCE:")
    print("=" * 70)
    print(f"Total rows:        {total_rows:,}")
    print(f"Positives (Used):  {pos_count:,} ({pos_count / total_rows * 100:.2f}%) [Class 1]")
    print(f"Negatives (Avail): {neg_count:,} ({neg_count / total_rows * 100:.2f}%) [Class 0]")
    print(f"Class Ratio:       1 : {neg_count / pos_count:.1f}")
    print("=" * 70)

    print("\nFINAL LIST OF COLUMNS:")
    print("-" * 70)
    for i, col in enumerate(df_combined.columns, 1):
        print(f"{i:2d}. {col}")
    print("-" * 70)

    print("\nSAMPLE 5 ROWS:")
    print("-" * 70)
    print(df_combined.head(5).to_string())
    print("-" * 70)

    return df_combined


if __name__ == "__main__":
    build_dataset()
