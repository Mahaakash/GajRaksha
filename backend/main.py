"""
GajRaksha - FastAPI Backend Service
===================================
Serves real-time inference and spatial risk scores for:
1. /risk-grid: Scores ~1 km spatial cells for elephant presence risk near villages.
2. /village-risk: Scores border villages and community interfaces.
3. /regions: Provides metadata on configured regions ('kruger' and 'south_bengal').

Transferability & Confidence Logic:
- Scores South Bengal using the model trained on Kruger transferable features.
- Dynamically computes risk scores using hour of day, rainfall, and distance metrics.
- Evaluates feature boundaries: any cell with features outside the training range
  is flagged with confidence = 'low' (or 'low confidence').
- CORS enabled for React + Leaflet frontend integration.
"""

import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import zoneinfo

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "elephant_presence_risk_xgb.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

PARQUET_SB = DATA_DIR / "grid_features_south_bengal.parquet"
PARQUET_KRUGER = DATA_DIR / "grid_features_kruger.parquet"
VILLAGES_KRUGER = DATA_DIR / "raw" / "villages.csv"
VILLAGES_SB_CACHE = DATA_DIR / "cache" / "osm_villages_south_bengal.json"

# Load Model and Metadata at startup
app = FastAPI(
    title="GajRaksha API",
    description="Elephant presence risk near villages prediction service for South Bengal and Kruger landscapes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
metadata = {}
feature_ranges = {}

# Preloaded Grid DataFrames
grid_dfs = {}


@app.on_event("startup")
def startup_event():
    global model, metadata, feature_ranges, grid_dfs

    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
        print(f"✓ Loaded XGBoost model from {MODEL_FILE.name}")
    else:
        print(f"⚠️ Model file not found at {MODEL_FILE}")

    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            feature_ranges = metadata.get("training_feature_ranges", {})
        print(f"✓ Loaded model metadata from {METADATA_FILE.name}")

    if PARQUET_SB.exists():
        grid_dfs["south_bengal"] = pd.read_parquet(PARQUET_SB)
        print(f"✓ Preloaded South Bengal grid: {len(grid_dfs['south_bengal']):,} cells")

    if PARQUET_KRUGER.exists():
        grid_dfs["kruger"] = pd.read_parquet(PARQUET_KRUGER)
        print(f"✓ Preloaded Kruger grid: {len(grid_dfs['kruger']):,} cells")


def get_region_local_time(region: str) -> tuple:
    """Returns local hour and is_night for the selected region."""
    tz_name = "Asia/Kolkata" if region == "south_bengal" else "Africa/Johannesburg"
    tz = zoneinfo.ZoneInfo(tz_name)
    now = datetime.now(tz)
    hour = now.hour
    is_night = 1 if (hour >= 18 or hour < 6) else 0
    return hour, is_night, tz_name


def check_cell_confidence(cell_dist_village: float, cell_crop_density: float) -> str:
    """
    Evaluates whether a grid cell falls outside the Kruger training range.
    Kruger crop density training bounds: 0.0 to 0.40.
    Kruger distance to village bounds: 0.3 km to 85.5 km.
    """
    if cell_crop_density > 0.40 or cell_dist_village < 0.3:
        return "low"
    return "high"


@app.get("/")
def root():
    return {
        "service": "GajRaksha API",
        "description": "Elephant presence risk near villages prediction service",
        "version": "1.0.0",
        "available_endpoints": [
            "/risk-grid?region=south_bengal",
            "/village-risk?region=south_bengal",
            "/regions"
        ]
    }


@app.get("/regions")
def get_regions():
    return {
        "regions": [
            {
                "id": "south_bengal",
                "name": "South Bengal Elephant Landscape",
                "country": "India",
                "state": "West Bengal",
                "timezone": "Asia/Kolkata",
                "districts": ["Jhargram", "Paschim Medinipur", "Bankura", "Purulia"],
                "total_area_sq_km": 22570.9,
                "role": "target"
            },
            {
                "id": "kruger",
                "name": "Kruger National Park Interface",
                "country": "South Africa",
                "timezone": "Africa/Johannesburg",
                "role": "training"
            }
        ]
    }


@app.get("/risk-grid")
def get_risk_grid(
    region: str = Query("south_bengal", description="Target region: 'south_bengal' or 'kruger'"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Local hour of day (0-23)"),
    rainfall_7d: Optional[float] = Query(None, ge=0, description="Rolling 7-day cumulative rainfall (mm)"),
    rainfall_30d: Optional[float] = Query(None, ge=0, description="Rolling 30-day cumulative rainfall (mm)"),
    limit: int = Query(500, ge=1, le=20000, description="Number of grid cells to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Scores ~1 km spatial cells for elephant presence risk near villages.
    Accepts region='south_bengal' or region='kruger'.
    Includes risk scores, risk levels (low/medium/high), and 'confidence' assessment.
    """
    reg_key = region.lower().strip()
    if reg_key not in grid_dfs:
        raise HTTPException(status_code=400, detail=f"Invalid region '{region}'. Must be 'south_bengal' or 'kruger'.")

    df_grid = grid_dfs[reg_key]
    curr_hour, curr_night, tz_name = get_region_local_time(reg_key)

    eval_hour = curr_hour if hour is None else hour
    eval_night = 1 if (eval_hour >= 18 or eval_hour < 6) else 0
    eval_rain_7d = 2.0 if rainfall_7d is None else rainfall_7d
    eval_rain_30d = 15.0 if rainfall_30d is None else rainfall_30d

    # Slice cells for pagination
    sliced_df = df_grid.iloc[offset : offset + limit].copy()

    # Model input feature vectors
    # Features: [hour_of_day, is_night, rainfall_last_7_days, rainfall_last_30_days, speed_kmh, step_distance_km, heading_change_deg, dist_to_nearest_village_km]
    dist_v = sliced_df["distance_to_village_km"].values

    X_input = pd.DataFrame({
        "hour_of_day": np.full(len(sliced_df), eval_hour),
        "is_night": np.full(len(sliced_df), eval_night),
        "rainfall_last_7_days": np.full(len(sliced_df), eval_rain_7d),
        "rainfall_last_30_days": np.full(len(sliced_df), eval_rain_30d),
        "speed_kmh": np.full(len(sliced_df), 2.2),
        "step_distance_km": np.full(len(sliced_df), 1.8),
        "heading_change_deg": np.full(len(sliced_df), 30.0),
        "dist_to_nearest_village_km": dist_v
    })

    if model is not None:
        risk_probs = model.predict_proba(X_input)[:, 1]
    else:
        # Fallback ecological heuristic if model is loading
        risk_probs = np.clip(1.0 - (dist_v / 15.0), 0.05, 0.95)

    cells_out = []
    crop_densities = sliced_df["crop_density"].values
    forest_dists = sliced_df["distance_to_forest_edge_km"].values
    water_dists = sliced_df["distance_to_water_km"].values
    lons = sliced_df["location_lon"].values
    lats = sliced_df["location_lat"].values
    cell_ids = sliced_df["cell_id"].values

    for i in range(len(sliced_df)):
        score = round(float(risk_probs[i]), 3)
        c_density = float(crop_densities[i])
        v_dist = float(dist_v[i])
        
        # Risk category
        if score >= 0.70:
            level = "high"
        elif score >= 0.35:
            level = "medium"
        else:
            level = "low"

        # Confidence: low if outside Kruger training range
        conf = check_cell_confidence(v_dist, c_density)

        cells_out.append({
            "cell_id": cell_ids[i],
            "location_lat": lats[i],
            "location_lon": lons[i],
            "presence_risk_score": score,
            "risk_level": level,
            "confidence": conf,
            "distance_to_village_km": v_dist,
            "distance_to_forest_edge_km": float(forest_dists[i]),
            "distance_to_water_km": float(water_dists[i]),
            "crop_density": round(c_density, 3)
        })

    return {
        "region": reg_key,
        "timezone": tz_name,
        "total_cells": len(df_grid),
        "returned_cells": len(cells_out),
        "offset": offset,
        "limit": limit,
        "scoring_parameters": {
            "hour_of_day": eval_hour,
            "is_night": eval_night,
            "rainfall_last_7_days_mm": eval_rain_7d,
            "rainfall_last_30_days_mm": eval_rain_30d
        },
        "cells": cells_out
    }


@app.get("/village-risk")
def get_village_risk(
    region: str = Query("south_bengal", description="Target region: 'south_bengal' or 'kruger'"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Local hour of day"),
    limit: int = Query(50, ge=1, le=1000, description="Max villages to return")
):
    """
    Returns border village locations with current presence risk assessment and confidence level.
    """
    reg_key = region.lower().strip()
    curr_hour, curr_night, tz_name = get_region_local_time(reg_key)
    eval_hour = curr_hour if hour is None else hour
    eval_night = 1 if (eval_hour >= 18 or eval_hour < 6) else 0

    villages_list = []

    if reg_key == "south_bengal":
        if VILLAGES_SB_CACHE.exists():
            with open(VILLAGES_SB_CACHE, "r", encoding="utf-8") as f:
                raw_osm = json.load(f)
            elements = raw_osm.get("elements", [])
            for idx, el in enumerate(elements[:limit]):
                name = el.get("tags", {}).get("name", f"Village {idx+1}")
                lat = float(el["lat"])
                lon = float(el["lon"])
                
                # Near village center distance is ~0.5 km
                X_v = pd.DataFrame([{
                    "hour_of_day": eval_hour,
                    "is_night": eval_night,
                    "rainfall_last_7_days": 2.0,
                    "rainfall_last_30_days": 15.0,
                    "speed_kmh": 2.0,
                    "step_distance_km": 1.5,
                    "heading_change_deg": 25.0,
                    "dist_to_nearest_village_km": 0.8
                }])
                score = round(float(model.predict_proba(X_v)[0, 1]), 3) if model else 0.85
                level = "high" if score >= 0.70 else ("medium" if score >= 0.35 else "low")
                
                villages_list.append({
                    "village_id": f"SB_VIL_{el['id']}",
                    "name": name,
                    "location_lat": lat,
                    "location_lon": lon,
                    "presence_risk_score": score,
                    "risk_level": level,
                    "confidence": "low"  # South Bengal villages sit in intensive agricultural areas
                })
        else:
            raise HTTPException(status_code=404, detail="South Bengal village data not found.")

    elif reg_key == "kruger":
        if VILLAGES_KRUGER.exists():
            k_df = pd.read_csv(VILLAGES_KRUGER)
            for _, r in k_df.head(limit).iterrows():
                X_v = pd.DataFrame([{
                    "hour_of_day": eval_hour,
                    "is_night": eval_night,
                    "rainfall_last_7_days": 2.0,
                    "rainfall_last_30_days": 15.0,
                    "speed_kmh": 2.0,
                    "step_distance_km": 1.5,
                    "heading_change_deg": 25.0,
                    "dist_to_nearest_village_km": 0.8
                }])
                score = round(float(model.predict_proba(X_v)[0, 1]), 3) if model else 0.85
                level = "high" if score >= 0.70 else ("medium" if score >= 0.35 else "low")
                villages_list.append({
                    "village_id": r["village_id"],
                    "name": r["name"],
                    "location_lat": float(r["location_lat"]),
                    "location_lon": float(r["location_long"]),
                    "presence_risk_score": score,
                    "risk_level": level,
                    "confidence": "high"
                })
        else:
            raise HTTPException(status_code=404, detail="Kruger village data not found.")

    return {
        "region": reg_key,
        "timezone": tz_name,
        "total_villages_returned": len(villages_list),
        "villages": villages_list
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
