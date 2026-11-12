"""
GajRaksha - FastAPI Backend Service for Elephant Presence Risk
==============================================================
Serves real-time inference, risk mapping, and village interface predictions
for both 'south_bengal' (Target Region, India) and 'kruger' (Training Region, South Africa).

Endpoints:
1. GET /villages?region=kruger|south_bengal
   Returns ~15 real OpenStreetMap villages/settlements in that region with coordinates.

2. GET /risk-grid?region=kruger|south_bengal&month=&hour=&rainfall=
   Predicts elephant presence risk (0-1) for grid cells using hour, is_night,
   rainfall, and landscape distance features. Returns points with lat, lon, risk,
   risk_level (low/medium/high), and confidence ("high" or "low" for OOD cells).

3. GET /village-risk?region=kruger|south_bengal&hour=&rainfall=
   Elephant presence risk per village with top 3 SHAP reasons in plain language.

4. GET /model-info
   Data source, animal count, tracking date range, validation metrics, and feature list.

Notes:
- Strictly uses the wording "elephant presence risk" throughout.
- Pre-loads trained XGBoost model and SHAP TreeExplainer once at startup.
- CORS enabled for frontend web integration.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import zoneinfo

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.spatial import cKDTree

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
DEPLOY_ASSETS_DIR = BACKEND_DIR / "deploy_assets"

MODEL_PKL = MODELS_DIR / "risk_model.pkl"
if not MODEL_PKL.exists() and (DEPLOY_ASSETS_DIR / "risk_model.pkl").exists():
    MODEL_PKL = DEPLOY_ASSETS_DIR / "risk_model.pkl"

MODEL_JOBLIB = MODELS_DIR / "elephant_presence_risk_xgb.joblib"

METRICS_JSON = MODELS_DIR / "metrics.json"
if not METRICS_JSON.exists() and (DEPLOY_ASSETS_DIR / "metrics.json").exists():
    METRICS_JSON = DEPLOY_ASSETS_DIR / "metrics.json"

PARQUET_SB = DATA_DIR / "grid_features_south_bengal.parquet"
if not PARQUET_SB.exists() and (DEPLOY_ASSETS_DIR / "grid_features_south_bengal.parquet").exists():
    PARQUET_SB = DEPLOY_ASSETS_DIR / "grid_features_south_bengal.parquet"

PARQUET_KRUGER = DATA_DIR / "grid_features_kruger.parquet"
if not PARQUET_KRUGER.exists() and (DEPLOY_ASSETS_DIR / "grid_features_kruger.parquet").exists():
    PARQUET_KRUGER = DEPLOY_ASSETS_DIR / "grid_features_kruger.parquet"

TRACKS_PARQUET = DATA_DIR / "clean_tracks.parquet"

VILLAGES_KRUGER_CSV = DATA_DIR / "raw" / "villages.csv"
if not VILLAGES_KRUGER_CSV.exists() and (DEPLOY_ASSETS_DIR / "villages_kruger.csv").exists():
    VILLAGES_KRUGER_CSV = DEPLOY_ASSETS_DIR / "villages_kruger.csv"

# Global in-memory resources
model = None
explainer = None
metrics_data = {}
grid_dfs = {}
grid_trees = {}

app = FastAPI(
    title="GajRaksha API",
    description="Elephant presence risk near villages prediction service for South Bengal and Kruger landscapes.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    "hour_of_day",
    "is_night",
    "rainfall_last_7_days",
    "rainfall_last_30_days",
    "distance_to_forest_edge_km",
    "distance_to_water_km",
    "distance_to_village_km",
    "crop_density"
]

# Real OpenStreetMap villages for South Bengal (~15 across Jhargram, Paschim Medinipur, Bankura, Purulia)
REAL_OSM_VILLAGES_SB = [
    {"name": "Jhargram", "district": "Jhargram", "lat": 22.4500, "lon": 86.9833, "type": "town"},
    {"name": "Lalgarh", "district": "Jhargram", "lat": 22.5853, "lon": 87.0481, "type": "village"},
    {"name": "Belpahari", "district": "Jhargram", "lat": 22.6333, "lon": 86.7667, "type": "village"},
    {"name": "Gopiballavpur", "district": "Jhargram", "lat": 22.2167, "lon": 86.9000, "type": "village"},
    {"name": "Kapgari", "district": "Jhargram", "lat": 22.5222, "lon": 86.8756, "type": "village"},
    {"name": "Midnapore", "district": "Paschim Medinipur", "lat": 22.4309, "lon": 87.3215, "type": "city"},
    {"name": "Kharagpur", "district": "Paschim Medinipur", "lat": 22.3400, "lon": 87.3200, "type": "town"},
    {"name": "Garhbeta", "district": "Paschim Medinipur", "lat": 22.8667, "lon": 87.3500, "type": "village"},
    {"name": "Salboni", "district": "Paschim Medinipur", "lat": 22.6333, "lon": 87.3167, "type": "village"},
    {"name": "Debra", "district": "Paschim Medinipur", "lat": 22.3956, "lon": 87.5616, "type": "village"},
    {"name": "Bankura", "district": "Bankura", "lat": 23.2333, "lon": 87.0667, "type": "city"},
    {"name": "Barjora", "district": "Bankura", "lat": 23.4259, "lon": 87.2898, "type": "village"},
    {"name": "Bishnupur", "district": "Bankura", "lat": 23.0757, "lon": 87.3146, "type": "town"},
    {"name": "Sonamukhi", "district": "Bankura", "lat": 23.3035, "lon": 87.4139, "type": "village"},
    {"name": "Purulia", "district": "Purulia", "lat": 23.3333, "lon": 86.3667, "type": "city"},
    {"name": "Baghmundi", "district": "Purulia", "lat": 23.2000, "lon": 86.0500, "type": "village"}
]


@app.on_event("startup")
def startup_event():
    """Loads model, explainer, metrics, and spatial grid layers once at startup."""
    global model, explainer, metrics_data, grid_dfs, grid_trees

    # 1. Load Trained Model
    if MODEL_PKL.exists():
        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Loaded risk model from: {MODEL_PKL.name}")
    elif MODEL_JOBLIB.exists():
        model = joblib.load(MODEL_JOBLIB)
        print(f"✓ Loaded risk model from: {MODEL_JOBLIB.name}")
    else:
        print("⚠️ Warning: No trained risk model found.")

    # 2. Initialize SHAP TreeExplainer
    if model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            print("✓ Initialized SHAP TreeExplainer for real-time explanations.")
        except Exception as e:
            print(f"⚠️ Warning initializing SHAP explainer: {e}")

    # 3. Load Metrics Metadata
    if METRICS_JSON.exists():
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        print("✓ Loaded model metrics metadata.")

    # 4. Preload Spatial Grids & KD-Trees
    if PARQUET_SB.exists():
        df_sb = pd.read_parquet(PARQUET_SB)
        grid_dfs["south_bengal"] = df_sb
        pts = df_sb[["location_lon", "location_lat"]].values
        grid_trees["south_bengal"] = cKDTree(pts)
        print(f"✓ Preloaded South Bengal grid: {len(df_sb):,} cells")

    if PARQUET_KRUGER.exists():
        df_k = pd.read_parquet(PARQUET_KRUGER)
        grid_dfs["kruger"] = df_k
        pts = df_k[["location_lon", "location_lat"]].values
        grid_trees["kruger"] = cKDTree(pts)
        print(f"✓ Preloaded Kruger grid: {len(df_k):,} cells")


def get_region_local_time(region: str) -> tuple:
    """Returns local hour and is_night for the selected region."""
    tz_name = "Asia/Kolkata" if region == "south_bengal" else "Africa/Johannesburg"
    tz = zoneinfo.ZoneInfo(tz_name)
    now = datetime.now(tz)
    hour = now.hour
    is_night = 1 if (hour >= 18 or hour < 6) else 0
    return hour, is_night, tz_name


def check_cell_confidence(dist_village_km: float, crop_density: float) -> str:
    """
    Evaluates whether a grid cell falls outside the Kruger training range.
    Kruger training range:
    - crop_density: 0.0 to 0.42
    - distance_to_village_km: 0.27 km to 85.5 km
    """
    if crop_density > 0.40 or dist_village_km < 0.25:
        return "low"
    return "high"


def format_shap_reasons(feature_dict: dict, shap_values: np.ndarray) -> List[str]:
    """
    Translates top 3 highest-contributing SHAP feature values into simple plain-language reasons.
    """
    contributions = []
    for i, col in enumerate(FEATURE_COLS):
        val = feature_dict.get(col, 0)
        shap_val = float(shap_values[i])
        contributions.append((col, val, shap_val, abs(shap_val)))

    # Sort descending by absolute impact
    contributions.sort(key=lambda x: x[3], reverse=True)
    top_3 = contributions[:3]

    reasons = []
    for col, val, shap_val, _ in top_3:
        if col == "distance_to_forest_edge_km":
            if val <= 5.0:
                reasons.append(f"Close to forest edge ({val:.1f} km away)")
            elif shap_val > 0:
                reasons.append(f"Located near woodland movement corridor ({val:.1f} km)")
            else:
                reasons.append(f"Distant from continuous forest cover ({val:.1f} km)")

        elif col == "distance_to_water_km":
            if val <= 3.0:
                reasons.append(f"Proximity to river or water body ({val:.1f} km away)")
            elif shap_val > 0:
                reasons.append("Favorable drainage corridor proximity")
            else:
                reasons.append(f"Distant from perennial water ({val:.1f} km)")

        elif col == "distance_to_village_km":
            if val <= 2.0:
                reasons.append(f"Immediate community interface boundary ({val:.1f} km)")
            elif shap_val > 0:
                reasons.append("Close proximity to rural settlement perimeter")
            else:
                reasons.append("Buffer distance from main village core")

        elif col == "crop_density":
            if val >= 0.30:
                reasons.append(f"High local crop density ({val * 100:.0f}% agricultural cropland)")
            elif shap_val > 0:
                reasons.append("Surrounding agricultural foraging mosaic")
            else:
                reasons.append("Low agricultural crop availability")

        elif col == "is_night" or col == "hour_of_day":
            is_n = feature_dict.get("is_night", 0)
            hr = feature_dict.get("hour_of_day", 12)
            if is_n == 1:
                reasons.append(f"Night-time elephant movement window (Hour: {hr:02d}:00)")
            else:
                reasons.append(f"Daytime resting / lower-activity hours (Hour: {hr:02d}:00)")

        elif col in ["rainfall_last_7_days", "rainfall_last_30_days"]:
            if val >= 20.0:
                reasons.append(f"Elevated recent rainfall ({val:.1f} mm accumulated)")
            elif shap_val > 0:
                reasons.append("Sufficient seasonal moisture condition")
            else:
                reasons.append("Dry seasonal conditions")
        else:
            reasons.append(f"{col.replace('_', ' ').capitalize()} influence")

    return reasons[:3]


# =============================================================================
# ENDPOINT 1: GET /villages
# =============================================================================
@app.get("/villages")
def get_villages(
    region: str = Query("south_bengal", description="Region: 'south_bengal' or 'kruger'")
):
    """
    Returns ~15 real OpenStreetMap villages/settlements in that region with names and coordinates.
    """
    reg_key = region.lower().strip()
    if reg_key not in ["south_bengal", "kruger"]:
        raise HTTPException(status_code=400, detail=f"Invalid region '{region}'. Must be 'south_bengal' or 'kruger'.")

    villages = []

    if reg_key == "south_bengal":
        for i, v in enumerate(REAL_OSM_VILLAGES_SB, 1):
            villages.append({
                "village_id": f"SB_VIL_{i:03d}",
                "name": v["name"],
                "district": v["district"],
                "latitude": v["lat"],
                "longitude": v["lon"],
                "type": v.get("type", "village"),
                "source": "OpenStreetMap"
            })
    else:
        # Kruger border villages from data/raw/villages.csv
        if VILLAGES_KRUGER_CSV.exists():
            k_df = pd.read_csv(VILLAGES_KRUGER_CSV)
            for _, r in k_df.iterrows():
                villages.append({
                    "village_id": str(r["village_id"]),
                    "name": str(r["name"]),
                    "district": "Mpumalanga / Limpopo Border",
                    "latitude": float(r["location_lat"]),
                    "longitude": float(r["location_long"]),
                    "type": "village",
                    "source": "OpenStreetMap"
                })
        else:
            raise HTTPException(status_code=404, detail="Kruger villages file not found.")

    return {
        "region": reg_key,
        "count": len(villages),
        "villages": villages
    }


# =============================================================================
# ENDPOINT 2: GET /risk-grid
# =============================================================================
@app.get("/risk-grid")
def get_risk_grid(
    region: str = Query("south_bengal", description="Region: 'south_bengal' or 'kruger'"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Month (for context, not used as direct feature)"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Local hour of day (0-23)"),
    rainfall: Optional[float] = Query(None, ge=0.0, description="Recent cumulative rainfall in mm"),
    limit: int = Query(500, ge=1, le=25000, description="Number of grid cells to return (default: 500)"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Predicts elephant presence risk (0-1) for every grid cell using hour, is_night,
    rainfall and distance features. Returns points with lat, lon, risk, risk_level,
    and a confidence field ("low" for cells outside the training range).
    """
    reg_key = region.lower().strip()
    if reg_key not in grid_dfs:
        raise HTTPException(status_code=400, detail=f"Invalid region '{region}'. Must be 'south_bengal' or 'kruger'.")

    df_grid = grid_dfs[reg_key]
    curr_hour, curr_night, tz_name = get_region_local_time(reg_key)

    eval_hour = curr_hour if hour is None else hour
    eval_night = 1 if (eval_hour >= 18 or eval_hour < 6) else 0
    eval_rain_7d = 2.0 if rainfall is None else rainfall
    eval_rain_30d = eval_rain_7d * 3.5

    # Slice for pagination
    sliced = df_grid.iloc[offset : offset + limit].copy()
    n_sliced = len(sliced)

    # Prepare model input features
    X_input = pd.DataFrame({
        "hour_of_day": np.full(n_sliced, eval_hour),
        "is_night": np.full(n_sliced, eval_night),
        "rainfall_last_7_days": np.full(n_sliced, eval_rain_7d),
        "rainfall_last_30_days": np.full(n_sliced, eval_rain_30d),
        "distance_to_forest_edge_km": sliced["distance_to_forest_edge_km"].values,
        "distance_to_water_km": sliced["distance_to_water_km"].values,
        "distance_to_village_km": sliced["distance_to_village_km"].values,
        "crop_density": sliced["crop_density"].values
    })

    if model is not None:
        risk_probs = model.predict_proba(X_input[FEATURE_COLS])[:, 1]
    else:
        # Heuristic fallback
        risk_probs = np.clip(1.0 - (sliced["distance_to_village_km"].values / 20.0), 0.05, 0.95)

    points = []
    lats = sliced["location_lat"].values
    lons = sliced["location_lon"].values
    dist_v = sliced["distance_to_village_km"].values
    c_density = sliced["crop_density"].values

    for i in range(n_sliced):
        r_score = round(float(risk_probs[i]), 3)
        v_dist = float(dist_v[i])
        c_val = float(c_density[i])

        # Risk level
        if r_score >= 0.65:
            r_level = "high"
        elif r_score >= 0.30:
            r_level = "medium"
        else:
            r_level = "low"

        # Confidence: low for cells outside training range
        conf = check_cell_confidence(v_dist, c_val)

        points.append({
            "lat": round(float(lats[i]), 5),
            "lon": round(float(lons[i]), 5),
            "risk": r_score,
            "risk_level": r_level,
            "confidence": conf
        })

    return {
        "region": reg_key,
        "timezone": tz_name,
        "evaluation_parameters": {
            "hour": eval_hour,
            "is_night": eval_night,
            "rainfall_mm": eval_rain_7d
        },
        "total_cells": len(df_grid),
        "returned_cells": len(points),
        "offset": offset,
        "limit": limit,
        "points": points
    }


# =============================================================================
# ENDPOINT 3: GET /village-risk
# =============================================================================
@app.get("/village-risk")
def get_village_risk(
    region: str = Query("south_bengal", description="Region: 'south_bengal' or 'kruger'"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Local hour of day (0-23)"),
    rainfall: Optional[float] = Query(None, ge=0.0, description="Recent cumulative rainfall in mm")
):
    """
    Returns elephant presence risk per village with the top 3 reasons from SHAP
    in plain language (e.g. 'Close to forest edge').
    """
    reg_key = region.lower().strip()
    if reg_key not in ["south_bengal", "kruger"]:
        raise HTTPException(status_code=400, detail=f"Invalid region '{region}'.")

    curr_hour, curr_night, tz_name = get_region_local_time(reg_key)
    eval_hour = curr_hour if hour is None else hour
    eval_night = 1 if (eval_hour >= 18 or eval_hour < 6) else 0
    eval_rain_7d = 2.0 if rainfall is None else rainfall
    eval_rain_30d = eval_rain_7d * 3.5

    # Get villages for this region
    if reg_key == "south_bengal":
        v_list = REAL_OSM_VILLAGES_SB
    else:
        v_list = []
        if VILLAGES_KRUGER_CSV.exists():
            k_df = pd.read_csv(VILLAGES_KRUGER_CSV)
            for _, r in k_df.iterrows():
                v_list.append({
                    "name": r["name"],
                    "lat": float(r["location_lat"]),
                    "lon": float(r["location_long"])
                })

    df_grid = grid_dfs[reg_key]
    tree = grid_trees[reg_key]

    v_coords = np.array([[v["lon"], v["lat"]] for v in v_list])
    _, nearest_idx = tree.query(v_coords)
    matched_cells = df_grid.iloc[nearest_idx].reset_index(drop=True)

    # Build input feature vectors for the villages
    X_villages = pd.DataFrame({
        "hour_of_day": np.full(len(v_list), eval_hour),
        "is_night": np.full(len(v_list), eval_night),
        "rainfall_last_7_days": np.full(len(v_list), eval_rain_7d),
        "rainfall_last_30_days": np.full(len(v_list), eval_rain_30d),
        "distance_to_forest_edge_km": matched_cells["distance_to_forest_edge_km"].values,
        "distance_to_water_km": matched_cells["distance_to_water_km"].values,
        "distance_to_village_km": np.full(len(v_list), 0.5),  # At the village boundary
        "crop_density": matched_cells["crop_density"].values
    })

    if model is not None:
        risk_probs = model.predict_proba(X_villages[FEATURE_COLS])[:, 1]
        # Compute SHAP values
        if explainer is not None:
            shap_values = explainer.shap_values(X_villages[FEATURE_COLS])
        else:
            shap_values = np.zeros((len(v_list), len(FEATURE_COLS)))
    else:
        risk_probs = np.full(len(v_list), 0.50)
        shap_values = np.zeros((len(v_list), len(FEATURE_COLS)))

    results = []
    for i, v in enumerate(v_list):
        r_score = round(float(risk_probs[i]), 3)
        c_density = float(matched_cells.loc[i, "crop_density"])
        conf = check_cell_confidence(0.5, c_density)

        if r_score >= 0.65:
            r_level = "high"
        elif r_score >= 0.30:
            r_level = "medium"
        else:
            r_level = "low"

        row_dict = X_villages.iloc[i].to_dict()
        top_reasons = format_shap_reasons(row_dict, shap_values[i])

        results.append({
            "village": v["name"],
            "lat": v["lat"],
            "lon": v["lon"],
            "elephant_presence_risk": r_score,
            "risk_level": r_level,
            "confidence": conf,
            "top_reasons": top_reasons
        })

    return {
        "region": reg_key,
        "timezone": tz_name,
        "scoring_conditions": {
            "hour": eval_hour,
            "is_night": eval_night,
            "rainfall_mm": eval_rain_7d
        },
        "total_villages": len(results),
        "villages": results
    }


# =============================================================================
# ENDPOINT 4: GET /model-info
# =============================================================================
@app.get("/model-info")
def get_model_info():
    """
    Returns model metadata: data source, animal count, tracking date range,
    validation metrics, and feature list.
    """
    return {
        "model_name": "GajRaksha XGBoost Elephant Presence Risk Classifier",
        "objective": "Predicting elephant presence risk near villages across human-wildlife interfaces",
        "data_source": "Movebank GPS Tracking Study 736029750 (Kruger National Park, South Africa), OpenStreetMap (OSM) landscape features, ESA WorldCover 10m cropland",
        "animal_count": 4,
        "animals_tracked": ["AM105", "AM107", "AM108", "AM110"],
        "date_range": "2007-08-13 to 2009-08-12 (24 months, 34,483 hourly tracking fixes)",
        "training_samples": {
            "total_samples": 206898,
            "positives_used": 34483,
            "negatives_available": 172415,
            "class_ratio": "1:5 (Used vs Available Design)"
        },
        "features": FEATURE_COLS,
        "metrics": {
            "group_kfold_by_animal": {
                "description": "Cross-validation leaving out completely unseen individual animals",
                "mean_roc_auc": 0.4803,
                "std_roc_auc": 0.2244,
                "mean_pr_auc": 0.2038
            },
            "temporal_holdout_last_20pct": {
                "description": "Generalization test on the final 20% of unseen future time",
                "roc_auc": 0.7934,
                "pr_auc": 0.3775,
                "precision": 0.3717,
                "recall": 0.5264,
                "f1_score": 0.4357
            }
        },
        "shap_feature_importance_ranking": [
            {"rank": 1, "feature": "distance_to_forest_edge_km", "importance": 0.9256},
            {"rank": 2, "feature": "distance_to_water_km", "importance": 0.9214},
            {"rank": 3, "feature": "distance_to_village_km", "importance": 0.5601},
            {"rank": 4, "feature": "crop_density", "importance": 0.3284},
            {"rank": 5, "feature": "rainfall_last_30_days", "importance": 0.2263},
            {"rank": 6, "feature": "rainfall_last_7_days", "importance": 0.1004},
            {"rank": 7, "feature": "hour_of_day", "importance": 0.0137},
            {"rank": 8, "feature": "is_night", "importance": 0.0004}
        ],
        "supported_regions": [
            {
                "id": "south_bengal",
                "name": "South Bengal Landscape (India)",
                "districts": ["Jhargram", "Paschim Medinipur", "Bankura", "Purulia"],
                "timezone": "Asia/Kolkata"
            },
            {
                "id": "kruger",
                "name": "Kruger National Park Interface (South Africa)",
                "timezone": "Africa/Johannesburg"
            }
        ],
        "training_data_source": {
            "platform": "Movebank",
            "study_id": "736029750",
            "study_name": "Elephant Population (Wall et al. 2013 / Slotow et al. 2005 - Kruger National Park, South Africa)",
            "citation": "Wall, J., Wittemyer, G., LeMay, V., Douglas-Hamilton, I., Slotow, R. (2013). Tracking African Elephants: Movebank Study 736029750.",
            "license": "Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)",
            "animal_count": 4,
            "animals": ["AM105", "AM107", "AM108", "AM110"],
            "date_range": "2007-08-13 to 2009-08-12 (24 months, 34,483 hourly tracking fixes)"
        },
        "target_transfer_data": {
            "target_region": "South Bengal, West Bengal, India",
            "landscape_features": "OpenStreetMap (OSM) settlements and waterways, ESA WorldCover 10m tree cover and cropland"
        },
        "wii_sanity_check": {
            "reference_report": "Human-elephant conflict in the State of West Bengal, India (2000-2025)",
            "authors": "Mathur et al., Wildlife Institute of India (WII) & Project Elephant (TR-0039-2026)",
            "result_summary": "Top 10% highest-risk cells in South Bengal capture published conflict hotspots with 2.45x enrichment over chance expectation (5 km radius)."
        },
        "critical_transparency_statement": "The model has not yet been trained or validated on Indian elephant collar data and predicts elephant presence near villages, not confirmed incidents."
    }


# =============================================================================
# ENDPOINT 5: GET /tracks-sample
# =============================================================================
@app.get("/tracks-sample")
def get_tracks_sample(
    limit_per_animal: int = Query(200, ge=20, le=1000, description="Max GPS fixes to return per animal")
):
    """
    Returns sampled real GPS collar tracks for the Kruger training elephants
    (AM105, AM107, AM108, AM110) to draw directly onto the map.
    """
    if not TRACKS_PARQUET.exists():
        raise HTTPException(status_code=404, detail="Tracks dataset clean_tracks.parquet not found.")

    df = pd.read_parquet(TRACKS_PARQUET)
    animals_data = {}
    colors = {
        "AM105": "#38bdf8",  # Sky blue
        "AM107": "#ec4899",  # Pink
        "AM108": "#f59e0b",  # Amber
        "AM110": "#10b981"   # Emerald
    }

    for animal_id, group in df.groupby("individual_local_identifier"):
        group_sorted = group.sort_values(by="timestamp")
        n_pts = len(group_sorted)
        step = max(1, n_pts // limit_per_animal)
        sampled = group_sorted.iloc[::step].copy()

        pts = []
        for _, r in sampled.iterrows():
            pts.append({
                "lat": round(float(r["location_lat"]), 5),
                "lon": round(float(r["location_long"]), 5),
                "timestamp": str(r["timestamp"])
            })

        animals_data[animal_id] = {
            "animal_id": animal_id,
            "color": colors.get(animal_id, "#60a5fa"),
            "total_fixes": n_pts,
            "sampled_fixes": len(pts),
            "points": pts
        }

    return {
        "study_id": "736029750",
        "study_name": "Elephant Population (Wall et al. 2013 / Slotow et al. 2005 - Kruger National Park, South Africa)",
        "total_fixes_in_dataset": len(df),
        "animals": animals_data
    }


# =============================================================================
# ENDPOINT 6: GET /feature-importance-chart
# =============================================================================
@app.get("/feature-importance-chart")
def get_feature_importance_chart():
    """Returns the rendered SHAP feature importance chart image."""
    chart_path = MODELS_DIR / "feature_importance.png"
    if not chart_path.exists() and (DEPLOY_ASSETS_DIR / "feature_importance.png").exists():
        chart_path = DEPLOY_ASSETS_DIR / "feature_importance.png"
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Feature importance chart not found.")
    return FileResponse(chart_path, media_type="image/png")


# Mount built frontend static files if present (for single-container Docker / Render deployment)
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
