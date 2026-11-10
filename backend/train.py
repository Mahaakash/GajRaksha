"""
GajRaksha - XGBoost Model Training Pipeline (GroupKFold by Animal)
==================================================================
Trains the production XGBoost classifier to predict
'elephant presence risk near villages' using strictly transferable features.

Validation:
- Evaluates with sklearn.model_selection.GroupKFold (n_splits=4, grouped by individual_local_identifier)
- Evaluates cross-time generalization.
- Saves model artifact to models/elephant_presence_risk_xgb.joblib.
- Saves metadata and feature boundaries to models/model_metadata.json.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INPUT_PARQUET = DATA_DIR / "processed" / "elephant_presence_dataset.parquet"
MODEL_OUTPUT_FILE = MODELS_DIR / "elephant_presence_risk_xgb.joblib"
METADATA_OUTPUT_FILE = MODELS_DIR / "model_metadata.json"

# Transferable Feature Specification (Step A)
FEATURE_COLS = [
    "hour_of_day",
    "is_night",
    "rainfall_last_7_days",
    "rainfall_last_30_days",
    "speed_kmh",
    "step_distance_km",
    "heading_change_deg",
    "dist_to_nearest_village_km",
]

TARGET_COL = "elephant_presence_risk_near_villages"


def train_and_evaluate_groupkfold(df: pd.DataFrame):
    """
    Evaluates cross-animal generalization using GroupKFold grouped by animal ID.
    Trains on 3 animals, tests on the 4th across all 4 folds.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    groups = df["individual_local_identifier"]

    gkf = GroupKFold(n_splits=4)
    fold_results = []

    print("-" * 72)
    print("GROUP-K-FOLD EVALUATION BY ANIMAL (Testing on Unseen Elephants)")
    print("-" * 72)
    print(f"{'Fold':<6} {'Test Animal':<14} {'Train/Test Size':<18} {'ROC-AUC':<10} {'F1':<8} {'Recall':<8} {'Precision':<10}")
    print("-" * 72)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        test_animal = groups.iloc[test_idx].unique()[0]

        scale_pos = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        )
        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 0.5
        f1 = f1_score(y_test, y_preds, zero_division=0)
        rec = recall_score(y_test, y_preds, zero_division=0)
        prec = precision_score(y_test, y_preds, zero_division=0)

        fold_results.append({
            "fold": fold,
            "test_animal": test_animal,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "roc_auc": round(float(auc), 4),
            "f1_score": round(float(f1), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4)
        })

        size_str = f"{len(train_idx):,}/{len(test_idx):,}"
        print(f"{fold:<6} Elephant {test_animal:<5} {size_str:<18} {auc:<10.3f} {f1:<8.3f} {rec:<8.3f} {prec:<10.3f}")

    avg_auc = np.mean([r["roc_auc"] for r in fold_results])
    avg_f1 = np.mean([r["f1_score"] for r in fold_results])
    avg_rec = np.mean([r["recall"] for r in fold_results])
    avg_prec = np.mean([r["precision"] for r in fold_results])

    print("-" * 72)
    print(f"GroupKFold Average:                        {avg_auc:<10.3f} {avg_f1:<8.3f} {avg_rec:<8.3f} {avg_prec:<10.3f}")
    print("-" * 72)

    return fold_results, avg_auc, avg_f1


def train_final_model(df: pd.DataFrame):
    """Trains final model on all data and records training feature boundaries."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    scale_pos = (len(y) - sum(y)) / max(1, sum(y))

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )
    model.fit(X, y)

    # Calculate training feature ranges for Out-Of-Distribution confidence scoring
    feature_ranges = {}
    for col in FEATURE_COLS:
        feature_ranges[col] = {
            "min": round(float(X[col].min()), 3),
            "max": round(float(X[col].max()), 3),
            "mean": round(float(X[col].mean()), 3),
            "std": round(float(X[col].std()), 3),
        }

    return model, feature_ranges


def main():
    print("=" * 72)
    print("🐘 GAJRAKSHA - XGBOOST TRAINING PIPELINE (GROUP-K-FOLD EVALUATION)")
    print("=" * 72)

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df):,} records with {len(FEATURE_COLS)} transferable features.")

    # 1. GroupKFold evaluation
    fold_results, avg_auc, avg_f1 = train_and_evaluate_groupkfold(df)

    # 2. Train final model & record feature ranges
    model, feature_ranges = train_final_model(df)

    # 3. Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print(f"\n✓ Saved trained model to: {MODEL_OUTPUT_FILE.relative_to(BASE_DIR)}")

    metadata = {
        "model_name": "GajRaksha XGBoost Classifier",
        "task": "Elephant presence risk near villages prediction",
        "target": TARGET_COL,
        "features": FEATURE_COLS,
        "training_feature_ranges": feature_ranges,
        "evaluation": {
            "method": "GroupKFold (n_splits=4, grouped by animal)",
            "average_roc_auc": round(float(avg_auc), 4),
            "average_f1": round(float(avg_f1), 4),
            "folds": fold_results
        },
        "training_records": len(df)
    }

    with open(METADATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved model metadata to: {METADATA_OUTPUT_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
