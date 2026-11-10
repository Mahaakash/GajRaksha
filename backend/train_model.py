"""
GajRaksha - XGBoost Model Training Pipeline
===========================================
Trains a gradient-boosted decision tree (XGBoost) model to predict
'elephant presence risk near villages' (a proxy for conflict).

Validation Rules & Ecological Standards:
1. Strictly evaluates with:
   - Split by Animal: Leave-One-Animal-Out cross-validation across all 4 elephants.
   - Split by Time: Chronological 75% train / 25% test split.
   - NEVER uses random train/test splits.
2. Features are 100% transferable to India:
   - hour_of_day, is_night, rainfall_last_7_days, rainfall_last_30_days,
     speed_kmh, step_distance_km, heading_change_deg, dist_to_nearest_village_km.
   - Excludes absolute latitude, longitude, calendar month, and spatial cell history.
3. Saves model artifact to models/elephant_presence_risk_xgb.joblib and
   metadata to models/model_metadata.json.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INPUT_PARQUET = DATA_DIR / "processed" / "elephant_presence_dataset.parquet"
MODEL_OUTPUT_FILE = MODELS_DIR / "elephant_presence_risk_xgb.joblib"
METADATA_OUTPUT_FILE = MODELS_DIR / "model_metadata.json"

# Transferable Feature Specification
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


def evaluate_split_by_animal(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Evaluates cross-individual generalization using Leave-One-Animal-Out validation.
    Trains on 3 elephants, tests on the 4th held-out elephant.
    """
    animals = df["individual_local_identifier"].unique()
    results = []

    print("-" * 68)
    print(f"ANIMAL-BASED SPLIT EVALUATION (Testing on Unseen Elephants)")
    print("-" * 68)
    print(f"{'Held-out Animal':<18} {'Train Size':<12} {'Test Size':<12} {'ROC-AUC':<10} {'F1-Score':<10} {'Recall':<10}")
    print("-" * 68)

    for test_animal in animals:
        train_mask = df["individual_local_identifier"] != test_animal
        test_mask = df["individual_local_identifier"] == test_animal

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, target_col]

        # Calculate imbalance weight
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

        results.append({
            "test_animal": test_animal,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "roc_auc": round(float(auc), 4),
            "f1_score": round(float(f1), 4),
            "recall": round(float(rec), 4)
        })

        print(f"Elephant {test_animal:<9} {len(X_train):<12,d} {len(X_test):<12,d} {auc:<10.3f} {f1:<10.3f} {rec:<10.3f}")

    avg_auc = np.mean([r["roc_auc"] for r in results])
    avg_f1 = np.mean([r["f1_score"] for r in results])
    avg_rec = np.mean([r["recall"] for r in results])
    print("-" * 68)
    print(f"Average Across Elephants:                    {avg_auc:<10.3f} {avg_f1:<10.3f} {avg_rec:<10.3f}")
    print("-" * 68)

    return results, avg_auc, avg_f1


def evaluate_split_by_time(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Evaluates temporal forecasting generalization using a chronological 75% train / 25% test split.
    Simulates predicting future elephant movement risk using only historical knowledge.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * 0.75)

    train_df = df_sorted.iloc[:cutoff_idx]
    test_df = df_sorted.iloc[cutoff_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    t_start_train = train_df['timestamp'].min()
    t_end_train = train_df['timestamp'].max()
    t_start_test = test_df['timestamp'].min()
    t_end_test = test_df['timestamp'].max()

    scale_pos = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

    model = XGBClassifier(
        n_estimators=120,
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

    auc = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_preds, zero_division=0)
    prec = precision_score(y_test, y_preds, zero_division=0)
    rec = recall_score(y_test, y_preds, zero_division=0)
    acc = accuracy_score(y_test, y_preds)
    cm = confusion_matrix(y_test, y_preds)

    print("\n" + "-" * 68)
    print("CHRONOLOGICAL TIME-BASED SPLIT EVALUATION (Predicting Future Timeline)")
    print("-" * 68)
    print(f"Train Period:  {t_start_train} to {t_end_train} ({len(train_df):,} records)")
    print(f"Test Period:   {t_start_test} to {t_end_test} ({len(test_df):,} records)")
    print(f"Test Class Balance: {sum(y_test):,} High Risk ({sum(y_test)/len(y_test)*100:.2f}%) / {len(y_test)-sum(y_test):,} Normal")
    print("-" * 68)
    print(f"ROC-AUC Score:      {auc:.4f}")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"Overall Accuracy:   {acc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:<6,d}  |  False Positives: {cm[0][1]:<6,d}")
    print(f"  False Negatives: {cm[1][0]:<6,d}  |  True Positives:  {cm[1][1]:<6,d}")
    print("-" * 68)

    return {
        "roc_auc": round(float(auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "train_range": [str(t_start_train), str(t_end_train)],
        "test_range": [str(t_start_test), str(t_end_test)]
    }


def train_production_model(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Trains the final production XGBoost model on the complete cleaned dataset
    and computes feature importance rankings.
    """
    print("\nTraining final production XGBoost model on full dataset...")
    X = df[feature_cols]
    y = df[target_col]

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

    # Feature Importances (Weight / Gain)
    importances = model.feature_importances_
    feat_rankings = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 68)
    print("XGBOOST FEATURE IMPORTANCE RANKINGS:")
    print("=" * 68)
    for rank, (feat, score) in enumerate(feat_rankings, 1):
        bar = "█" * int(score * 40)
        print(f"  {rank}. {feat:<28} {score:>7.4f}  {bar}")
    print("=" * 68)

    return model, feat_rankings


def main():
    print("=" * 68)
    print("🐘 GAJRAKSHA - ELEPHANT PRESENCE RISK MODEL TRAINING (XGBOOST)")
    print("=" * 68)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Processed dataset missing: {INPUT_PARQUET}. Run build_dataset.py first.")

    print(f"Loading dataset: {INPUT_PARQUET.relative_to(BASE_DIR)}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df):,} records with {len(FEATURE_COLS)} features.")

    # 1. Animal-based evaluation (Leave-One-Animal-Out)
    animal_results, avg_animal_auc, avg_animal_f1 = evaluate_split_by_animal(df, FEATURE_COLS, TARGET_COL)

    # 2. Time-based evaluation (75% past / 25% future)
    time_results = evaluate_split_by_time(df, FEATURE_COLS, TARGET_COL)

    # 3. Train final model
    prod_model, feat_rankings = train_production_model(df, FEATURE_COLS, TARGET_COL)

    # 4. Save Model Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod_model, MODEL_OUTPUT_FILE)
    print(f"\n✓ Saved trained XGBoost model to: {MODEL_OUTPUT_FILE.relative_to(BASE_DIR)}")

    # 5. Save Model Metadata
    metadata = {
        "model_type": "XGBoost Classifier",
        "task": "Elephant presence risk near villages prediction",
        "target_variable": TARGET_COL,
        "features": FEATURE_COLS,
        "feature_importances": {feat: round(float(score), 4) for feat, score in feat_rankings},
        "evaluation_metrics": {
            "split_by_animal": {
                "method": "Leave-One-Animal-Out (4 elephants)",
                "average_roc_auc": round(float(avg_animal_auc), 4),
                "average_f1_score": round(float(avg_animal_f1), 4),
                "individual_results": animal_results
            },
            "split_by_time": {
                "method": "Chronological (75% Train / 25% Test Out-of-Time)",
                **time_results
            }
        },
        "training_records": len(df),
        "target_balance": {
            "normal_fixes": int((df[TARGET_COL] == 0).sum()),
            "high_presence_risk_fixes": int((df[TARGET_COL] == 1).sum()),
            "risk_percentage": round(float((df[TARGET_COL] == 1).mean() * 100), 2)
        },
        "transferability": {
            "transferable_to_india": True,
            "excluded_features": ["latitude", "longitude", "month", "past_presence_count"],
            "reason": "Avoids spatial coordinate overfitting and hemispheric seasonal inversion"
        }
    }

    with open(METADATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved model metadata to: {METADATA_OUTPUT_FILE.relative_to(BASE_DIR)}")

    print("\n" + "=" * 68)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  • Cross-Animal ROC-AUC: {avg_animal_auc:.4f}")
    print(f"  • Cross-Time ROC-AUC:   {time_results['roc_auc']:.4f}")
    print(f"  • Model File:           {MODEL_OUTPUT_FILE.relative_to(BASE_DIR)}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
