"""
WildGuard AI - Random Forest Trend Model Trainer
==================================================
Standalone script to train the Random Forest trend classification model
used by the dashboard. This avoids requiring TensorFlow for retraining.

Usage: python models/train_rf_trend.py

Author: WildGuard AI Project Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent
RANDOM_STATE = 42

TREND_CLASSES = {0: 'Sharp Decline', 1: 'Moderate Decline', 2: 'Stable', 3: 'Recovering'}

FEATURE_COLS = [
    'population_change_rate', 'population_cv', 'population_relative_to_peak',
    'population_rolling_std', 'cumulative_change_rate', 'conservation_urgency',
    'years_since_baseline', 'is_declining', 'is_growing'
]


def assign_trend(change_rate):
    """Assign trend label based on change rate thresholds."""
    if change_rate < -10:
        return 0  # Sharp Decline
    elif change_rate < -2:
        return 1  # Moderate Decline
    elif change_rate <= 5:
        return 2  # Stable
    else:
        return 3  # Recovering


def main():
    print("\n" + "=" * 70)
    print("    WILDGUARD AI - RANDOM FOREST TREND MODEL TRAINING")
    print("=" * 70)

    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    df = pd.read_csv(DATA_DIR / "engineered_wildlife_data.csv")
    print(f"✓ Loaded {len(df)} records for {df['species_common_name'].nunique()} species")

    # Step 2: Create trend labels
    print("\n" + "=" * 70)
    print("STEP 2: CREATING TREND LABELS")
    print("=" * 70)
    df['trend_label'] = df['population_change_rate'].apply(assign_trend)
    
    print("✓ Trend Label Distribution:")
    for label, count in df['trend_label'].value_counts().sort_index().items():
        print(f"   {TREND_CLASSES[label]}: {count} ({count/len(df)*100:.1f}%)")

    # Step 3: Prepare features
    print("\n" + "=" * 70)
    print("STEP 3: PREPARING FEATURES")
    print("=" * 70)
    features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[features].fillna(0).values
    y = df['trend_label'].values
    print(f"✓ Selected {len(features)} features: {features}")

    # Step 4: Scale features
    scaler = StandardScaler()
    
    # Step 5: Split data
    print("\n" + "=" * 70)
    print("STEP 4: SPLITTING DATA (80/20)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Training: {len(X_train)} | Test: {len(X_test)}")

    # Step 6: Train Random Forest
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING RANDOM FOREST")
    print("=" * 70)
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE, 
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf.fit(X_train_scaled, y_train)
    print("✓ Training complete")

    # Step 7: Evaluate
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 70)
    
    y_pred = rf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_val = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"\n   Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall_val:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    target_names = [TREND_CLASSES[i] for i in sorted(TREND_CLASSES.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Step 8: Save model
    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODEL")
    print("=" * 70)
    
    model_path = MODELS_DIR / "rf_trend_model.pkl"
    model_data = {
        'model': rf,
        'scaler': scaler,
        'features': features,
        'accuracy': accuracy,
        'f1_score': f1,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved: {model_path}")
    print(f"✓ File size: {model_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("RANDOM FOREST TREND MODEL TRAINING COMPLETE ✓")
    print("=" * 70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall_val,
        'f1_score': f1,
    }


if __name__ == "__main__":
    metrics = main()
