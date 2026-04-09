"""
WildGuard AI - XGBoost Risk Classification Model
=================================================
Uses native XGBoost API for optimal compatibility.

Author: WildGuard AI Project Team
Date: January 2026

WHY XGBOOST FOR RISK CLASSIFICATION:
====================================
1. Superior performance on tabular/structured data
2. Built-in feature importance for interpretability  
3. Handles class imbalance effectively
4. Fast training and inference for real-time dashboard
5. Robust regularization prevents overfitting
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

print(f"✓ XGBoost version: {xgb.__version__}")

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent
PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
RISK_MAPPING = {'CR': 'High', 'EN': 'High', 'VU': 'Medium', 'NT': 'Low', 'LC': 'Low'}
RISK_CLASSES = {0: 'Low', 1: 'Medium', 2: 'High'}

def main():
    print("\n" + "=" * 70)
    print("    WILDGUARD AI - XGBOOST RISK CLASSIFICATION")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    df = pd.read_csv(DATA_DIR / "engineered_wildlife_data.csv")
    print(f"✓ Loaded {len(df)} records for {df['species_common_name'].nunique()} species")
    
    # Step 2: Define risk levels
    print("\n" + "=" * 70)
    print("STEP 2: DEFINING RISK LEVELS")
    print("=" * 70)
    df['risk_level_new'] = df['iucn_status'].map(RISK_MAPPING)
    
    print("✓ Risk Mapping:")
    for status, risk in RISK_MAPPING.items():
        count = len(df[df['iucn_status'] == status])
        print(f"   {status} → {risk} ({count} records)")
    
    print("\n✓ Risk Distribution:")
    for risk, count in df['risk_level_new'].value_counts().items():
        print(f"   {risk}: {count} ({count/len(df)*100:.1f}%)")
    
    # Step 3: Encode labels
    print("\n" + "=" * 70)
    print("STEP 3: ENCODING LABELS")
    print("=" * 70)
    risk_order = {'Low': 0, 'Medium': 1, 'High': 2}
    y = df['risk_level_new'].map(risk_order).values
    print("✓ Labels: Low=0, Medium=1, High=2")
    
    # Step 4: Prepare features
    print("\n" + "=" * 70)
    print("STEP 4: PREPARING FEATURES")
    print("=" * 70)
    feature_columns = [
        'population_change_rate', 'population_rolling_avg', 'population_rolling_std',
        'population_cv', 'population_relative_to_peak', 'population_relative_to_baseline',
        'cumulative_change_rate', 'is_declining', 'is_growing', 'iucn_status_numeric',
        'conservation_urgency', 'years_since_baseline', 'taxonomic_group_encoded', 'region_encoded'
    ]
    features = [f for f in feature_columns if f in df.columns]
    X = df[features].fillna(df[features].median()).values
    print(f"✓ Selected {len(features)} features")
    
    # Step 5: Split data
    print("\n" + "=" * 70)
    print("STEP 5: SPLITTING DATA (80/20)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✓ Training: {len(X_train)} | Test: {len(X_test)}")
    
    # Step 6: Train XGBoost
    print("\n" + "=" * 70)
    print("STEP 6: TRAINING XGBOOST")
    print("=" * 70)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    
    params = {
        'max_depth': 4,
        'eta': 0.1,
        'objective': 'multi:softmax',
        'num_class': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': RANDOM_STATE
    }
    
    print("✓ Parameters:", params)
    model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
    print("✓ Training complete")
    
    # Step 7: Evaluate
    print("\n" + "=" * 70)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 70)
    
    y_pred = model.predict(dtest).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS")
    print("=" * 50)
    print(f"   Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print("=" * 50)
    
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    feat_imp = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    
    # Step 8: Plot results
    print("\n" + "=" * 70)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('XGBoost Risk Classification Results', fontsize=14, fontweight='bold')
    
    # Confusion matrix
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'], ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
    
    # Feature importance
    ax2 = axes[1]
    top_feat = feat_imp.head(10)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_feat)))
    ax2.barh(top_feat['feature'], top_feat['importance'], color=colors)
    ax2.set_xlabel('Importance (Gain)')
    ax2.set_title('Top 10 Feature Importance')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "xgboost_risk_classification_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # Step 9: Save model
    print("\n" + "=" * 70)
    print("STEP 9: SAVING MODEL")
    print("=" * 70)
    
    model_path = MODELS_DIR / "xgboost_risk_model.json"
    model.save_model(model_path)
    print(f"✓ Model saved: {model_path}")
    
    metrics = {
        'accuracy': float(accuracy), 'precision': float(precision),
        'recall': float(recall), 'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'feature_importance': feat_imp.to_dict('records')
    }
    with open(MODELS_DIR / "xgboost_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {MODELS_DIR / 'xgboost_metrics.json'}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("MODEL INTERPRETATION")
    print("=" * 70)
    print(f"""
XGBOOST RISK CLASSIFICATION RESULTS:
====================================
• Accuracy:  {accuracy*100:.1f}%
• Precision: {precision:.3f}
• Recall:    {recall:.3f}
• F1-Score:  {f1:.3f}

TOP FEATURES:""")
    for _, row in feat_imp.head(5).iterrows():
        print(f"   • {row['feature']}: {row['importance']:.2f}")
    
    print(f"""
WHY XGBOOST EXCELS:
✓ Handles tabular data optimally
✓ Provides interpretable feature importance
✓ Robust regularization for small datasets
✓ Fast inference for real-time predictions

PRACTICAL USE:
The model classifies species into risk categories based on
conservation status, population trends, and change rates.
Feature importance shows IUCN status and change metrics
are the strongest predictors.
""")
    
    print("\n" + "=" * 70)
    print("XGBOOST RISK CLASSIFICATION COMPLETE ✓")
    print("=" * 70)

if __name__ == "__main__":
    main()
