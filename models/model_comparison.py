"""
WildGuard AI - Model Comparison & Validation
=============================================
Temporal Hold-Out Validation:
  - Train: data ≤ 2020
  - Test:  data > 2020 (real census values)

Compares 9 models across 3 tasks:
  Task A: Forecasting    → Prophet vs ARIMA vs LSTM
  Task B: Trend Class.   → LSTM vs Random Forest vs SVM
  Task C: Risk Class.    → XGBoost vs Random Forest vs Logistic Regression

Author: WildGuard AI Project Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error
)

# XGBoost
import xgboost as xgb

# Statsmodels
from statsmodels.tsa.arima.model import ARIMA

# Prophet
from prophet import Prophet

# TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# CONFIG
# ============================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

CUTOFF_YEAR = 2020
RANDOM_STATE = 42

RISK_MAPPING = {'CR': 'High', 'EN': 'High', 'VU': 'Medium', 'NT': 'Low', 'LC': 'Low'}

# ============================
# HELPER: MAPE
# ============================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =============================================================================
# TASK A: FORECASTING COMPARISON
# =============================================================================
def run_forecasting_comparison(df_raw):
    """Compare Prophet vs ARIMA vs LSTM Forecasting."""
    print("\n" + "=" * 70)
    print("   TASK A: FORECASTING COMPARISON")
    print("   Prophet vs ARIMA vs LSTM")
    print("=" * 70)

    species_list = df_raw['species_common_name'].unique()
    results = {m: {'rmse': [], 'mae': [], 'mape': [], 'per_species': {}} for m in ['Prophet', 'ARIMA', 'LSTM_Forecast']}

    for sp in species_list:
        sp_data = df_raw[df_raw['species_common_name'] == sp].sort_values('year')
        train = sp_data[sp_data['year'] <= CUTOFF_YEAR]
        test = sp_data[sp_data['year'] > CUTOFF_YEAR]

        if len(train) < 5 or len(test) == 0:
            print(f"  ⚠ Skipping {sp}: insufficient data")
            continue

        actual = test['population'].values
        test_years = test['year'].values
        n_forecast = len(test)

        print(f"\n  🔸 {sp} (train: ≤{CUTOFF_YEAR}, test: {n_forecast} years)")

        # --- PROPHET ---
        try:
            df_p = train[['year', 'population']].rename(columns={'year': 'ds', 'population': 'y'})
            df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                        changepoint_prior_scale=0.05)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=n_forecast, freq='YE')
            fc = m.predict(future)
            pred_prophet = fc.tail(n_forecast)['yhat'].values

            rmse = np.sqrt(mean_squared_error(actual, pred_prophet))
            mae = mean_absolute_error(actual, pred_prophet)
            mape = mean_absolute_percentage_error(actual, pred_prophet)
            results['Prophet']['rmse'].append(rmse)
            results['Prophet']['mae'].append(mae)
            results['Prophet']['mape'].append(mape)
            results['Prophet']['per_species'][sp] = {'rmse': rmse, 'mae': mae, 'mape': mape,
                                                      'actual': actual.tolist(), 'predicted': pred_prophet.tolist()}
            print(f"    Prophet   → MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"    Prophet   → FAILED: {e}")

        # --- ARIMA ---
        try:
            train_pop = train['population'].values
            model_arima = ARIMA(train_pop, order=(1, 1, 1))
            fit_arima = model_arima.fit()
            pred_arima = fit_arima.forecast(steps=n_forecast)

            rmse = np.sqrt(mean_squared_error(actual, pred_arima))
            mae = mean_absolute_error(actual, pred_arima)
            mape = mean_absolute_percentage_error(actual, pred_arima)
            results['ARIMA']['rmse'].append(rmse)
            results['ARIMA']['mae'].append(mae)
            results['ARIMA']['mape'].append(mape)
            results['ARIMA']['per_species'][sp] = {'rmse': rmse, 'mae': mae, 'mape': mape,
                                                    'actual': actual.tolist(), 'predicted': pred_arima.tolist()}
            print(f"    ARIMA     → MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"    ARIMA     → FAILED: {e}")

        # --- LSTM FORECASTING ---
        try:
            train_pop = train['population'].values.reshape(-1, 1).astype(float)
            scaler_mean = train_pop.mean()
            scaler_std = train_pop.std() if train_pop.std() > 0 else 1.0
            train_scaled = (train_pop - scaler_mean) / scaler_std

            lookback = min(5, len(train_scaled) - 1)
            X_lstm, y_lstm = [], []
            for i in range(lookback, len(train_scaled)):
                X_lstm.append(train_scaled[i - lookback:i, 0])
                y_lstm.append(train_scaled[i, 0])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

            model_lstm = Sequential([
                Input(shape=(lookback, 1)),
                LSTM(32, activation='relu'),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_lstm, y_lstm, epochs=100, batch_size=4, verbose=0)

            # Predict iteratively
            last_seq = train_scaled[-lookback:].flatten().tolist()
            pred_lstm = []
            for _ in range(n_forecast):
                x_in = np.array(last_seq[-lookback:]).reshape(1, lookback, 1)
                p = model_lstm.predict(x_in, verbose=0)[0, 0]
                pred_lstm.append(p)
                last_seq.append(p)

            pred_lstm = np.array(pred_lstm) * scaler_std + scaler_mean

            rmse = np.sqrt(mean_squared_error(actual, pred_lstm))
            mae = mean_absolute_error(actual, pred_lstm)
            mape = mean_absolute_percentage_error(actual, pred_lstm)
            results['LSTM_Forecast']['rmse'].append(rmse)
            results['LSTM_Forecast']['mae'].append(mae)
            results['LSTM_Forecast']['mape'].append(mape)
            results['LSTM_Forecast']['per_species'][sp] = {'rmse': rmse, 'mae': mae, 'mape': mape,
                                                            'actual': actual.tolist(), 'predicted': pred_lstm.tolist()}
            print(f"    LSTM      → MAPE: {mape:.2f}%")
        except Exception as e:
            print(f"    LSTM      → FAILED: {e}")

    # Aggregate
    summary = {}
    for model_name, data in results.items():
        summary[model_name] = {
            'avg_rmse': float(np.mean(data['rmse'])) if data['rmse'] else None,
            'avg_mae': float(np.mean(data['mae'])) if data['mae'] else None,
            'avg_mape': float(np.mean(data['mape'])) if data['mape'] else None,
            'per_species': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
                           for k, v in data['per_species'].items()}
        }

    print("\n" + "-" * 60)
    print("  FORECASTING SUMMARY (Average across all species)")
    print("-" * 60)
    print(f"  {'Model':<18} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")
    for m in ['Prophet', 'ARIMA', 'LSTM_Forecast']:
        s = summary[m]
        r = f"{s['avg_rmse']:.1f}" if s['avg_rmse'] else "N/A"
        a = f"{s['avg_mae']:.1f}" if s['avg_mae'] else "N/A"
        p = f"{s['avg_mape']:.2f}%" if s['avg_mape'] else "N/A"
        print(f"  {m:<18} {r:>12} {a:>12} {p:>10}")

    return summary


# =============================================================================
# TASK B: TREND CLASSIFICATION COMPARISON
# =============================================================================
def run_trend_comparison(df_eng):
    """Compare LSTM vs Random Forest vs SVM for trend classification."""
    print("\n" + "=" * 70)
    print("   TASK B: TREND CLASSIFICATION COMPARISON")
    print("   LSTM vs Random Forest vs SVM")
    print("=" * 70)

    # Create trend labels
    def assign_trend(row):
        cr = row.get('population_change_rate', 0)
        if pd.isna(cr):
            cr = 0
        if cr < -10:
            return 0  # Sharp Decline
        elif cr < -2:
            return 1  # Moderate Decline
        elif cr <= 5:
            return 2  # Stable
        else:
            return 3  # Recovering

    df = df_eng.copy()
    df['trend_label'] = df.apply(assign_trend, axis=1)

    TREND_NAMES = {0: 'Sharp Decline', 1: 'Moderate Decline', 2: 'Stable', 3: 'Recovering'}

    # Train/test split by year
    train = df[df['year'] <= CUTOFF_YEAR]
    test = df[df['year'] > CUTOFF_YEAR]

    print(f"  Train: {len(train)} samples (≤{CUTOFF_YEAR})")
    print(f"  Test:  {len(test)} samples (>{CUTOFF_YEAR})")

    # Features for tabular models
    feature_cols = [
        'population_change_rate', 'population_cv', 'population_relative_to_peak',
        'population_rolling_std', 'cumulative_change_rate', 'conservation_urgency',
        'years_since_baseline', 'is_declining', 'is_growing'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]

    X_train_tab = train[feature_cols].fillna(0).values
    y_train = train['trend_label'].values
    X_test_tab = test[feature_cols].fillna(0).values
    y_test = test['trend_label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_tab)
    X_test_scaled = scaler.transform(X_test_tab)

    results = {}

    # --- Random Forest ---
    print("\n  🌲 Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=8)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_rf, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_rf, average='macro', zero_division=0)),
        'cm': confusion_matrix(y_test, y_pred_rf).tolist()
    }
    print(f"    Accuracy: {results['Random Forest']['accuracy']:.4f}, F1: {results['Random Forest']['f1']:.4f}")

    # --- SVM ---
    print("  🎯 SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    results['SVM'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_svm)),
        'precision': float(precision_score(y_test, y_pred_svm, average='macro', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_svm, average='macro', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_svm, average='macro', zero_division=0)),
        'cm': confusion_matrix(y_test, y_pred_svm).tolist()
    }
    print(f"    Accuracy: {results['SVM']['accuracy']:.4f}, F1: {results['SVM']['f1']:.4f}")

    # --- LSTM ---
    print("  🧠 LSTM...")

    # Build sequences per species for LSTM
    seq_len = 5
    X_seq_train, y_seq_train = [], []
    X_seq_test, y_seq_test = [], []

    for sp in df['species_common_name'].unique():
        sp_data = df[df['species_common_name'] == sp].sort_values('year')
        sp_features = sp_data[feature_cols].fillna(0).values

        for i in range(seq_len, len(sp_data)):
            seq = sp_features[i - seq_len:i]
            label = sp_data.iloc[i]['trend_label']
            year = sp_data.iloc[i]['year']

            if year <= CUTOFF_YEAR:
                X_seq_train.append(seq)
                y_seq_train.append(label)
            else:
                X_seq_test.append(seq)
                y_seq_test.append(label)

    if len(X_seq_test) > 0:
        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)
        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)

        num_classes = 4
        y_train_cat = to_categorical(y_seq_train, num_classes)
        y_test_cat = to_categorical(y_seq_test, num_classes)

        model_lstm = Sequential([
            Input(shape=(seq_len, len(feature_cols))),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_lstm.fit(X_seq_train, y_train_cat, epochs=50, batch_size=16, verbose=0,
                       validation_split=0.2)

        y_pred_lstm = np.argmax(model_lstm.predict(X_seq_test, verbose=0), axis=1)
        y_true_lstm = y_seq_test

        results['LSTM'] = {
            'accuracy': float(accuracy_score(y_true_lstm, y_pred_lstm)),
            'precision': float(precision_score(y_true_lstm, y_pred_lstm, average='macro', zero_division=0)),
            'recall': float(recall_score(y_true_lstm, y_pred_lstm, average='macro', zero_division=0)),
            'f1': float(f1_score(y_true_lstm, y_pred_lstm, average='macro', zero_division=0)),
            'cm': confusion_matrix(y_true_lstm, y_pred_lstm).tolist()
        }
        print(f"    Accuracy: {results['LSTM']['accuracy']:.4f}, F1: {results['LSTM']['f1']:.4f}")
    else:
        print("    ⚠ Insufficient LSTM test sequences")
        results['LSTM'] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'cm': []}

    # Print summary
    print("\n" + "-" * 60)
    print("  TREND CLASSIFICATION SUMMARY")
    print("-" * 60)
    print(f"  {'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for m in ['LSTM', 'Random Forest', 'SVM']:
        r = results[m]
        print(f"  {m:<18} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

    return results


# =============================================================================
# TASK C: RISK CLASSIFICATION COMPARISON
# =============================================================================
def run_risk_comparison(df_eng):
    """Compare XGBoost vs Random Forest vs Logistic Regression for risk classification."""
    print("\n" + "=" * 70)
    print("   TASK C: RISK CLASSIFICATION COMPARISON")
    print("   XGBoost vs Random Forest vs Logistic Regression")
    print("=" * 70)

    df = df_eng.copy()
    df['risk_target'] = df['iucn_status'].map(RISK_MAPPING)
    risk_order = {'Low': 0, 'Medium': 1, 'High': 2}
    df['risk_numeric'] = df['risk_target'].map(risk_order)

    feature_cols = [
        'population_change_rate', 'population_rolling_avg', 'population_rolling_std',
        'population_cv', 'population_relative_to_peak', 'population_relative_to_baseline',
        'cumulative_change_rate', 'is_declining', 'is_growing', 'iucn_status_numeric',
        'conservation_urgency', 'years_since_baseline', 'taxonomic_group_encoded', 'region_encoded'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]

    # Temporal split
    train = df[df['year'] <= CUTOFF_YEAR]
    test = df[df['year'] > CUTOFF_YEAR]

    X_train = train[feature_cols].fillna(train[feature_cols].median()).values
    y_train = train['risk_numeric'].values
    X_test = test[feature_cols].fillna(test[feature_cols].median()).values
    y_test = test['risk_numeric'].values

    print(f"  Train: {len(X_train)} samples (≤{CUTOFF_YEAR})")
    print(f"  Test:  {len(X_test)} samples (>{CUTOFF_YEAR})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- XGBoost ---
    print("\n  🚀 XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    params = {
        'max_depth': 4, 'eta': 0.1, 'objective': 'multi:softmax',
        'num_class': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': RANDOM_STATE
    }
    model_xgb = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
    y_pred_xgb = model_xgb.predict(dtest).astype(int)
    results['XGBoost'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
        'precision': float(precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0)),
        'cm': confusion_matrix(y_test, y_pred_xgb).tolist()
    }
    print(f"    Accuracy: {results['XGBoost']['accuracy']:.4f}, F1: {results['XGBoost']['f1']:.4f}")

    # --- Random Forest ---
    print("  🌲 Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=6)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_rf)),
        'precision': float(precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)),
        'cm': confusion_matrix(y_test, y_pred_rf).tolist()
    }
    print(f"    Accuracy: {results['Random Forest']['accuracy']:.4f}, F1: {results['Random Forest']['f1']:.4f}")

    # --- Logistic Regression ---
    print("  📈 Logistic Regression...")
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results['Logistic Regression'] = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)),
        'cm': confusion_matrix(y_test, y_pred_lr).tolist()
    }
    print(f"    Accuracy: {results['Logistic Regression']['accuracy']:.4f}, F1: {results['Logistic Regression']['f1']:.4f}")

    # Summary
    print("\n" + "-" * 60)
    print("  RISK CLASSIFICATION SUMMARY")
    print("-" * 60)
    print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for m in ['XGBoost', 'Random Forest', 'Logistic Regression']:
        r = results[m]
        print(f"  {m:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

    return results


# =============================================================================
# PLOTTING
# =============================================================================
def create_comparison_plots(forecast_results, trend_results, risk_results):
    """Generate professional comparison charts."""
    print("\n" + "=" * 70)
    print("   GENERATING COMPARISON PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('WildGuard AI — Model Comparison Results', fontsize=16, fontweight='bold', y=1.02)

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # --- Plot A: Forecasting MAPE ---
    ax = axes[0]
    models_f = ['Prophet', 'ARIMA', 'LSTM_Forecast']
    labels_f = ['Prophet', 'ARIMA', 'LSTM']
    mapes = [forecast_results[m]['avg_mape'] or 0 for m in models_f]
    bars = ax.bar(labels_f, mapes, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('Task A: Forecasting\n(Lower is Better)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylim(0, max(mapes) * 1.3 if max(mapes) > 0 else 10)
    # Mark winner
    winner_idx = np.argmin(mapes)
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(3)

    # --- Plot B: Trend Classification F1 ---
    ax = axes[1]
    models_t = ['LSTM', 'Random Forest', 'SVM']
    f1s_t = [trend_results[m]['f1'] for m in models_t]
    bars = ax.bar(models_t, f1s_t, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax.set_title('Task B: Trend Classification\n(Higher is Better)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, f1s_t):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.15)
    winner_idx = np.argmax(f1s_t)
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(3)

    # --- Plot C: Risk Classification F1 ---
    ax = axes[2]
    models_r = ['XGBoost', 'Random Forest', 'Logistic Regression']
    labels_r = ['XGBoost', 'Random\nForest', 'Logistic\nRegression']
    f1s_r = [risk_results[m]['f1'] for m in models_r]
    bars = ax.bar(labels_r, f1s_r, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('F1-Score (Weighted)', fontsize=12)
    ax.set_title('Task C: Risk Classification\n(Higher is Better)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, f1s_r):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.15)
    winner_idx = np.argmax(f1s_r)
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "model_comparison_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {plot_path}")

    return str(plot_path)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "█" * 70)
    print("  WILDGUARD AI — MODEL COMPARISON & VALIDATION")
    print(f"  Temporal Hold-Out: Train ≤{CUTOFF_YEAR} | Test >{CUTOFF_YEAR}")
    print("█" * 70)

    # Load data
    df_raw = pd.read_csv(DATA_DIR / "raw_wildlife_data.csv")
    df_eng = pd.read_csv(DATA_DIR / "engineered_wildlife_data.csv")
    print(f"\n✓ Loaded {len(df_raw)} raw records, {len(df_eng)} engineered records")
    print(f"✓ Species: {df_raw['species_common_name'].nunique()}")

    # Run all comparisons
    forecast_results = run_forecasting_comparison(df_raw)
    trend_results = run_trend_comparison(df_eng)
    risk_results = run_risk_comparison(df_eng)

    # Generate plots
    plot_path = create_comparison_plots(forecast_results, trend_results, risk_results)

    # Save all results
    all_results = {
        'metadata': {
            'cutoff_year': CUTOFF_YEAR,
            'num_species': int(df_raw['species_common_name'].nunique()),
            'species': df_raw['species_common_name'].unique().tolist()
        },
        'task_a_forecasting': forecast_results,
        'task_b_trend_classification': trend_results,
        'task_c_risk_classification': risk_results,
        'plot_path': plot_path
    }

    results_path = RESULTS_DIR / "comparison_tables.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved: {results_path}")

    # Final summary
    print("\n" + "█" * 70)
    print("  FINAL RESULTS: BEST MODEL PER TASK")
    print("█" * 70)

    # Forecasting winner
    fc_models = ['Prophet', 'ARIMA', 'LSTM_Forecast']
    fc_mapes = [forecast_results[m]['avg_mape'] or 999 for m in fc_models]
    fc_winner = fc_models[np.argmin(fc_mapes)]
    print(f"\n  Task A (Forecasting):         🏆 {fc_winner} (MAPE: {min(fc_mapes):.2f}%)")

    # Trend winner
    t_models = ['LSTM', 'Random Forest', 'SVM']
    t_f1s = [trend_results[m]['f1'] for m in t_models]
    t_winner = t_models[np.argmax(t_f1s)]
    print(f"  Task B (Trend Classification): 🏆 {t_winner} (F1: {max(t_f1s):.4f})")

    # Risk winner
    r_models = ['XGBoost', 'Random Forest', 'Logistic Regression']
    r_f1s = [risk_results[m]['f1'] for m in r_models]
    r_winner = r_models[np.argmax(r_f1s)]
    print(f"  Task C (Risk Classification):  🏆 {r_winner} (F1: {max(r_f1s):.4f})")

    print("\n" + "█" * 70)
    print("  MODEL COMPARISON COMPLETE ✓")
    print("█" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
