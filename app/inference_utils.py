"""
WildGuard AI - Inference Utilities
==================================
Unified interface for loading pretrained models and running inference.
Models:
  - ARIMA:   Population forecasting (replaces Prophet — lower MAPE)
  - RF:      Trend classification   (replaces LSTM  — higher F1)
  - XGBoost: Risk classification    (unchanged — best performer)

Author: WildGuard AI Project Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Constants
TREND_CLASSES = {0: 'Sharp Decline', 1: 'Moderate Decline', 2: 'Stable', 3: 'Recovering'}
RISK_CLASSES = {0: 'Low', 1: 'Medium', 2: 'High'}

# Feature columns for XGBoost (must match training order)
XGB_FEATURES = [
    'population_change_rate', 'population_rolling_avg', 'population_rolling_std',
    'population_cv', 'population_relative_to_peak', 'population_relative_to_baseline',
    'cumulative_change_rate', 'is_declining', 'is_growing', 'iucn_status_numeric',
    'conservation_urgency', 'years_since_baseline', 'taxonomic_group_encoded', 'region_encoded'
]


class InferenceEngine:
    """
    Handles model loading and inference for the dashboard.
    Uses Streamlit caching to prevent reloading on every rerun.
    """
    
    def __init__(self):
        self.rf_model = None
        self.rf_scaler = None
        self.rf_features = None
        self.xgboost_model = None
    
    @st.cache_resource
    def load_static_models(_self):
        """
        Load Random Forest (trend) and XGBoost (risk) models.
        """
        # Load Random Forest
        rf_path = MODELS_DIR / "rf_trend_model.pkl"
        rf_model = None
        rf_scaler = None
        rf_features = None
        if rf_path.exists():
            try:
                with open(rf_path, 'rb') as f:
                    rf_data = pickle.load(f)
                rf_model = rf_data['model']
                rf_scaler = rf_data['scaler']
                rf_features = rf_data['features']
                print(f"✓ Random Forest model loaded")
            except Exception as e:
                print(f"⚠ Error loading RF: {e}")
        
        # Load XGBoost
        xgb_path = MODELS_DIR / "xgboost_risk_model.json"
        xgb_model = None
        if xgb_path.exists():
            try:
                import xgboost as xgb
                xgb_model = xgb.Booster()
                xgb_model.load_model(str(xgb_path))
                print(f"✓ XGBoost model loaded")
            except Exception as e:
                print(f"⚠ Error loading XGBoost: {e}")
                
        return rf_model, rf_scaler, rf_features, xgb_model

    def get_data(self):
        """Load the engineered dataset."""
        return pd.read_csv(DATA_DIR / "engineered_wildlife_data.csv")

    def get_forecast_data(self):
        """Load the forecast dataset."""
        return pd.read_csv(DATA_DIR / "forecast_dataset.csv")

    def run_arima_forecast(self, species_name, years=5):
        """
        Run ARIMA(1,1,1) forecast for a specific species.
        Returns (model_info, forecast_df) matching the format the dashboard expects.
        
        The forecast_df has columns: ds, yhat, yhat_lower, yhat_upper
        to maintain compatibility with the existing Plotly chart code.
        """
        from statsmodels.tsa.arima.model import ARIMA
        
        # Load raw data for this species
        raw_df = pd.read_csv(DATA_DIR / "raw_wildlife_data.csv")
        species_df = raw_df[raw_df['species_common_name'] == species_name][['year', 'population']].copy()
        species_df = species_df.sort_values('year').drop_duplicates('year')
        
        if len(species_df) < 3:
            return None, None
        
        # Fit ARIMA
        try:
            train_pop = species_df['population'].values.astype(float)
            model = ARIMA(train_pop, order=(1, 1, 1))
            # innovations_mle is much faster than default MLE for dashboard use
            fit = model.fit(method='innovations_mle')
            
            # Forecast
            fc = fit.get_forecast(steps=years)
            pred_arr = np.asarray(fc.predicted_mean).flatten()
            ci = np.asarray(fc.conf_int(alpha=0.05))
            ci_lower = ci[:, 0] if ci.ndim == 2 else ci
            ci_upper = ci[:, 1] if ci.ndim == 2 else ci
            
            # Create historical DataFrame
            hist_years = species_df['year'].values
            hist_ds = pd.to_datetime(hist_years, format='%Y')
            
            hist_df = pd.DataFrame({
                'ds': hist_ds,
                'yhat': train_pop,
            })
            
            # Create future DataFrame
            last_year = int(hist_years[-1])
            future_years = [last_year + i + 1 for i in range(years)]
            future_ds = pd.to_datetime(future_years, format='%Y')
            
            future_df = pd.DataFrame({
                'ds': future_ds,
                'yhat': pred_arr,
                'yhat_lower': ci_lower,
                'yhat_upper': ci_upper,
            })
            
            # Combine (same format Prophet used)
            hist_df['yhat_lower'] = hist_df['yhat'] * 0.95
            hist_df['yhat_upper'] = hist_df['yhat'] * 1.05
            
            forecast = pd.concat([hist_df, future_df], ignore_index=True)
            
            return fit, forecast
            
        except Exception as e:
            print(f"⚠ ARIMA forecast failed for {species_name}: {e}")
            return None, None

    # Keep Prophet as a fallback method
    def run_prophet_forecast(self, species_name, years=5):
        """Fallback: Run Prophet forecast (kept for compatibility)."""
        return self.run_arima_forecast(species_name, years)

    def predict_trend(self, rf_model, species_data, rf_scaler=None, rf_features=None):
        """
        Predict trend using Random Forest on tabular features.
        """
        if rf_model is None:
            return "Model Not Loaded", 0.0
        
        if rf_features is None:
            rf_features = [
                'population_change_rate', 'population_cv', 'population_relative_to_peak',
                'population_rolling_std', 'cumulative_change_rate', 'conservation_urgency',
                'years_since_baseline', 'is_declining', 'is_growing'
            ]
        
        # Use the latest row of species data
        latest = species_data.sort_values('year').tail(1)
        
        features = []
        for col in rf_features:
            val = latest[col].values[0] if col in latest.columns else 0
            if pd.isna(val):
                val = 0
            features.append(val)
        
        X = np.array(features).reshape(1, -1)
        
        # Scale if scaler available
        if rf_scaler is not None:
            X = rf_scaler.transform(X)
        
        prediction = rf_model.predict(X)[0]
        confidence = float(np.max(rf_model.predict_proba(X)[0]))
        
        return TREND_CLASSES[prediction], confidence

    def predict_risk(self, xgb_model, species_latest_row):
        """
        Predict risk using XGBoost model.
        """
        if xgb_model is None:
            return "Model Not Loaded"
            
        import xgboost as xgb
        
        features = []
        for col in XGB_FEATURES:
            val = species_latest_row.get(col, 0)
            if isinstance(val, pd.Series):
                val = val.values[0]
            features.append(val)
            
        X = np.array(features).reshape(1, -1)
        dmatrix = xgb.DMatrix(X, feature_names=XGB_FEATURES)
        
        prediction = int(xgb_model.predict(dmatrix)[0])
        
        return RISK_CLASSES[prediction]

# Instantiate global engine
engine = InferenceEngine()
