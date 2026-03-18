"""
WildGuard AI - Feature Engineering Pipeline
============================================
This script performs feature engineering on the cleaned wildlife dataset
and prepares task-specific datasets for the three ML models:
1. Prophet - Population Forecasting
2. LSTM - Trend Detection
3. XGBoost - Risk Classification

Author: WildGuard AI Project Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent
CLEANED_DATA_PATH = DATA_DIR / "cleaned_wildlife_data.csv"

# Output paths for task-specific datasets
FORECAST_DATA_PATH = DATA_DIR / "forecast_dataset.csv"
TREND_DATA_PATH = DATA_DIR / "trend_dataset.csv"
CLASSIFICATION_DATA_PATH = DATA_DIR / "classification_dataset.csv"
ENGINEERED_DATA_PATH = DATA_DIR / "engineered_wildlife_data.csv"


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned wildlife dataset."""
    print("=" * 70)
    print("LOADING CLEANED DATA")
    print("=" * 70)
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    df = df.sort_values(['species_common_name', 'year']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} records for {df['species_common_name'].nunique()} species")
    print()
    return df


def create_population_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: Population Change Rate (Year-on-Year)
    =================================================
    
    FORMULA:
        change_rate = (P_t - P_{t-1}) / P_{t-1} × 100
    
    WHY THIS FEATURE IS IMPORTANT:
    ------------------------------
    1. TREND INDICATOR: The rate of change reveals whether a population is
       growing, declining, or stable, independent of absolute population size.
       A small species with 100 individuals declining by 10% is in more danger
       than a large species with 1 million declining by 1%.
    
    2. SCALE NORMALIZATION: Raw population values span several orders of
       magnitude (22 to 1.4 billion). Change rates normalize this, allowing
       fair comparison across species.
    
    3. LSTM INPUT: Neural networks learn patterns better from normalized,
       bounded values. Change rates typically fall within [-50%, +50%],
       making them ideal LSTM inputs.
    
    4. EARLY WARNING: Sudden spikes in negative change rates can signal
       emerging threats before absolute numbers become critically low.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with population data
        
    Returns:
        pd.DataFrame: Dataframe with population_change_rate column
    """
    print("=" * 70)
    print("FEATURE 1: Population Change Rate (Year-on-Year)")
    print("=" * 70)
    
    # Calculate year-on-year change rate within each species
    df['population_change_rate'] = df.groupby('species_common_name')['population'].transform(
        lambda x: x.pct_change() * 100  # Convert to percentage
    )
    
    # Handle first year of each species (no previous year to compare)
    # Fill with 0 (assuming no change for the baseline year)
    df['population_change_rate'] = df['population_change_rate'].fillna(0)
    
    # Clip extreme values to prevent outlier influence
    df['population_change_rate'] = df['population_change_rate'].clip(-100, 100)
    
    # Statistics
    print(f"✓ Created population_change_rate column")
    print(f"   Min: {df['population_change_rate'].min():.2f}%")
    print(f"   Max: {df['population_change_rate'].max():.2f}%")
    print(f"   Mean: {df['population_change_rate'].mean():.2f}%")
    print(f"   Std: {df['population_change_rate'].std():.2f}%")
    print()
    
    return df


def create_rolling_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Feature 2: Rolling Average Population (3-Year Window)
    ======================================================
    
    FORMULA:
        rolling_avg_3yr = (P_t + P_{t-1} + P_{t-2}) / 3
    
    WHY THIS FEATURE IS IMPORTANT:
    ------------------------------
    1. NOISE REDUCTION: Wildlife census data often contains measurement errors
       and natural year-to-year fluctuations. A rolling average smooths these
       irregularities, revealing the true underlying trend.
    
    2. PROPHET INPUT: Time-series forecasting models like Prophet perform
       better with smoothed data that reduces high-frequency noise while
       preserving long-term patterns.
    
    3. SEASONAL ADJUSTMENT: Some species populations fluctuate due to breeding
       cycles, migration patterns, or survey timing. Averaging over multiple
       years mitigates these seasonal effects.
    
    4. TREND STABILITY: Rolling averages make trend detection more reliable
       by reducing the impact of single-year anomalies.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        window (int): Rolling window size (default: 3 years)
        
    Returns:
        pd.DataFrame: Dataframe with population_rolling_avg column
    """
    print("=" * 70)
    print(f"FEATURE 2: Rolling Average Population ({window}-Year Window)")
    print("=" * 70)
    
    # Calculate rolling average within each species
    df['population_rolling_avg'] = df.groupby('species_common_name')['population'].transform(
        lambda x: x.rolling(window=window, min_periods=1, center=False).mean()
    )
    
    # Also create a rolling standard deviation for volatility measure
    df['population_rolling_std'] = df.groupby('species_common_name')['population'].transform(
        lambda x: x.rolling(window=window, min_periods=1, center=False).std()
    )
    df['population_rolling_std'] = df['population_rolling_std'].fillna(0)
    
    # Create coefficient of variation (CV) for normalized volatility
    df['population_cv'] = df['population_rolling_std'] / (df['population_rolling_avg'] + 1e-10)
    
    print(f"✓ Created population_rolling_avg column (window={window})")
    print(f"✓ Created population_rolling_std column (volatility measure)")
    print(f"✓ Created population_cv column (coefficient of variation)")
    print()
    
    return df


def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Additional Derived Features
    ===================================
    
    These features enhance model performance by capturing temporal patterns,
    population dynamics, and species-specific characteristics.
    """
    print("=" * 70)
    print("ADDITIONAL DERIVED FEATURES")
    print("=" * 70)
    
    # ---------------------------------------------------------------------
    # FEATURE 3: Years Since Baseline
    # WHY: Captures temporal distance from first observation, useful for
    #      trend models to understand long-term dynamics
    # ---------------------------------------------------------------------
    df['years_since_baseline'] = df.groupby('species_common_name')['year'].transform(
        lambda x: x - x.min()
    )
    print("✓ years_since_baseline: Time elapsed since first record")
    
    # ---------------------------------------------------------------------
    # FEATURE 4: Population Relative to Peak
    # WHY: Shows how far current population is from historical maximum,
    #      critical for understanding recovery or decline status
    # ---------------------------------------------------------------------
    df['population_peak'] = df.groupby('species_common_name')['population'].transform('max')
    df['population_relative_to_peak'] = df['population'] / df['population_peak']
    print("✓ population_relative_to_peak: Current population / Historical maximum")
    
    # ---------------------------------------------------------------------
    # FEATURE 5: Population Relative to Baseline
    # WHY: Shows growth/decline from starting point, useful for
    #      assessing conservation program effectiveness
    # ---------------------------------------------------------------------
    df['population_baseline'] = df.groupby('species_common_name')['population'].transform('first')
    df['population_relative_to_baseline'] = df['population'] / df['population_baseline']
    print("✓ population_relative_to_baseline: Current / Starting population")
    
    # ---------------------------------------------------------------------
    # FEATURE 6: Cumulative Change Rate
    # WHY: Accumulates change over time, showing total growth/decline
    #      trajectory rather than just point-in-time changes
    # ---------------------------------------------------------------------
    df['cumulative_change_rate'] = df.groupby('species_common_name')['population_change_rate'].transform(
        lambda x: x.cumsum()
    )
    print("✓ cumulative_change_rate: Running sum of year-over-year changes")
    
    # ---------------------------------------------------------------------
    # FEATURE 7: Trend Direction Indicator
    # WHY: Binary flags for quick trend identification in models
    # ---------------------------------------------------------------------
    df['is_declining'] = (df['population_change_rate'] < 0).astype(int)
    df['is_growing'] = (df['population_change_rate'] > 0).astype(int)
    print("✓ is_declining / is_growing: Binary trend direction flags")
    
    # ---------------------------------------------------------------------
    # FEATURE 8: Conservation Urgency Score
    # WHY: Combines IUCN status with decline rate for composite risk metric
    # ---------------------------------------------------------------------
    df['conservation_urgency'] = df['iucn_status_numeric'] * (1 + abs(df['population_change_rate']) / 100)
    print("✓ conservation_urgency: IUCN status × change rate composite")
    
    print()
    return df


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Categorical Variables
    =============================
    
    Machine learning models require numerical inputs. This function converts
    categorical variables into numerical representations using appropriate
    encoding strategies.
    
    ENCODING STRATEGIES:
    -------------------
    1. LABEL ENCODING: For ordinal categories (iucn_status, risk_level)
       where there's a natural order.
    
    2. ONE-HOT ENCODING: For nominal categories (taxonomic_group, region)
       where no natural order exists.
    
    WHY ENCODING IS IMPORTANT:
    -------------------------
    - XGBoost can handle categorical features natively but performs better
      with properly encoded numeric features.
    - LSTM requires fully numeric input tensors.
    - Proper encoding preserves the semantic meaning of categories.
    """
    print("=" * 70)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 70)
    
    # Store encoders for potential inverse transformation
    encoders = {}
    
    # -------------------------------------------------------------------------
    # Label Encoding for Ordinal Variables
    # -------------------------------------------------------------------------
    
    # Overall Trend (ordinal: strong_decline < moderate_decline < stable < moderate_recovery < strong_recovery)
    trend_order = {
        'strong_decline': 0,
        'moderate_decline': 1, 
        'stable': 2,
        'moderate_recovery': 3,
        'strong_recovery': 4
    }
    df['overall_trend_encoded'] = df['overall_trend'].map(trend_order)
    # Handle any unmapped values
    df['overall_trend_encoded'] = df['overall_trend_encoded'].fillna(2)  # Default to stable
    print(f"✓ overall_trend → overall_trend_encoded (ordinal: 0-4)")
    
    # Risk Level (ordinal: Low < Medium < High)
    risk_order = {'Low': 0, 'Medium': 1, 'High': 2}
    df['risk_level_encoded'] = df['risk_level'].map(risk_order)
    df['risk_level_encoded'] = df['risk_level_encoded'].fillna(1)  # Default to Medium
    print(f"✓ risk_level → risk_level_encoded (ordinal: 0-2)")
    
    # -------------------------------------------------------------------------
    # Label Encoding for Nominal Variables (with stored encoders)
    # -------------------------------------------------------------------------
    
    # Taxonomic Group
    le_taxon = LabelEncoder()
    df['taxonomic_group_encoded'] = le_taxon.fit_transform(df['taxonomic_group'])
    encoders['taxonomic_group'] = le_taxon
    print(f"✓ taxonomic_group → taxonomic_group_encoded ({len(le_taxon.classes_)} classes)")
    
    # Region
    le_region = LabelEncoder()
    df['region_encoded'] = le_region.fit_transform(df['region'])
    encoders['region'] = le_region
    print(f"✓ region → region_encoded ({len(le_region.classes_)} classes)")
    
    # Species (for species-specific modeling)
    le_species = LabelEncoder()
    df['species_encoded'] = le_species.fit_transform(df['species_common_name'])
    encoders['species'] = le_species
    print(f"✓ species_common_name → species_encoded ({len(le_species.classes_)} classes)")
    
    # -------------------------------------------------------------------------
    # One-Hot Encoding for Nominal Variables (optional alternative)
    # -------------------------------------------------------------------------
    
    # Create dummy variables for taxonomic group
    taxon_dummies = pd.get_dummies(df['taxonomic_group'], prefix='taxon')
    df = pd.concat([df, taxon_dummies], axis=1)
    print(f"✓ Created {len(taxon_dummies.columns)} one-hot columns for taxonomic_group")
    
    # Create dummy variables for region
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    print(f"✓ Created {len(region_dummies.columns)} one-hot columns for region")
    
    print()
    return df


def prepare_forecast_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Dataset for Prophet Forecasting Model
    ==============================================
    
    Prophet requires a specific format:
    - Column 'ds': datetime column (date/timestamp)
    - Column 'y': target variable (population)
    
    ADDITIONAL FEATURES FOR PROPHET:
    - Regressors can be added for improved accuracy
    - One dataset per species (Prophet models single time series)
    
    WHY THIS FORMAT:
    ----------------
    - Prophet is designed for univariate time-series forecasting
    - Uses additive/multiplicative decomposition for trend + seasonality
    - Each species requires a separate Prophet model instance
    """
    print("=" * 70)
    print("PREPARING FORECAST DATASET (Prophet)")
    print("=" * 70)
    
    # Select relevant columns for forecasting
    forecast_cols = [
        'species_common_name',
        'species_encoded',
        'year',
        'population',
        'population_rolling_avg',
        'iucn_status_numeric',
        'taxonomic_group_encoded',
        'region_encoded',
        'population_relative_to_peak',
        'is_interpolated'
    ]
    
    forecast_df = df[forecast_cols].copy()
    
    # Create Prophet-compatible datetime column
    forecast_df['ds'] = pd.to_datetime(forecast_df['year'], format='%Y')
    forecast_df['y'] = forecast_df['population']
    
    # Create log-transformed target for multiplicative modeling
    forecast_df['y_log'] = np.log1p(forecast_df['population'])
    
    # Create smoothed target
    forecast_df['y_smooth'] = forecast_df['population_rolling_avg']
    
    print(f"✓ Created Prophet-compatible columns: 'ds' (datetime), 'y' (target)")
    print(f"✓ Added 'y_log' for log-transformed predictions")
    print(f"✓ Added 'y_smooth' for smoothed predictions")
    print(f"✓ Total records: {len(forecast_df)}")
    print(f"✓ Species count: {forecast_df['species_common_name'].nunique()}")
    
    # Save forecast dataset
    forecast_df.to_csv(FORECAST_DATA_PATH, index=False)
    print(f"✓ Saved to: {FORECAST_DATA_PATH.name}")
    print()
    
    return forecast_df


def prepare_trend_dataset(df: pd.DataFrame, sequence_length: int = 5) -> pd.DataFrame:
    """
    Prepare Dataset for LSTM Trend Detection Model
    ===============================================
    
    LSTM models require sequential data organized as:
    - Input: Sequence of features over time (e.g., 5 consecutive years)
    - Output: Trend classification for the sequence
    
    FEATURES SELECTED FOR LSTM:
    - population_change_rate: Primary trend indicator
    - population_relative_to_peak: Position relative to maximum
    - population_cv: Volatility measure
    - cumulative_change_rate: Long-term trajectory
    
    WHY THESE FEATURES:
    ------------------
    1. LSTM learns temporal patterns from sequences
    2. Normalized features prevent scale bias
    3. Multiple features capture different aspects of population dynamics
    4. Change rates are more informative than absolute values for trend detection
    
    TARGET VARIABLE:
    - overall_trend_encoded: 5-class classification
      (strong_decline, moderate_decline, stable, moderate_recovery, strong_recovery)
    """
    print("=" * 70)
    print("PREPARING TREND DATASET (LSTM)")
    print("=" * 70)
    
    # Select features for LSTM input
    lstm_features = [
        'species_common_name',
        'species_encoded',
        'year',
        'population_change_rate',
        'population_relative_to_peak',
        'population_relative_to_baseline',
        'population_cv',
        'cumulative_change_rate',
        'is_declining',
        'is_growing',
        'overall_trend',
        'overall_trend_encoded',
        'iucn_status_numeric',
        'taxonomic_group_encoded'
    ]
    
    trend_df = df[lstm_features].copy()
    
    # Normalize features for LSTM (scale to 0-1 range)
    scaler = MinMaxScaler()
    numeric_cols = [
        'population_change_rate',
        'population_relative_to_peak',
        'population_relative_to_baseline',
        'population_cv',
        'cumulative_change_rate'
    ]
    
    trend_df[[f'{col}_scaled' for col in numeric_cols]] = scaler.fit_transform(
        trend_df[numeric_cols]
    )
    
    # Create sequence identifier for LSTM windowing
    trend_df['sequence_id'] = trend_df.groupby('species_common_name').cumcount()
    
    print(f"✓ Selected {len(lstm_features)} features for LSTM")
    print(f"✓ Normalized {len(numeric_cols)} numeric features to [0, 1] range")
    print(f"✓ Created sequence_id for temporal windowing")
    print(f"✓ Target classes: {trend_df['overall_trend'].unique().tolist()}")
    print(f"✓ Total records: {len(trend_df)}")
    
    # Save trend dataset
    trend_df.to_csv(TREND_DATA_PATH, index=False)
    print(f"✓ Saved to: {TREND_DATA_PATH.name}")
    print()
    
    return trend_df


def prepare_classification_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Dataset for XGBoost Risk Classification Model
    ======================================================
    
    XGBoost performs multi-class classification to predict risk level
    (High, Medium, Low) based on species characteristics and population metrics.
    
    FEATURES SELECTED FOR XGBOOST:
    - Population metrics (current, rolling average, relative values)
    - Trend indicators (change rate, cumulative change)
    - Species characteristics (taxonomic group, region)
    - Conservation status
    
    WHY THESE FEATURES:
    ------------------
    1. XGBoost excels with tabular data and mixed feature types
    2. Can capture non-linear relationships between features
    3. Feature importance analysis helps explain predictions
    4. Handles categorical features efficiently with encoding
    
    TARGET VARIABLE:
    - risk_level_encoded: 3-class classification (0=Low, 1=Medium, 2=High)
    """
    print("=" * 70)
    print("PREPARING CLASSIFICATION DATASET (XGBoost)")
    print("=" * 70)
    
    # Select features for XGBoost
    xgb_features = [
        # Identifiers
        'species_common_name',
        'species_encoded',
        'year',
        
        # Population metrics
        'population',
        'population_rolling_avg',
        'population_rolling_std',
        'population_cv',
        'population_change_rate',
        'population_relative_to_peak',
        'population_relative_to_baseline',
        'cumulative_change_rate',
        
        # Trend indicators
        'is_declining',
        'is_growing',
        'overall_trend_encoded',
        
        # Species characteristics
        'iucn_status_numeric',
        'taxonomic_group_encoded',
        'region_encoded',
        'years_since_baseline',
        
        # Composite scores
        'conservation_urgency',
        
        # Target variable
        'risk_level',
        'risk_level_encoded'
    ]
    
    # Add one-hot encoded columns
    taxon_cols = [col for col in df.columns if col.startswith('taxon_')]
    region_cols = [col for col in df.columns if col.startswith('region_')]
    xgb_features.extend(taxon_cols)
    xgb_features.extend(region_cols)
    
    # Only include columns that exist
    xgb_features = [col for col in xgb_features if col in df.columns]
    
    classification_df = df[xgb_features].copy()
    
    # Create additional aggregate features per species
    species_stats = df.groupby('species_common_name').agg({
        'population': ['mean', 'std', 'min', 'max'],
        'population_change_rate': ['mean', 'std'],
        'is_declining': 'sum'
    }).reset_index()
    
    species_stats.columns = [
        'species_common_name',
        'species_pop_mean', 'species_pop_std', 'species_pop_min', 'species_pop_max',
        'species_change_mean', 'species_change_std',
        'species_decline_years'
    ]
    
    classification_df = classification_df.merge(species_stats, on='species_common_name', how='left')
    
    print(f"✓ Selected {len(xgb_features)} base features for XGBoost")
    print(f"✓ Added 7 species-level aggregate features")
    print(f"✓ Total features: {len(classification_df.columns)}")
    print(f"✓ Target distribution:")
    for level, count in classification_df['risk_level'].value_counts().items():
        print(f"      {level}: {count} records ({count/len(classification_df)*100:.1f}%)")
    
    # Save classification dataset
    classification_df.to_csv(CLASSIFICATION_DATA_PATH, index=False)
    print(f"✓ Saved to: {CLASSIFICATION_DATA_PATH.name}")
    print()
    
    return classification_df


def generate_feature_summary(df: pd.DataFrame) -> None:
    """Generate a summary of all engineered features."""
    print("=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    
    print("\n📊 ENGINEERED FEATURES:")
    print("-" * 70)
    
    feature_descriptions = {
        'population_change_rate': 'Year-over-year % change in population',
        'population_rolling_avg': '3-year moving average of population',
        'population_rolling_std': '3-year rolling standard deviation',
        'population_cv': 'Coefficient of variation (std/mean)',
        'years_since_baseline': 'Years elapsed since first observation',
        'population_relative_to_peak': 'Current pop / Historical maximum',
        'population_relative_to_baseline': 'Current pop / Starting population',
        'cumulative_change_rate': 'Cumulative sum of change rates',
        'is_declining': 'Binary flag: 1 if declining, 0 otherwise',
        'is_growing': 'Binary flag: 1 if growing, 0 otherwise',
        'conservation_urgency': 'IUCN status × change rate composite',
        'overall_trend_encoded': 'Ordinal encoding of trend (0-4)',
        'risk_level_encoded': 'Ordinal encoding of risk (0-2)',
        'taxonomic_group_encoded': 'Label encoding of taxonomic group',
        'region_encoded': 'Label encoding of region',
        'species_encoded': 'Label encoding of species name'
    }
    
    for feature, description in feature_descriptions.items():
        if feature in df.columns:
            print(f"   • {feature}")
            print(f"     └─ {description}")
    
    print("\n📁 OUTPUT DATASETS:")
    print("-" * 70)
    print(f"   1. {ENGINEERED_DATA_PATH.name}")
    print(f"      └─ Complete dataset with all engineered features")
    print(f"   2. {FORECAST_DATA_PATH.name}")
    print(f"      └─ Prophet-ready format with 'ds' and 'y' columns")
    print(f"   3. {TREND_DATA_PATH.name}")
    print(f"      └─ LSTM-ready with scaled features and sequence IDs")
    print(f"   4. {CLASSIFICATION_DATA_PATH.name}")
    print(f"      └─ XGBoost-ready with all features and risk labels")
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE ✓")
    print("=" * 70)


def main():
    """Execute the complete feature engineering pipeline."""
    print("\n" + "=" * 70)
    print("    WILDGUARD AI - FEATURE ENGINEERING PIPELINE")
    print("=" * 70 + "\n")
    
    # Load cleaned data
    df = load_cleaned_data()
    
    # Create engineered features
    df = create_population_change_rate(df)
    df = create_rolling_average(df, window=3)
    df = create_additional_features(df)
    df = encode_categorical_variables(df)
    
    # Save complete engineered dataset
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"✓ Saved complete engineered dataset: {ENGINEERED_DATA_PATH.name}\n")
    
    # Prepare task-specific datasets
    forecast_df = prepare_forecast_dataset(df)
    trend_df = prepare_trend_dataset(df)
    classification_df = prepare_classification_dataset(df)
    
    # Generate summary
    generate_feature_summary(df)
    
    return df, forecast_df, trend_df, classification_df


if __name__ == "__main__":
    engineered_df, forecast_df, trend_df, classification_df = main()
