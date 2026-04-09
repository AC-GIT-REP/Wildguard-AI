"""
WildGuard AI - Prophet Population Forecasting Model
====================================================
This script demonstrates population forecasting using Facebook Prophet
for endangered species time-series prediction.

Author: WildGuard AI Project Team
Date: January 2026

WHY PROPHET IS CHOSEN:
======================
1. DESIGNED FOR TIME-SERIES WITH TRENDS: Prophet excels at capturing
   non-linear growth patterns common in wildlife populations. It uses
   piecewise linear or logistic growth curves that naturally model
   population recovery or decline phases.

2. HANDLES MISSING DATA GRACEFULLY: Wildlife census data often has gaps
   (e.g., surveys every 4 years). Prophet interpolates missing points
   without requiring explicit imputation, making it robust for sparse data.

3. UNCERTAINTY QUANTIFICATION: Prophet provides confidence intervals
   (uncertainty bounds) for forecasts, which is critical for conservation
   decision-making where uncertainty must be communicated.

4. INTERPRETABLE DECOMPOSITION: The model decomposes time-series into
   trend + seasonality + holidays, allowing ecologists to understand
   what drives population changes.

5. ADDITIVE/MULTIPLICATIVE MODELING: Supports both additive (constant
   variations) and multiplicative (proportional variations) seasonality,
   suitable for populations of varying scales.

6. EASE OF USE: Minimal hyperparameter tuning required, making it ideal
   for prototype/academic projects while still producing publication-quality
   forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet (handles different installation names)
try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        print("ERROR: Prophet not installed. Run: pip install prophet")
        exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent
PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Species to forecast (demonstration)
DEMO_SPECIES = "Bengal Tiger"
FORECAST_YEARS = 5


def load_species_data(species_name: str) -> pd.DataFrame:
    """
    Load and prepare data for a specific species.
    
    Parameters:
        species_name (str): Common name of the species
        
    Returns:
        pd.DataFrame: Prophet-ready dataframe with 'ds' and 'y' columns
    """
    print("=" * 70)
    print(f"LOADING DATA FOR: {species_name}")
    print("=" * 70)
    
    # Load the forecast dataset
    df = pd.read_csv(DATA_DIR / "forecast_dataset.csv")
    
    # Filter for the selected species
    species_df = df[df['species_common_name'] == species_name].copy()
    
    if len(species_df) == 0:
        available = df['species_common_name'].unique()
        print(f"ERROR: Species '{species_name}' not found.")
        print(f"Available species: {list(available)}")
        return None
    
    # Prepare Prophet format
    # Prophet requires columns: 'ds' (datestamp) and 'y' (target)
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(species_df['year'], format='%Y'),
        'y': species_df['population'].values
    })
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    print(f"✓ Records loaded: {len(prophet_df)}")
    print(f"✓ Year range: {prophet_df['ds'].dt.year.min()} - {prophet_df['ds'].dt.year.max()}")
    print(f"✓ Population range: {prophet_df['y'].min():,.0f} - {prophet_df['y'].max():,.0f}")
    print()
    
    return prophet_df


def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on the species data.
    
    Prophet Model Configuration:
    ---------------------------
    - growth='linear': Assumes linear trend (appropriate for steady recovery)
    - yearly_seasonality=False: Wildlife data is annual, no within-year seasonality
    - weekly_seasonality=False: Not applicable for annual data
    - daily_seasonality=False: Not applicable for annual data
    - interval_width=0.95: 95% confidence interval for uncertainty
    
    Parameters:
        df (pd.DataFrame): Prophet-formatted dataframe
        
    Returns:
        Prophet: Trained Prophet model
    """
    print("=" * 70)
    print("TRAINING PROPHET MODEL")
    print("=" * 70)
    
    # Initialize Prophet model with appropriate settings
    model = Prophet(
        growth='linear',           # Linear growth trend
        yearly_seasonality=False,  # No yearly seasonality (annual data)
        weekly_seasonality=False,  # No weekly seasonality
        daily_seasonality=False,   # No daily seasonality
        interval_width=0.95,       # 95% confidence interval
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        seasonality_mode='additive'    # Additive model
    )
    
    # Fit the model
    print("✓ Fitting Prophet model...")
    model.fit(df)
    
    print(f"✓ Model trained successfully")
    print(f"✓ Detected {len(model.changepoints)} trend changepoints")
    print()
    
    return model


def forecast_population(model: Prophet, periods: int) -> pd.DataFrame:
    """
    Generate future predictions for the specified number of years.
    
    Parameters:
        model (Prophet): Trained Prophet model
        periods (int): Number of years to forecast
        
    Returns:
        pd.DataFrame: Forecast dataframe with predictions and intervals
    """
    print("=" * 70)
    print(f"FORECASTING NEXT {periods} YEARS")
    print("=" * 70)
    
    # Create future dataframe
    # freq='YS' = Year Start (annual frequency)
    future = model.make_future_dataframe(periods=periods, freq='YS')
    
    # Generate predictions
    forecast = model.predict(future)
    
    # Display forecast summary
    future_only = forecast[forecast['ds'] > model.history['ds'].max()]
    
    print("✓ Forecast Generated:")
    print("-" * 50)
    print(f"{'Year':<10} {'Predicted':<15} {'Lower 95%':<15} {'Upper 95%':<15}")
    print("-" * 50)
    
    for _, row in future_only.iterrows():
        year = row['ds'].year
        pred = row['yhat']
        lower = row['yhat_lower']
        upper = row['yhat_upper']
        print(f"{year:<10} {pred:>14,.0f} {lower:>14,.0f} {upper:>14,.0f}")
    
    print()
    return forecast


def create_forecast_plot(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    species_name: str,
    save_path: Path
) -> None:
    """
    Create visualization of observed vs forecasted population.
    
    The plot includes:
    - Historical observations (blue dots)
    - Fitted trend line (blue line)
    - Future forecast (red line)
    - 95% confidence interval (shaded area)
    
    Parameters:
        df (pd.DataFrame): Original data with observations
        forecast (pd.DataFrame): Prophet forecast output
        species_name (str): Species name for title
        save_path (Path): Path to save the plot
    """
    print("=" * 70)
    print("CREATING FORECAST VISUALIZATION")
    print("=" * 70)
    
    # Create figure with custom styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Get the last historical date
    last_historical_date = df['ds'].max()
    
    # Split forecast into historical and future
    historical_forecast = forecast[forecast['ds'] <= last_historical_date]
    future_forecast = forecast[forecast['ds'] > last_historical_date]
    
    # Plot confidence interval (full range)
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='#3498db',
        alpha=0.15,
        label='95% Confidence Interval'
    )
    
    # Plot fitted line (historical)
    ax.plot(
        historical_forecast['ds'],
        historical_forecast['yhat'],
        color='#2980b9',
        linewidth=2,
        label='Fitted Trend',
        linestyle='-'
    )
    
    # Plot forecast line (future)
    if len(future_forecast) > 0:
        # Connect historical to future
        connection_point = historical_forecast.iloc[-1]
        future_with_connection = pd.concat([
            pd.DataFrame({'ds': [connection_point['ds']], 'yhat': [connection_point['yhat']]}),
            future_forecast[['ds', 'yhat']]
        ])
        
        ax.plot(
            future_with_connection['ds'],
            future_with_connection['yhat'],
            color='#e74c3c',
            linewidth=2.5,
            label='Forecast',
            linestyle='--',
            marker='o',
            markersize=6
        )
        
        # Highlight future confidence interval
        ax.fill_between(
            future_forecast['ds'],
            future_forecast['yhat_lower'],
            future_forecast['yhat_upper'],
            color='#e74c3c',
            alpha=0.2
        )
    
    # Plot actual observations
    ax.scatter(
        df['ds'],
        df['y'],
        color='#2c3e50',
        s=80,
        zorder=5,
        label='Observed Population',
        edgecolors='white',
        linewidth=1.5
    )
    
    # Add vertical line at forecast start
    ax.axvline(
        x=last_historical_date,
        color='#7f8c8d',
        linestyle=':',
        linewidth=1.5,
        alpha=0.7
    )
    ax.text(
        last_historical_date,
        ax.get_ylim()[1] * 0.95,
        ' Forecast →',
        fontsize=10,
        color='#7f8c8d',
        ha='left',
        va='top'
    )
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Population', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Population Forecast: {species_name}\n'
        f'Prophet Time-Series Model with 95% Confidence Interval',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    ax.legend(
        loc='upper left',
        fontsize=10,
        frameon=True,
        facecolor='white',
        edgecolor='#bdc3c7'
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add annotation box with key statistics
    last_observed = df['y'].iloc[-1]
    last_forecast = future_forecast['yhat'].iloc[-1] if len(future_forecast) > 0 else last_observed
    change = ((last_forecast - last_observed) / last_observed) * 100
    
    stats_text = (
        f"Last Observed: {last_observed:,.0f} ({df['ds'].dt.year.iloc[-1]})\n"
        f"Final Forecast: {last_forecast:,.0f} ({future_forecast['ds'].dt.year.iloc[-1]})\n"
        f"Projected Change: {change:+.1f}%"
    )
    
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#bdc3c7', alpha=0.9)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"✓ Plot saved to: {save_path}")
    print()


def interpret_forecast(df: pd.DataFrame, forecast: pd.DataFrame, species_name: str) -> str:
    """
    Generate interpretation of the forecast results.
    
    Parameters:
        df (pd.DataFrame): Original data
        forecast (pd.DataFrame): Forecast results
        species_name (str): Species name
        
    Returns:
        str: Interpretation text
    """
    print("=" * 70)
    print("FORECAST INTERPRETATION")
    print("=" * 70)
    
    # Get key values
    last_historical = df['ds'].max()
    last_observed = df['y'].iloc[-1]
    first_observed = df['y'].iloc[0]
    
    future_forecast = forecast[forecast['ds'] > last_historical]
    final_prediction = future_forecast['yhat'].iloc[-1]
    final_lower = future_forecast['yhat_lower'].iloc[-1]
    final_upper = future_forecast['yhat_upper'].iloc[-1]
    
    # Calculate trends
    historical_change = ((last_observed - first_observed) / first_observed) * 100
    forecast_change = ((final_prediction - last_observed) / last_observed) * 100
    
    # Determine trend direction
    if forecast_change > 5:
        trend_direction = "INCREASING"
        trend_color = "positive"
    elif forecast_change < -5:
        trend_direction = "DECREASING"
        trend_color = "concerning"
    else:
        trend_direction = "STABLE"
        trend_color = "stable"
    
    interpretation = f"""
WHAT THE FORECAST INDICATES FOR {species_name.upper()}:
{'=' * 60}

1. HISTORICAL TREND:
   - Starting population ({df['ds'].dt.year.iloc[0]}): {first_observed:,.0f}
   - Current population ({df['ds'].dt.year.iloc[-1]}): {last_observed:,.0f}
   - Historical change: {historical_change:+.1f}%

2. PROJECTED TREND ({FORECAST_YEARS}-YEAR FORECAST):
   - Direction: {trend_direction}
   - Predicted population ({future_forecast['ds'].dt.year.iloc[-1]}): {final_prediction:,.0f}
   - Confidence interval: [{final_lower:,.0f} - {final_upper:,.0f}]
   - Projected change: {forecast_change:+.1f}%

3. CONSERVATION IMPLICATIONS:
"""
    
    if trend_direction == "INCREASING":
        interpretation += """
   ✓ The population shows a POSITIVE growth trajectory.
   ✓ Conservation efforts appear to be effective.
   ✓ Current protection measures should be maintained.
   ✓ The species is likely on a path toward recovery.
"""
    elif trend_direction == "DECREASING":
        interpretation += """
   ⚠ The population shows a DECLINING trajectory.
   ⚠ Immediate conservation intervention may be required.
   ⚠ Threat assessment and mitigation strategies needed.
   ⚠ Habitat protection and anti-poaching efforts should be intensified.
"""
    else:
        interpretation += """
   → The population appears STABLE with minimal change.
   → Current conservation efforts are maintaining population levels.
   → Continued monitoring is essential to detect any emerging threats.
   → Focus should be on long-term habitat preservation.
"""
    
    interpretation += f"""
4. UNCERTAINTY CONSIDERATIONS:
   - The 95% confidence interval spans {(final_upper - final_lower):,.0f} individuals.
   - This uncertainty reflects natural population variability and data limitations.
   - Conservation planning should consider the lower bound ({final_lower:,.0f}) as
     a precautionary estimate for risk assessment.

5. MODEL LIMITATIONS:
   - Prophet assumes continuation of historical trends.
   - External factors (climate change, policy changes, disease) are not modeled.
   - Forecast accuracy decreases with prediction horizon.
   - Regular model retraining with new census data is recommended.
"""
    
    print(interpretation)
    return interpretation


def main():
    """Execute the complete Prophet forecasting pipeline."""
    print("\n" + "=" * 70)
    print("    WILDGUARD AI - PROPHET POPULATION FORECASTING")
    print("=" * 70 + "\n")
    
    # Step 1: Load species data
    df = load_species_data(DEMO_SPECIES)
    if df is None:
        return
    
    # Step 2: Train Prophet model
    model = train_prophet_model(df)
    
    # Step 3: Generate forecast
    forecast = forecast_population(model, FORECAST_YEARS)
    
    # Step 4: Create visualization
    plot_filename = f"prophet_forecast_{DEMO_SPECIES.lower().replace(' ', '_')}.png"
    plot_path = PLOTS_DIR / plot_filename
    create_forecast_plot(df, forecast, DEMO_SPECIES, plot_path)
    
    # Step 5: Interpret results
    interpretation = interpret_forecast(df, forecast, DEMO_SPECIES)
    
    # Save forecast data
    forecast_output_path = DATA_DIR / f"forecast_results_{DEMO_SPECIES.lower().replace(' ', '_')}.csv"
    forecast.to_csv(forecast_output_path, index=False)
    print(f"✓ Forecast data saved to: {forecast_output_path.name}")
    
    print("\n" + "=" * 70)
    print("PROPHET FORECASTING COMPLETE ✓")
    print("=" * 70)
    
    return model, forecast


if __name__ == "__main__":
    model, forecast = main()
