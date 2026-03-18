"""
WildGuard AI - Data Preprocessing Script
=========================================
This script performs data cleaning and preprocessing on the raw wildlife
population dataset to prepare it for machine learning model training.

Author: WildGuard AI Project Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent
RAW_DATA_PATH = DATA_DIR / "raw_wildlife_data.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_wildlife_data.csv"

# Conservation status mapping (IUCN Red List categories to numeric values)
# Higher values indicate greater extinction risk
CONSERVATION_STATUS_MAP = {
    'EX': 5,   # Extinct
    'CR': 4,   # Critically Endangered
    'EN': 3,   # Endangered
    'VU': 2,   # Vulnerable
    'NT': 1,   # Near Threatened
    'LC': 0    # Least Concern
}


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Step 1: Load the raw wildlife dataset from CSV file.
    
    This function reads the CSV file containing population data for multiple
    wildlife species. The data includes temporal records with population counts,
    conservation status, and derived features.
    
    Parameters:
        filepath (Path): Path to the raw CSV file
        
    Returns:
        pd.DataFrame: Raw dataframe loaded from CSV
    """
    print("=" * 60)
    print("STEP 1: Loading Raw Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {len(df)} records from {filepath.name}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Species count: {df['species_common_name'].nunique()}")
    print(f"✓ Year range: {df['year'].min()} - {df['year'].max()}")
    print()
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Handle missing population values through interpolation or removal.
    
    This function addresses missing values in the population column using a
    two-pronged approach:
    1. For gaps within a species' time series: Linear interpolation
    2. For gaps at the boundaries: Forward-fill or backward-fill
    3. Remove any remaining records with null population values
    
    Linear interpolation is chosen as it preserves the temporal continuity
    essential for time-series forecasting models.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with potential missing values
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    print("=" * 60)
    print("STEP 2: Handling Missing Values")
    print("=" * 60)
    
    # Check initial missing values
    missing_before = df['population'].isna().sum()
    print(f"✓ Missing population values before: {missing_before}")
    
    # Sort by species and year for proper interpolation
    df = df.sort_values(['species_common_name', 'year']).reset_index(drop=True)
    
    # Apply interpolation within each species group
    # This ensures we don't interpolate across different species
    df['population'] = df.groupby('species_common_name')['population'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # Handle any remaining NaN values in numeric columns
    numeric_cols = ['population_change', 'population_pct_change', 
                    'population_rolling_3yr', 'population_rolling_5yr']
    
    for col in numeric_cols:
        if col in df.columns:
            # Fill missing derived features using forward-fill within species
            df[col] = df.groupby('species_common_name')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
    
    # Remove any remaining records with missing population (edge cases)
    records_before = len(df)
    df = df.dropna(subset=['population'])
    records_removed = records_before - len(df)
    
    missing_after = df['population'].isna().sum()
    print(f"✓ Missing population values after: {missing_after}")
    print(f"✓ Records removed due to unrecoverable missing values: {records_removed}")
    print()
    
    return df


def standardize_species_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Standardize species names for consistency.
    
    This function ensures uniformity in species naming by:
    1. Stripping leading/trailing whitespace
    2. Converting to title case for common names
    3. Converting to proper scientific nomenclature format for scientific names
    4. Removing duplicate spaces
    
    Standardization is crucial for accurate grouping and joining operations
    in downstream analysis.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with potentially inconsistent names
        
    Returns:
        pd.DataFrame: Dataframe with standardized species names
    """
    print("=" * 60)
    print("STEP 3: Standardizing Species Names")
    print("=" * 60)
    
    # Store original unique names for comparison
    original_common = df['species_common_name'].unique()
    original_scientific = df['species_scientific_name'].unique()
    
    # Standardize common names: strip whitespace, title case
    df['species_common_name'] = (
        df['species_common_name']
        .str.strip()                    # Remove leading/trailing whitespace
        .str.replace(r'\s+', ' ', regex=True)  # Remove duplicate spaces
        .str.title()                    # Convert to Title Case
    )
    
    # Standardize scientific names: strip whitespace, proper format
    # Scientific names should be: Genus species (first word capitalized, rest lowercase)
    def format_scientific_name(name):
        if pd.isna(name):
            return name
        name = str(name).strip()
        name = ' '.join(name.split())  # Remove duplicate spaces
        parts = name.split()
        if len(parts) >= 2:
            # Genus capitalized, species lowercase
            return parts[0].capitalize() + ' ' + ' '.join(p.lower() for p in parts[1:])
        return name.capitalize()
    
    df['species_scientific_name'] = df['species_scientific_name'].apply(format_scientific_name)
    
    # Standardize taxonomic group
    df['taxonomic_group'] = df['taxonomic_group'].str.strip().str.title()
    
    # Standardize region
    df['region'] = df['region'].str.strip().str.title()
    
    # Report changes
    new_common = df['species_common_name'].unique()
    print(f"✓ Unique common names: {len(original_common)} → {len(new_common)}")
    print(f"✓ Sample standardized names: {list(new_common[:5])}")
    print()
    
    return df


def convert_conservation_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Convert conservation status to numeric values.
    
    This function maps IUCN Red List categorical status codes to ordinal
    numeric values, enabling their use as features in machine learning models.
    
    The mapping follows the IUCN threat hierarchy:
        EX (Extinct) = 5         - Highest risk
        CR (Critically Endangered) = 4
        EN (Endangered) = 3
        VU (Vulnerable) = 2
        NT (Near Threatened) = 1
        LC (Least Concern) = 0   - Lowest risk
    
    The original categorical column is preserved as 'iucn_status', and a new
    column 'iucn_status_numeric' is created for model training.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with categorical conservation status
        
    Returns:
        pd.DataFrame: Dataframe with additional numeric status column
    """
    print("=" * 60)
    print("STEP 4: Converting Conservation Status to Numeric")
    print("=" * 60)
    
    # Standardize status codes to uppercase
    df['iucn_status'] = df['iucn_status'].str.strip().str.upper()
    
    # Create numeric mapping column
    df['iucn_status_numeric'] = df['iucn_status'].map(CONSERVATION_STATUS_MAP)
    
    # Check for any unmapped values
    unmapped = df[df['iucn_status_numeric'].isna()]['iucn_status'].unique()
    if len(unmapped) > 0:
        print(f"⚠ Warning: Unmapped status codes found: {unmapped}")
        # Assign a default value for unmapped codes (treating as Data Deficient)
        df['iucn_status_numeric'] = df['iucn_status_numeric'].fillna(-1)
    
    # Display mapping summary
    status_counts = df.groupby(['iucn_status', 'iucn_status_numeric']).size()
    print("✓ Conservation Status Mapping Applied:")
    print("-" * 40)
    for (status, numeric), count in status_counts.items():
        print(f"   {status} → {int(numeric):2d}  ({count:4d} records)")
    print()
    
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Sort data by Species and Year.
    
    Proper sorting is essential for:
    1. Time-series analysis and forecasting
    2. Correct calculation of lagged features
    3. Reproducibility of train-test splits
    4. Visual inspection and debugging
    
    The data is sorted primarily by species name (alphabetically) and
    secondarily by year (chronologically).
    
    Parameters:
        df (pd.DataFrame): Unsorted or partially sorted dataframe
        
    Returns:
        pd.DataFrame: Sorted dataframe with reset index
    """
    print("=" * 60)
    print("STEP 5: Sorting Data by Species and Year")
    print("=" * 60)
    
    df = df.sort_values(
        by=['species_common_name', 'year'],
        ascending=[True, True]
    ).reset_index(drop=True)
    
    print("✓ Data sorted by: species_common_name (A-Z), year (ascending)")
    print(f"✓ First record: {df.iloc[0]['species_common_name']} ({df.iloc[0]['year']})")
    print(f"✓ Last record: {df.iloc[-1]['species_common_name']} ({df.iloc[-1]['year']})")
    print()
    
    return df


def save_cleaned_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Step 6: Save the cleaned dataset to a new CSV file.
    
    The cleaned dataset is saved with the following considerations:
    1. UTF-8 encoding for international character support
    2. No index column to avoid redundant data
    3. Float precision limited to 6 decimal places
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe to save
        filepath (Path): Destination path for the cleaned CSV
    """
    print("=" * 60)
    print("STEP 6: Saving Cleaned Data")
    print("=" * 60)
    
    # Reorder columns for better readability
    column_order = [
        'species_common_name',
        'species_scientific_name', 
        'taxonomic_group',
        'region',
        'iucn_status',
        'iucn_status_numeric',
        'year',
        'population',
        'is_interpolated',
        'data_source',
        'population_change',
        'population_pct_change',
        'population_rolling_3yr',
        'population_rolling_5yr',
        'overall_trend',
        'risk_level'
    ]
    
    # Only include columns that exist in the dataframe
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]
    
    # Save to CSV
    df.to_csv(filepath, index=False, encoding='utf-8', float_format='%.6f')
    
    print(f"✓ Saved cleaned data to: {filepath.name}")
    print(f"✓ Final record count: {len(df)}")
    print(f"✓ Final column count: {len(df.columns)}")
    print(f"✓ File size: {filepath.stat().st_size / 1024:.2f} KB")
    print()


def generate_summary_report(df: pd.DataFrame) -> None:
    """
    Generate a summary report of the cleaned dataset.
    
    This function prints a comprehensive overview of the processed data,
    including statistical summaries and data quality metrics.
    """
    print("=" * 60)
    print("PREPROCESSING SUMMARY REPORT")
    print("=" * 60)
    
    print("\n📊 Dataset Overview:")
    print(f"   Total Records: {len(df)}")
    print(f"   Total Species: {df['species_common_name'].nunique()}")
    print(f"   Year Range: {df['year'].min()} - {df['year'].max()}")
    
    print("\n📈 Population Statistics:")
    print(f"   Min Population: {df['population'].min():,.0f}")
    print(f"   Max Population: {df['population'].max():,.0f}")
    print(f"   Mean Population: {df['population'].mean():,.2f}")
    
    print("\n🏷️ Conservation Status Distribution:")
    status_dist = df.groupby('iucn_status')['species_common_name'].nunique()
    for status, count in status_dist.items():
        print(f"   {status}: {count} species")
    
    print("\n🦁 Taxonomic Group Distribution:")
    taxon_dist = df.groupby('taxonomic_group')['species_common_name'].nunique()
    for taxon, count in taxon_dist.items():
        print(f"   {taxon}: {count} species")
    
    print("\n🌍 Regional Distribution:")
    region_dist = df.groupby('region')['species_common_name'].nunique()
    for region, count in region_dist.items():
        print(f"   {region}: {count} species")
    
    print("\n✅ Data Quality Metrics:")
    print(f"   Missing Values: {df.isna().sum().sum()}")
    print(f"   Duplicate Rows: {df.duplicated().sum()}")
    print(f"   Interpolated Records: {df['is_interpolated'].sum()} ({df['is_interpolated'].mean()*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE ✓")
    print("=" * 60)


def main():
    """
    Main execution function for the preprocessing pipeline.
    
    This function orchestrates the complete preprocessing workflow:
    1. Load raw data
    2. Handle missing values
    3. Standardize species names
    4. Convert conservation status to numeric
    5. Sort data
    6. Save cleaned data
    7. Generate summary report
    """
    print("\n" + "=" * 60)
    print("    WILDGUARD AI - DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")
    
    # Execute preprocessing steps
    df = load_data(RAW_DATA_PATH)
    df = handle_missing_values(df)
    df = standardize_species_names(df)
    df = convert_conservation_status(df)
    df = sort_data(df)
    save_cleaned_data(df, CLEANED_DATA_PATH)
    generate_summary_report(df)
    
    return df


if __name__ == "__main__":
    cleaned_df = main()
