"""
WildGuard AI - Data Validator
==============================
Validates uploaded CSV files and merges approved data into the raw dataset.
Used by the Data Management tab in the dashboard.

Author: WildGuard AI Project Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_wildlife_data.csv"

REQUIRED_COLUMNS = [
    'species_common_name',
    'species_scientific_name',
    'taxonomic_group',
    'region',
    'iucn_status',
    'year',
    'population',
]

VALID_IUCN_CODES = ['CR', 'EN', 'VU', 'NT', 'LC']

VALID_TAXONOMIC_GROUPS = [
    'Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish',
    'Marine Mammal', 'Aquatic Mammal'
]


@dataclass
class ValidationResult:
    """Container for validation outcomes."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_status: List[str] = field(default_factory=list)  # 'valid', 'warning', 'error' per row


@dataclass
class MergeResult:
    """Container for merge outcomes."""
    success: bool = False
    new_records: int = 0
    updated_records: int = 0
    skipped_records: int = 0
    message: str = ""


class DataValidator:
    """
    Validates and processes uploaded wildlife census CSV files.
    
    Validation checks:
    1. Required columns present
    2. IUCN status codes valid
    3. Population values numeric and positive
    4. Year values numeric and reasonable
    5. Duplicate detection against existing data
    """
    
    def __init__(self):
        self.existing_data = None
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load the current raw_wildlife_data.csv for duplicate checking."""
        if RAW_DATA_PATH.exists():
            self.existing_data = pd.read_csv(RAW_DATA_PATH)
        else:
            self.existing_data = pd.DataFrame()
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks on the uploaded dataframe.
        
        Parameters:
            df: Uploaded dataframe to validate
            
        Returns:
            ValidationResult with errors, warnings, and cleaned data
        """
        result = ValidationResult()
        
        # --- Check 1: Required columns ---
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            result.is_valid = False
            result.errors.append(
                f"❌ Missing required columns: {', '.join(missing_cols)}"
            )
            return result
        
        result.info.append(f"✅ All {len(REQUIRED_COLUMNS)} required columns present")
        
        # Work with a copy
        cleaned = df.copy()
        row_status = ['valid'] * len(cleaned)
        
        # --- Check 2: Strip whitespace from string columns ---
        str_cols = ['species_common_name', 'species_scientific_name', 
                    'taxonomic_group', 'region', 'iucn_status']
        for col in str_cols:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].astype(str).str.strip()
        
        # --- Check 3: IUCN status codes ---
        cleaned['iucn_status'] = cleaned['iucn_status'].str.upper()
        invalid_iucn = cleaned[~cleaned['iucn_status'].isin(VALID_IUCN_CODES)]
        if len(invalid_iucn) > 0:
            bad_codes = invalid_iucn['iucn_status'].unique().tolist()
            result.errors.append(
                f"❌ Invalid IUCN codes found: {bad_codes} "
                f"(valid: {VALID_IUCN_CODES})"
            )
            for idx in invalid_iucn.index:
                row_status[idx] = 'error'
            result.is_valid = False
        else:
            result.info.append("✅ All IUCN status codes are valid")
        
        # --- Check 4: Population values ---
        # Convert to numeric, coercing errors
        cleaned['population'] = pd.to_numeric(cleaned['population'], errors='coerce')
        null_pop = cleaned['population'].isna()
        if null_pop.any():
            result.errors.append(
                f"❌ {null_pop.sum()} rows have non-numeric population values"
            )
            for idx in cleaned[null_pop].index:
                row_status[idx] = 'error'
            result.is_valid = False
        
        negative_pop = cleaned['population'] < 0
        if negative_pop.any():
            result.errors.append(
                f"❌ {negative_pop.sum()} rows have negative population values"
            )
            for idx in cleaned[negative_pop].index:
                row_status[idx] = 'error'
            result.is_valid = False
        
        if not null_pop.any() and not negative_pop.any():
            result.info.append("✅ All population values are valid positive numbers")
        
        # --- Check 5: Year values ---
        cleaned['year'] = pd.to_numeric(cleaned['year'], errors='coerce')
        null_year = cleaned['year'].isna()
        if null_year.any():
            result.errors.append(
                f"❌ {null_year.sum()} rows have non-numeric year values"
            )
            for idx in cleaned[null_year].index:
                row_status[idx] = 'error'
            result.is_valid = False
        else:
            # Check year range sanity (1900-2100)
            bad_years = (cleaned['year'] < 1900) | (cleaned['year'] > 2100)
            if bad_years.any():
                result.warnings.append(
                    f"⚠️ {bad_years.sum()} rows have unusual year values "
                    f"(outside 1900-2100)"
                )
                for idx in cleaned[bad_years].index:
                    if row_status[idx] == 'valid':
                        row_status[idx] = 'warning'
            else:
                result.info.append("✅ All year values are within expected range")
        
        # --- Check 6: Duplicate detection ---
        duplicates = self.check_duplicates(cleaned)
        if len(duplicates) > 0:
            result.warnings.append(
                f"⚠️ {len(duplicates)} records already exist in the dataset "
                f"(same species + year). These will overwrite existing data if approved."
            )
            for idx in duplicates:
                if row_status[idx] == 'valid':
                    row_status[idx] = 'warning'
        else:
            result.info.append("✅ No duplicate records found")
        
        # --- Check 7: Species count summary ---
        n_species = cleaned['species_common_name'].nunique()
        n_records = len(cleaned)
        year_range = f"{int(cleaned['year'].min())}-{int(cleaned['year'].max())}" if not cleaned['year'].isna().all() else "N/A"
        result.info.append(
            f"📊 Summary: {n_records} records for {n_species} species, "
            f"years {year_range}"
        )
        
        # Fill optional columns with defaults
        if 'is_interpolated' not in cleaned.columns:
            cleaned['is_interpolated'] = False
        if 'data_source' not in cleaned.columns:
            cleaned['data_source'] = 'User Upload'
        
        # Compute derived columns that raw_wildlife_data.csv expects
        cleaned = self._compute_derived_columns(cleaned)
        
        result.cleaned_df = cleaned
        result.row_status = row_status
        
        return result
    
    def check_duplicates(self, df: pd.DataFrame) -> List[int]:
        """
        Find rows that duplicate existing data (same species + year).
        
        Returns:
            List of row indices that are duplicates
        """
        if self.existing_data is None or len(self.existing_data) == 0:
            return []
        
        duplicate_indices = []
        existing_keys = set(
            zip(
                self.existing_data['species_common_name'].str.strip().str.title(),
                self.existing_data['year'].astype(int)
            )
        )
        
        for idx, row in df.iterrows():
            try:
                key = (str(row['species_common_name']).strip().title(), int(row['year']))
                if key in existing_keys:
                    duplicate_indices.append(idx)
            except (ValueError, TypeError):
                continue
        
        return duplicate_indices
    
    def _compute_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the derived columns expected in raw_wildlife_data.csv:
        population_change, population_pct_change, population_rolling_3yr,
        population_rolling_5yr, overall_trend, risk_level
        """
        df = df.sort_values(['species_common_name', 'year']).reset_index(drop=True)
        
        # Population change (year-over-year absolute change)
        df['population_change'] = df.groupby('species_common_name')['population'].diff()
        
        # Population percent change
        df['population_pct_change'] = df.groupby('species_common_name')['population'].pct_change() * 100
        
        # Rolling averages
        df['population_rolling_3yr'] = df.groupby('species_common_name')['population'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['population_rolling_5yr'] = df.groupby('species_common_name')['population'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Overall trend based on change rate
        def get_trend(pct_change):
            if pd.isna(pct_change):
                return 'stable'
            if pct_change < -10:
                return 'strong_decline'
            elif pct_change < -2:
                return 'moderate_decline'
            elif pct_change <= 5:
                return 'stable'
            elif pct_change <= 15:
                return 'moderate_recovery'
            else:
                return 'strong_recovery'
        
        df['overall_trend'] = df['population_pct_change'].apply(get_trend)
        
        # Risk level based on IUCN status
        risk_mapping = {'CR': 'High', 'EN': 'Medium', 'VU': 'Low', 'NT': 'Low', 'LC': 'Low'}
        df['risk_level'] = df['iucn_status'].map(risk_mapping).fillna('Medium')
        
        return df
    
    def merge_data(self, new_df: pd.DataFrame, include_mask: list = None) -> MergeResult:
        """
        Merge approved rows into raw_wildlife_data.csv.
        
        Parameters:
            new_df: Validated dataframe to merge
            include_mask: List of booleans indicating which rows to include
            
        Returns:
            MergeResult with counts and status
        """
        result = MergeResult()
        
        try:
            # Apply inclusion mask
            if include_mask is not None:
                new_df = new_df[include_mask].copy()
            
            if len(new_df) == 0:
                result.message = "No rows selected for merge."
                return result
            
            # Load existing data
            if RAW_DATA_PATH.exists():
                existing = pd.read_csv(RAW_DATA_PATH)
            else:
                existing = pd.DataFrame()
            
            # Determine which are new vs updates
            new_count = 0
            update_count = 0
            
            if len(existing) > 0:
                existing_keys = set(
                    zip(
                        existing['species_common_name'].str.strip(),
                        existing['year'].astype(int)
                    )
                )
                
                for _, row in new_df.iterrows():
                    key = (str(row['species_common_name']).strip(), int(row['year']))
                    if key in existing_keys:
                        update_count += 1
                        # Remove old record
                        mask = ~(
                            (existing['species_common_name'].str.strip() == key[0]) & 
                            (existing['year'].astype(int) == key[1])
                        )
                        existing = existing[mask]
                    else:
                        new_count += 1
            else:
                new_count = len(new_df)
            
            # Ensure columns match
            expected_cols = [
                'species_common_name', 'species_scientific_name', 'taxonomic_group',
                'region', 'iucn_status', 'year', 'population', 'is_interpolated',
                'data_source', 'population_change', 'population_pct_change',
                'population_rolling_3yr', 'population_rolling_5yr', 
                'overall_trend', 'risk_level'
            ]
            
            # Only include columns that exist in both
            merge_cols = [c for c in expected_cols if c in new_df.columns]
            new_to_merge = new_df[merge_cols].copy()
            
            # Add any missing columns with NaN
            for col in expected_cols:
                if col not in new_to_merge.columns:
                    new_to_merge[col] = np.nan
            
            # Concatenate
            merged = pd.concat([existing, new_to_merge[expected_cols]], ignore_index=True)
            merged = merged.sort_values(['species_common_name', 'year']).reset_index(drop=True)
            
            # Save
            merged.to_csv(RAW_DATA_PATH, index=False)
            
            # Reload existing data for future checks
            self._load_existing_data()
            
            result.success = True
            result.new_records = new_count
            result.updated_records = update_count
            result.message = (
                f"✅ Successfully merged! "
                f"Added {new_count} new records, updated {update_count} existing records."
            )
            
        except Exception as e:
            result.success = False
            result.message = f"❌ Merge failed: {str(e)}"
        
        return result
