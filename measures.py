#!/usr/bin/env python
# coding: utf-8

# ## Config

# In[10]:


"""
PICU Patient-Centric Measures Analysis
Complete harmonized implementation of 14 patient-centric measures for PICU team analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from google.cloud import bigquery
import warnings
import pickle
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set style for visualizations - NO GRID BACKGROUNDS
plt.style.use('default')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1

# Define consistent color palette
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#E63946',    # Red
    'tertiary': '#F77F00',     # Orange
    'quaternary': '#06A77D',   # Green
    'quinary': '#9D4EDD',      # Purple
    'senary': '#F72585',       # Pink
    'palette': ['#2E86AB', '#E63946', '#F77F00', '#06A77D', '#9D4EDD', '#F72585']
}

# Initialize BigQuery client
client = bigquery.Client(project="som-nero-phi-dtawfik-dt")

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def save_and_show_plot(fig, filename, dpi=100):
    """Save plot to file and display inline with standardized background and grid"""
    # Standardize figure + axes backgrounds
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        ax.set_facecolor("white")  # axes background
        # Standardize grid
        ax.grid(True,color="gray", alpha=0.3, linestyle="--")

        
    # Save and display
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

def calculate_data_completeness(df, columns):
    """Calculate percentage of non-null values for specified columns"""
    completeness = {}
    for col in columns:
        if col in df.columns:
            completeness[col] = (df[col].notna().sum() / len(df)) * 100
    return completeness

def create_summary_table(data, title="Summary Statistics"):
    """Create formatted summary statistics table"""
    summary_df = pd.DataFrame(data)
    print(f"\n{title}")
    print("="*len(title))
    print(summary_df.to_string())
    return summary_df

def add_truncation_indicator(ax, data, y_max=None, percentile=None):
    """
    Add truncation indicator to a plot.

    Parameters
    ----------
    ax : matplotlib Axes
        The subplot to annotate.
    data : pd.Series
        Data plotted on the y-axis.
    y_max : int or float, optional
        Hard truncation value for the y-axis. If provided, this overrides percentile.
    percentile : int, optional
        Percentile (e.g., 75, 95) to use for dynamic truncation detection.
    """
    if len(data) == 0:
        return False
    
    max_val = data.max()

    if y_max is not None:
        # Hard cutoff
        if max_val > y_max:
            ax.text(0.98, 0.98, f'Truncated at {y_max}', 
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    fontsize=8)
            return True

    elif percentile is not None:
        # Dynamic cutoff
        p_val = np.percentile(data.dropna(), percentile)
        if max_val > p_val * 1.1:  # signal if unusually large
            ax.text(0.98, 0.98, f'Truncated at P{percentile}', 
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    fontsize=8)
            return True
    
    return False

# Checkpoint configuration
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Create output directories
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# In[2]:


# ============================================================================
# HELPER FUNCTION DEFINITIONS
# ============================================================================

print("\nDefining helper functions...")

def get_lookback_data(df, csn, admission, current_date, window_days):
    """
    Get data for lookback window for stability calculations.
    
    Parameters:
    -----------
    df : DataFrame
        Source dataframe (df_analysis or df_primary)
    csn : str/int
        Patient CSN
    admission : int
        Admission number
    current_date : datetime
        Current date to look back from
    window_days : int
        Number of days to look back
    
    Returns:
    --------
    DataFrame : Rows within lookback window
    """
    mask = (
        (df['CSN'] == csn) &
        (df['admission_number_this_csn'] == admission) &
        (df['access_date'] < current_date) &
        (df['access_date'] >= current_date - timedelta(days=window_days))
    )
    return df[mask].sort_values('access_date')


def calculate_individual_stability(df, role, window):
    """
    Calculate individual provider stability for given role and window.
    
    Returns proportion of days in lookback window where same provider worked.
    Excludes patient-days where sufficient lookback is not available.
    """
    role_col = f'{role}_user_id'
    stability_col = f'S_{role}_{window}d'
    has_lookback_col = f'has_lookback_{window}d'
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), 
                         desc=f"Computing {stability_col}"):
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_provider = row[role_col]
        days_since_adm = row['days_since_admission']
        
        # Check if we have sufficient lookback data
        if days_since_adm < window:
            has_lookback = False
            stability = np.nan
        else:
            has_lookback = True
            
            # Get lookback data
            lookback = get_lookback_data(df, csn, admission, current_date, window)
            
            if len(lookback) == 0:
                stability = 0.0
            else:
                # Count matches
                matches = (lookback[role_col] == current_provider).sum()
                stability = matches / window
        
        results.append({
            'idx': idx,
            stability_col: stability,
            has_lookback_col: has_lookback
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def calculate_individual_streak(df, role):
    """
    Calculate consecutive days current provider has worked on patient.
    
    Returns number of consecutive days (including today).
    """
    role_col = f'{role}_user_id'
    streak_col = f'S_{role}_streak'
    
    results = []
    
    # Sort by patient, admission, date
    df_sorted = df.sort_values(['CSN', 'admission_number_this_csn', 'access_date'])
    
    for idx, row in tqdm(df_sorted.iterrows(), total=len(df_sorted),
                         desc=f"Computing {streak_col}"):
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_provider = row[role_col]
        
        # Get all prior days in this admission
        mask = (
            (df_sorted['CSN'] == csn) &
            (df_sorted['admission_number_this_csn'] == admission) &
            (df_sorted['access_date'] <= current_date)
        )
        admission_data = df_sorted[mask].sort_values('access_date', ascending=False)
        
        # Count consecutive days from today backward
        streak = 0
        for _, prior_row in admission_data.iterrows():
            if prior_row[role_col] == current_provider:
                streak += 1
            else:
                break
        
        results.append({
            'idx': idx,
            streak_col: streak
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def calculate_dyadic_stability(df, pair_roles, window):
    """
    Calculate dyadic stability for pair of roles.
    
    pair_roles: tuple like ('nurse', 'frontline')
    """
    role1, role2 = pair_roles
    role1_col = f'{role1}_user_id'
    role2_col = f'{role2}_user_id'
    
    pair_name = f"{role1[0].upper()}{role2[0].upper()}"  # e.g., "NF"
    stability_col = f'S_{pair_name}_{window}d'
    has_lookback_col = f'has_lookback_{window}d'
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc=f"Computing {stability_col}"):
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_provider1 = row[role1_col]
        current_provider2 = row[role2_col]
        days_since_adm = row['days_since_admission']
        
        # Check if we have sufficient lookback
        if days_since_adm < window:
            stability = np.nan
        else:
            lookback = get_lookback_data(df, csn, admission, current_date, window)
            
            if len(lookback) == 0:
                stability = 0.0
            else:
                # Count days where BOTH providers match
                matches = (
                    (lookback[role1_col] == current_provider1) &
                    (lookback[role2_col] == current_provider2)
                ).sum()
                stability = matches / window
        
        results.append({
            'idx': idx,
            stability_col: stability
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def calculate_dyadic_streak(df, pair_roles):
    """Calculate consecutive days pair has worked together."""
    role1, role2 = pair_roles
    role1_col = f'{role1}_user_id'
    role2_col = f'{role2}_user_id'
    
    pair_name = f"{role1[0].upper()}{role2[0].upper()}"
    streak_col = f'S_{pair_name}_streak'
    
    results = []
    df_sorted = df.sort_values(['CSN', 'admission_number_this_csn', 'access_date'])
    
    for idx, row in tqdm(df_sorted.iterrows(), total=len(df_sorted),
                         desc=f"Computing {streak_col}"):
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_provider1 = row[role1_col]
        current_provider2 = row[role2_col]
        
        mask = (
            (df_sorted['CSN'] == csn) &
            (df_sorted['admission_number_this_csn'] == admission) &
            (df_sorted['access_date'] <= current_date)
        )
        admission_data = df_sorted[mask].sort_values('access_date', ascending=False)
        
        streak = 0
        for _, prior_row in admission_data.iterrows():
            if (prior_row[role1_col] == current_provider1 and 
                prior_row[role2_col] == current_provider2):
                streak += 1
            else:
                break
        
        results.append({
            'idx': idx,
            streak_col: streak
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def calculate_team_stability(df, window):
    """Calculate complete 3-person team stability."""
    stability_col = f'S_team_{window}d'
    has_lookback_col = f'has_lookback_{window}d'
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc=f"Computing {stability_col}"):
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_nurse = row['nurse_user_id']
        current_frontline = row['frontline_user_id']
        current_attending = row['attending_user_id']
        days_since_adm = row['days_since_admission']
        
        if days_since_adm < window:
            stability = np.nan
        else:
            lookback = get_lookback_data(df, csn, admission, current_date, window)
            
            if len(lookback) == 0:
                stability = 0.0
            else:
                # Count days where ALL THREE match
                matches = (
                    (lookback['nurse_user_id'] == current_nurse) &
                    (lookback['frontline_user_id'] == current_frontline) &
                    (lookback['attending_user_id'] == current_attending)
                ).sum()
                stability = matches / window
        
        results.append({
            'idx': idx,
            stability_col: stability
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def calculate_team_streak(df):
    """Calculate consecutive days exact team has worked together."""
    streak_col = 'S_team_streak'
    
    results = []
    df_sorted = df.sort_values(['CSN', 'admission_number_this_csn', 'access_date'])
    
    for idx, row in tqdm(df_sorted.iterrows(), total=len(df_sorted),
                         desc=f"Computing {streak_col}"):
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_nurse = row['nurse_user_id']
        current_frontline = row['frontline_user_id']
        current_attending = row['attending_user_id']
        
        mask = (
            (df_sorted['CSN'] == csn) &
            (df_sorted['admission_number_this_csn'] == admission) &
            (df_sorted['access_date'] <= current_date)
        )
        admission_data = df_sorted[mask].sort_values('access_date', ascending=False)
        
        streak = 0
        for _, prior_row in admission_data.iterrows():
            if (prior_row['nurse_user_id'] == current_nurse and
                prior_row['frontline_user_id'] == current_frontline and
                prior_row['attending_user_id'] == current_attending):
                streak += 1
            else:
                break
        
        results.append({
            'idx': idx,
            streak_col: streak
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def get_context_lookback(df_team, patient_id, csn, admission, current_date, context):
    """
    Get historical data for familiarity context.
    
    Parameters:
    -----------
    context : str
        'C1' = this admission (prior days)
        'C2' = previous admissions (same patient)
        'C3' = any patient (includes current)
    
    Returns:
    --------
    DataFrame : Historical patient-days for the context
    """
    if context == 'C1':
        # This patient, this admission, prior days
        mask = (
            (df_team['PAT_ID'] == patient_id) &
            (df_team['CSN'] == csn) &
            (df_team['admission_number_this_csn'] == admission) &
            (df_team['access_date'] < current_date)
        )
    elif context == 'C2':
        # This patient, previous admissions (any date)
        mask = (
            (df_team['PAT_ID'] == patient_id) &
            ((df_team['CSN'] != csn) | 
             (df_team['admission_number_this_csn'] < admission))
        )
    elif context == 'C3':
        # Any patient (exclude current patient-day, include rest of current admission)
        mask = (
            ((df_team['PAT_ID'] != patient_id) |
             (df_team['CSN'] != csn) |
             (df_team['admission_number_this_csn'] != admission) |
             (df_team['access_date'] != current_date))
        )
    else:
        raise ValueError(f"Unknown context: {context}")
    
    return df_team[mask]


def calculate_dyadic_familiarity(df, df_team, pair_roles, context, output_type):
    """
    Calculate dyadic familiarity.
    
    Parameters:
    -----------
    pair_roles : tuple
        e.g., ('nurse', 'frontline')
    context : str
        'C1', 'C2', or 'C3'
    output_type : str
        'binary', 'count', or 'rate'
    """
    role1, role2 = pair_roles
    role1_col = f'{role1}_user_id'
    role2_col = f'{role2}_user_id'
    
    pair_name = f"{role1[0].upper()}{role2[0].upper()}"
    fam_col = f'F_{pair_name}_{context}_{output_type}'
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc=f"Computing {fam_col}"):
        patient_id = row['PAT_ID']
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_provider1 = row[role1_col]
        current_provider2 = row[role2_col]
        
        # Get context lookback
        lookback = get_context_lookback(df_team, patient_id, csn, admission, 
                                       current_date, context)
        
        # Find matches where both providers worked together
        matches = lookback[
            (lookback[role1_col] == current_provider1) &
            (lookback[role2_col] == current_provider2)
        ]
        
        if output_type == 'binary':
            value = 1 if len(matches) > 0 else 0
        elif output_type == 'count':
            value = len(matches)
        elif output_type == 'rate':
            if len(lookback) == 0:
                value = 0.0
            else:
                value = len(matches) / len(lookback)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        results.append({
            'idx': idx,
            fam_col: value
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


def calculate_team_familiarity(df, df_team, context, output_type):
    """Calculate complete team familiarity."""
    fam_col = f'F_team_{context}_{output_type}'
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc=f"Computing {fam_col}"):
        patient_id = row['PAT_ID']
        csn = row['CSN']
        admission = row['admission_number_this_csn']
        current_date = row['access_date']
        current_nurse = row['nurse_user_id']
        current_frontline = row['frontline_user_id']
        current_attending = row['attending_user_id']
        
        lookback = get_context_lookback(df_team, patient_id, csn, admission,
                                       current_date, context)
        
        # Find matches where entire team worked together
        matches = lookback[
            (lookback['nurse_user_id'] == current_nurse) &
            (lookback['frontline_user_id'] == current_frontline) &
            (lookback['attending_user_id'] == current_attending)
        ]
        
        if output_type == 'binary':
            value = 1 if len(matches) > 0 else 0
        elif output_type == 'count':
            value = len(matches)
        elif output_type == 'rate':
            if len(lookback) == 0:
                value = 0.0
            else:
                value = len(matches) / len(lookback)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        results.append({
            'idx': idx,
            fam_col: value
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    return results_df


print("✓ All helper functions defined successfully!")


# ## Load Data

# In[3]:


# ============================================================================
# DATA LOADING
# ============================================================================
print("\nLoading data from BigQuery...")
# Primary analysis table - 7+ day stays with comprehensive metrics
query_primary = """
SELECT *
FROM `som-nero-phi-dtawfik-dt.liemn.picu_analysis_comprehensive`
"""
df_primary = client.query(query_primary).to_dataframe()
# Team identification table for familiarity calculations
query_team = """
SELECT *
FROM `som-nero-phi-dtawfik-dt.liemn.all_picu_team_identification`
"""
df_team = client.query(query_team).to_dataframe()
# All audit logs for additional provider counts
query_audit = """
SELECT DISTINCT
  csn as CSN,
  DATE(access_time) as access_date,
  admission_number_this_csn,
  user_id,
  PAT_ID
FROM `som-nero-phi-dtawfik-dt.liemn.all_picu_audit_logs`
"""
df_audit = client.query(query_audit).to_dataframe()
print(f"Loaded {len(df_primary):,} patient-days from primary table")
print(f"Loaded {len(df_team):,} team assignments")
print(f"Loaded {len(df_audit):,} audit log entries")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\nPreprocessing data...")
# Standardize column names to uppercase CSN
for df in [df_primary, df_team, df_audit]:
    if 'csn' in df.columns:
        df.rename(columns={'csn': 'CSN'}, inplace=True)
# Convert date columns
df_primary['access_date'] = pd.to_datetime(df_primary['access_date'])
df_team['access_date'] = pd.to_datetime(df_team['access_date'])
df_audit['access_date'] = pd.to_datetime(df_audit['access_date'])
# Add day of week
df_primary['day_of_week'] = df_primary['access_date'].dt.day_name()
df_team['day_of_week'] = df_team['access_date'].dt.day_name()
# Calculate days since admission for each patient-day
df_primary['days_since_admission'] = df_primary.groupby(['CSN', 'admission_number_this_csn'])['access_date'].transform(
    lambda x: (x - x.min()).dt.days
)

# Define roles with frontline subcategories (3-person team: nurse, frontline, attending)
roles = ['nurse', 'frontline_resident', 'frontline_app_or_hospitalist', 'attending']

# Classify frontline providers into subcategories
df_primary['frontline_subcategory'] = None
frontline_mask = df_primary['frontline_user_id'].notna()
df_primary.loc[frontline_mask & (df_primary['frontline_is_resident'] == True), 'frontline_subcategory'] = 'frontline_resident'
df_primary.loc[frontline_mask & ((df_primary['frontline_is_app'] == True) | (df_primary['frontline_is_hospitalist'] == True)), 'frontline_subcategory'] = 'frontline_app_or_hospitalist'

# Diagnostic output
print(f"\nFrontline provider breakdown:")
print(df_primary[frontline_mask]['frontline_subcategory'].value_counts())
print(f"Unclassified frontline providers: {df_primary[frontline_mask & df_primary['frontline_subcategory'].isna()].shape[0]}")


# ## Build and Visualize Cohort

# In[4]:


# ============================================
# 7-DAY READMISSION GAP LOGIC & COHORT BUILD
# (robust to re-runs; no extra queries)
# ============================================

print("\n" + "="*70)
print("7-DAY READMISSION GAP LOGIC & COHORT SELECTION (idempotent)")
print("="*70)

# ---- Safe defaults for params if not already defined
try:
    MIN_LOS_DAYS
except NameError:
    MIN_LOS_DAYS = 7
try:
    MAX_DAYS_ANALYZED
except NameError:
    MAX_DAYS_ANALYZED = 56  # 0-based days_since_admission => keep rows with < 56

# ---- Ensure admission key exists (recompute every run)
df_primary['picu_admission_id'] = (
    df_primary['CSN'].astype(str) + '_' + df_primary['admission_number_this_csn'].astype(str)
)

# ---- Build fresh per-admission table from df_primary (idempotent)
admissions_agg = (
    df_primary
    .groupby(['PAT_ID', 'CSN', 'admission_number_this_csn'], as_index=False)
    .agg(
        admission_date=('access_date', 'min'),
        discharge_date=('access_date', 'max')
    )
).sort_values(['PAT_ID', 'admission_date', 'admission_number_this_csn'])
admissions_agg['los_days'] = (admissions_agg['discharge_date'] - admissions_agg['admission_date']).dt.days + 1

# ---- Compute 7-day gap flags (recompute every run)
admissions_agg['prev_picu_discharge_date'] = admissions_agg.groupby('PAT_ID')['discharge_date'].shift(1)
admissions_agg['days_since_prev_picu_discharge'] = (
    admissions_agg['admission_date'] - admissions_agg['prev_picu_discharge_date']
).dt.days
admissions_agg['exclude_7day_gap'] = admissions_agg['days_since_prev_picu_discharge'] < 7

# ---- Merge flags/LOS back to df_primary (overwrite existing cleanly; robust to repeats)
admissions_agg['picu_admission_id'] = (
    admissions_agg['CSN'].astype(str) + '_' + admissions_agg['admission_number_this_csn'].astype(str)
)

# Drop possibly stale columns before merge; safe on re-run
df_primary = df_primary.drop(columns=['los_days', 'exclude_7day_gap'], errors='ignore')

df_primary = df_primary.merge(
    admissions_agg[['picu_admission_id', 'los_days', 'exclude_7day_gap']],
    on='picu_admission_id',
    how='left',
    validate='m:1'
)

# ---- days_since_admission (keep existing; otherwise compute 0-based)
if 'days_since_admission' not in df_primary.columns:
    df_primary['days_since_admission'] = df_primary.groupby(
        ['CSN', 'admission_number_this_csn']
    )['access_date'].transform(lambda x: (x - x.min()).dt.days)

# ---- NEW: Create admission/discharge time exclusion flags
# Convert TIME columns to comparable format (convert to timedelta for comparison)
def time_to_comparable(time_col):
    """Convert TIME column to hours for comparison"""
    if time_col.dtype == 'object':
        # If stored as string like "08:00:00"
        return pd.to_timedelta(time_col).dt.total_seconds() / 3600
    elif pd.api.types.is_timedelta64_dtype(time_col):
        return time_col.dt.total_seconds() / 3600
    else:
        # Already numeric or other format
        return time_col

# Initialize exclusion flags
df_primary['invalid_admission_time'] = False
df_primary['invalid_discharge_time'] = False

# Check if time columns exist and apply filters
if 'admission_time' in df_primary.columns and 'is_admission_day' in df_primary.columns:
    # Exclude admission days where admitted at or after 8:00 AM
    admission_mask = (
        df_primary['is_admission_day'] & 
        df_primary['admission_time'].notna()
    )
    # Convert admission_time to datetime.time for comparison
    df_primary.loc[admission_mask, 'invalid_admission_time'] = (
        pd.to_datetime(df_primary.loc[admission_mask, 'admission_time'].astype(str)).dt.time >= 
        pd.to_datetime('08:00:00').time()
    )

if 'discharge_time' in df_primary.columns and 'is_discharge_day' in df_primary.columns:
    # Exclude discharge days where discharged before 11:00 AM
    discharge_mask = (
        df_primary['is_discharge_day'] & 
        df_primary['discharge_time'].notna()
    )
    df_primary.loc[discharge_mask, 'invalid_discharge_time'] = (
        pd.to_datetime(df_primary.loc[discharge_mask, 'discharge_time'].astype(str)).dt.time < 
        pd.to_datetime('11:00:00').time()
    )

# ---- Report on invalid admission/discharge times before filtering
invalid_admission_count = df_primary['invalid_admission_time'].sum()
invalid_discharge_count = df_primary['invalid_discharge_time'].sum()
invalid_admission_pct = (df_primary['invalid_admission_time'].mean() * 100) if len(df_primary) > 0 else 0.0
invalid_discharge_pct = (df_primary['invalid_discharge_time'].mean() * 100) if len(df_primary) > 0 else 0.0

print(f"\n⏰ ADMISSION/DISCHARGE TIME VALIDATION (pre-filter):")
print(f"   Invalid admission days (admitted ≥ 8:00 AM):")
print(f"      - Count: {invalid_admission_count:,}")
print(f"      - Percent: {invalid_admission_pct:.2f}%")
print(f"   Invalid discharge days (discharged < 11:00 AM):")
print(f"      - Count: {invalid_discharge_count:,}")
print(f"      - Percent: {invalid_discharge_pct:.2f}%")

# ---- Identify rows missing ALL 4 team roles
team_role_cols = ['nurse_user_id', 'frontline_user_id', 'attending_user_id']
# Check which columns exist
existing_team_cols = [col for col in team_role_cols if col in df_primary.columns]

if existing_team_cols:
    # Create flag for rows missing all existing team roles
    df_primary['all_roles_missing'] = df_primary[existing_team_cols].isna().all(axis=1)
else:
    # If no team role columns exist, don't exclude any rows
    df_primary['all_roles_missing'] = False

# ---- Report on ALL roles missing before filtering
all_roles_missing_count = df_primary['all_roles_missing'].sum()
all_roles_missing_pct = (df_primary['all_roles_missing'].mean() * 100) if len(df_primary) > 0 else 0.0

print(f"\n⚠️  PATIENT-DAYS WITH ALL TEAM ROLES MISSING (pre-filter):")
print(f"   - Count: {all_roles_missing_count:,}")
print(f"   - Percent: {all_roles_missing_pct:.2f}%")

# ---- Build analysis cohort (now excludes rows with all roles missing AND invalid times)
mask_eligible = (
    (~df_primary['exclude_7day_gap'].fillna(False)) &
    (df_primary['los_days'].fillna(0) >= MIN_LOS_DAYS) &
    (df_primary['days_since_admission'] < MAX_DAYS_ANALYZED) &
    (~df_primary['all_roles_missing']) &
    (~df_primary['invalid_admission_time']) &  # NEW: exclude invalid admission times
    (~df_primary['invalid_discharge_time'])     # NEW: exclude invalid discharge times
)

# Count exclusions by reason for reporting
excluded_7day = ((~df_primary['exclude_7day_gap'].fillna(False)) == False).sum()
excluded_los = ((df_primary['los_days'].fillna(0) >= MIN_LOS_DAYS) == False).sum()
excluded_days = ((df_primary['days_since_admission'] < MAX_DAYS_ANALYZED) == False).sum()
excluded_all_roles = df_primary['all_roles_missing'].sum()
excluded_admission_time = df_primary['invalid_admission_time'].sum()  # NEW
excluded_discharge_time = df_primary['invalid_discharge_time'].sum()  # NEW

df_analysis = df_primary.loc[mask_eligible].copy()

cohort_description = "All PICU admissions with 7-day gap filter, time validation, and team role presence"
print(f"\nCOHORT SELECTED: {cohort_description}")

# ---- Exclusion Summary
print("\nEXCLUSION SUMMARY (from df_primary):")
print(f"  - Excluded due to 7-day readmission gap: {excluded_7day:,} patient-days")
print(f"  - Excluded due to LOS < {MIN_LOS_DAYS} days: {excluded_los:,} patient-days")
print(f"  - Excluded due to days_since_admission ≥ {MAX_DAYS_ANALYZED}: {excluded_days:,} patient-days")
print(f"  - Excluded due to ALL roles missing: {excluded_all_roles:,} patient-days")
print(f"  - Excluded due to invalid admission time (≥ 8:00 AM): {excluded_admission_time:,} patient-days")
print(f"  - Excluded due to invalid discharge time (< 11:00 AM): {excluded_discharge_time:,} patient-days")
print(f"  - Total rows after all filters: {len(df_analysis):,} patient-days")

# ---- Cohort summary (idempotent)
n_picu_admissions = df_analysis.groupby(['CSN', 'admission_number_this_csn']).ngroups
n_csns = df_analysis['CSN'].nunique()
n_patients = df_analysis['PAT_ID'].nunique()
n_days = len(df_analysis)

print("\nAnalysis cohort summary:")
print(f"- Unique PICU admissions: {n_picu_admissions}")
print(f"- Unique hospital encounters (CSNs): {n_csns}")
print(f"- Unique patients: {n_patients}")
print(f"- Patient-days: {n_days}")

# Distribution of admission_number_this_csn
admission_dist = (
    df_analysis.groupby(['CSN', 'admission_number_this_csn'])
    .size()
    .reset_index(name='n_days')
)
admission_counts = admission_dist['admission_number_this_csn'].value_counts().sort_index()
print("\nDistribution of PICU admissions within CSNs:")
for num, count in admission_counts.items():
    suffix = 'st' if num == 1 else ('nd' if num == 2 else ('rd' if num == 3 else 'th'))
    print(f"  {num}{suffix} PICU admission: {count}")


# ---- Missing team members summary (now on filtered data)
print("\nMissing Team Members Summary (counts and % of df_analysis rows after filtering):")
_roles_cols = {
    'Nurse': 'nurse_user_id',
    'Frontline': 'frontline_user_id',
    'Attending': 'attending_user_id',
}
_missing_rows = []
for role_name, col in _roles_cols.items():
    if col in df_analysis.columns:
        miss_ct = df_analysis[col].isna().sum()
        miss_pct = (df_analysis[col].isna().mean() * 100) if len(df_analysis) else 0.0
    else:
        miss_ct = 0
        miss_pct = 0.0
    _missing_rows.append((role_name, int(miss_ct), round(miss_pct, 2)))

missing_df = pd.DataFrame(_missing_rows, columns=['Role', 'Missing_Count', 'Missing_Percent'])
print(missing_df.to_string(index=False))

# ---- Verify no rows with all roles missing remain
if existing_team_cols:
    remaining_all_missing = df_analysis[existing_team_cols].isna().all(axis=1).sum()
    print(f"\n✓ Verification: Patient-days with ALL roles missing in final cohort: {remaining_all_missing}")
    if remaining_all_missing > 0:
        print("  ⚠️ WARNING: Some rows with all roles missing still present!")

# ---- Verify no invalid admission/discharge times remain
remaining_invalid_admission = df_analysis['invalid_admission_time'].sum() if 'invalid_admission_time' in df_analysis.columns else 0
remaining_invalid_discharge = df_analysis['invalid_discharge_time'].sum() if 'invalid_discharge_time' in df_analysis.columns else 0
print(f"\n✓ Verification: Patient-days with invalid admission time in final cohort: {remaining_invalid_admission}")
print(f"✓ Verification: Patient-days with invalid discharge time in final cohort: {remaining_invalid_discharge}")
if remaining_invalid_admission > 0 or remaining_invalid_discharge > 0:
    print("  ⚠️ WARNING: Some rows with invalid times still present!")

# ---- Ensure day-of-week/day_name exist on df_analysis (idempotent)
df_analysis['day_of_week'] = df_analysis['access_date'].dt.dayofweek
df_analysis['day_name'] = df_analysis['access_date'].dt.day_name()

# ---- patient_stay_id construction (idempotent; requires admission_sequence col)
if {'PAT_ID', 'admission_sequence', 'admission_number_this_csn'}.issubset(df_analysis.columns):
    df_analysis['patient_stay_id'] = (
        df_analysis['PAT_ID'].astype(str) + '_' +
        df_analysis['admission_sequence'].astype(str) + '_' +
        df_analysis['admission_number_this_csn'].astype(str)
    )

print("\nCOHORT SUMMARY NOTES:")
print("- Applies 7-day readmission gap across all PICU admissions per patient")
print(f"- Only admissions with LOS ≥ {MIN_LOS_DAYS} days")
print(f"- Only first {MAX_DAYS_ANALYZED} days per admission (0-based indexing in code)")
print("- Excludes patient-days where ALL team roles (nurse, frontline, attending) are missing")
print("- Excludes admission days where patient admitted at or after 8:00 AM")
print("- Excludes discharge days where patient discharged before 11:00 AM")


# In[5]:


# ============================================================
# ENHANCED COHORT DESCRIPTIVES & VISUALIZATION (adapted)
# (assumes `admissions_agg`, `df_primary`, `df_analysis`,
#  MIN_LOS_DAYS=7, MAX_DAYS_ANALYZED=56 are already defined)
# ============================================================

print("=== PATIENT COHORT CHARACTERISTICS ===\n")

# Use per-admission table derived from df_primary
all_admissions_data = admissions_agg.copy()
all_admissions_data = all_admissions_data.assign(
    picu_admission_id=all_admissions_data['CSN'].astype(str) + '_' +
                      all_admissions_data['admission_number_this_csn'].astype(str),
    los_days_calc=all_admissions_data['los_days']  # align naming with prior code
)

print("LOS Calculation Verification:")
print(f"PICU admissions with calculated LOS = 7: {(all_admissions_data['los_days_calc'] == 7).sum()}")
print(f"PICU admissions with calculated LOS >= 7: {(all_admissions_data['los_days_calc'] >= 7).sum()}")

# Calculate patient-days by category
print("\nCalculating patient-day exclusions...")

# 1) ALL patient-days in PICU (all admissions)
all_patient_days = all_admissions_data['los_days_calc'].sum()

# 2) 7-day gap exclusions
excluded_by_gap = all_admissions_data['exclude_7day_gap'].sum()
excluded_gap_days = all_admissions_data.loc[all_admissions_data['exclude_7day_gap'], 'los_days_calc'].sum()

# 3) Patient-days from admissions with LOS < 7
short_stay_days = all_admissions_data.loc[all_admissions_data['los_days_calc'] < MIN_LOS_DAYS, 'los_days_calc'].sum()

# 4) Patient-days after day 56 from eligible admissions (not excluded by gap, LOS >= 7)
eligible_admissions = all_admissions_data.loc[
    (~all_admissions_data['exclude_7day_gap']) &
    (all_admissions_data['los_days_calc'] >= MIN_LOS_DAYS)
].copy()

days_after_56 = int((eligible_admissions['los_days_calc'] - MAX_DAYS_ANALYZED).clip(lower=0).sum())

# 5) Analyzed patient-days (days 0..55 => 56 days max per admission)
analyzed_days = len(df_analysis)

# Verify decomposition
expected_analyzed = int(eligible_admissions['los_days_calc'].clip(upper=MAX_DAYS_ANALYZED).sum())

print(f"\nVerification:")
print(f"Expected analyzed days: {expected_analyzed}")
print(f"Actual analyzed days: {analyzed_days}")
print(f"Difference: {expected_analyzed - analyzed_days}")

# Exclusion summary table
exclusion_summary = pd.DataFrame({
    'Category': [
        'Total PICU patient-days (all admissions)',
        'Excluded: Within 7 days of previous PICU discharge',
        'Excluded: PICU admissions with LOS < 7 days',
        'Excluded: Days after day 56 (from eligible admissions)',
        'ANALYZED: Days 1-56 of eligible PICU admissions'
    ],
    'Patient_Days': [
        all_patient_days,
        excluded_gap_days,
        max(0, short_stay_days - excluded_gap_days),  # avoid double count
        days_after_56,
        analyzed_days
    ]
})
exclusion_summary['Percent_of_Total'] = (exclusion_summary['Patient_Days'] / all_patient_days * 100).round(2)

print("\nPatient-Day Exclusion Summary:")
display(exclusion_summary)

# ===========================
# PICU ADMISSION ANALYSIS
# ===========================
print("\n=== PICU ADMISSION ANALYSIS ===")

# Unique PICU admissions in analysis cohort
n_picu_admissions_analyzed = df_analysis.groupby(['CSN', 'admission_number_this_csn']).ngroups
n_csns_analyzed = df_analysis['CSN'].nunique()
n_patients_analyzed = df_analysis['PAT_ID'].nunique()

print(f"Analysis cohort contains:")
print(f"- {n_picu_admissions_analyzed} PICU admissions")
print(f"- {n_csns_analyzed} unique hospital encounters (CSNs)")
print(f"- {n_patients_analyzed} unique patients")

# Distribution of PICU admissions within CSNs
picu_admission_dist = df_analysis.groupby(['CSN', 'admission_number_this_csn']).size().reset_index(name='n_days')
admission_number_counts = picu_admission_dist['admission_number_this_csn'].value_counts().sort_index()

print(f"\nPICU admission distribution within hospital stays:")
for num, count in admission_number_counts.items():
    pct = count / n_picu_admissions_analyzed * 100 if n_picu_admissions_analyzed > 0 else 0
    suffix = 'st' if num == 1 else ('nd' if num == 2 else ('rd' if num == 3 else 'th'))
    print(f"  {num}{suffix} PICU admission: {count} ({pct:.1f}%)")

# ===========================
# VISUALIZATIONS
# ===========================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1) LOS distribution for ALL PICU admissions
ax = axes[0, 0]
los_counts_all = all_admissions_data['los_days_calc'].value_counts().sort_index()
ax.bar(los_counts_all.index, los_counts_all.values, alpha=0.7, color='skyblue', edgecolor='black', width=0.8)
ax.axvline(MIN_LOS_DAYS, color='red', linestyle='--', label=f'{MIN_LOS_DAYS} days cutoff', linewidth=2)
ax.axvline(MAX_DAYS_ANALYZED, color='orange', linestyle='--', label=f'{MAX_DAYS_ANALYZED} days analysis limit')
ax.set_xlabel('Length of Stay (days)')
ax.set_ylabel('Number of PICU Admissions')
ax.set_title(f'All PICU Admissions\n(n={len(all_admissions_data)} admissions)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, min(100, int(los_counts_all.index.max()) if len(los_counts_all) else 100))

# 2) Gap distribution between PICU admissions
ax = axes[0, 1]
gaps = all_admissions_data.loc[all_admissions_data['days_since_prev_picu_discharge'].notna(), 'days_since_prev_picu_discharge']
gaps_clipped = gaps.clip(upper=60)
ax.hist(gaps_clipped, bins=75, alpha=0.7, color='lightcoral', edgecolor='black')
ax.axvline(MIN_LOS_DAYS, color='red', linestyle='--', linewidth=2, label='7-day cutoff')
ax.set_xlabel('Days Since Previous PICU Discharge')
ax.set_ylabel('Number of PICU Admissions')
ax.set_title('Gap Distribution Between PICU Admissions')
ax.legend()
ax.grid(True, alpha=0.3)
excluded_text = f'Excluded: {(gaps < 7).sum()}/{len(gaps)} ({(gaps < 7).mean()*100:.1f}%)'
ax.text(0.5, 0.95, excluded_text, transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 3) Pie chart of exclusion categories (patient-days)
ax = axes[0, 2]
sizes = [
    analyzed_days,
    excluded_gap_days,
    max(0, short_stay_days - excluded_gap_days),
    days_after_56
]
labels = ['Analyzed\n(Days 1-56)', '7-Day Gap\nExclusions', 'LOS < 7', 'Days > 56']
colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
if non_zero:
    nz_sizes, nz_labels, nz_colors = zip(*non_zero)
    explode = [0.1 if l.startswith('Analyzed') else 0 for l in nz_labels]
    ax.pie(nz_sizes, explode=explode, labels=nz_labels, colors=nz_colors, autopct='%1.1f%%', startangle=90)
else:
    ax.text(0.5, 0.5, "No data", ha='center', va='center')
ax.set_title('Patient-Day Distribution\n(All PICU Patient-Days)')

# 4) Days analyzed per PICU admission (0-based days_since_admission => +1)
ax = axes[1, 0]
days_per_admission = (
    df_analysis.groupby(['CSN', 'admission_number_this_csn'])['days_since_admission'].max().astype('Int64') + 1
)
day_counts = days_per_admission.value_counts().sort_index()
ax.bar(day_counts.index.astype(int), day_counts.values, alpha=0.7, color='salmon', width=0.8)
ax.set_xlabel('Days Analyzed per PICU Admission')
ax.set_ylabel('Number of PICU Admissions')
ax.set_title(f'Days Analyzed per PICU Admission\n(All eligible, capped at {MAX_DAYS_ANALYZED})')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, MAX_DAYS_ANALYZED + 1)
# Override the x-tick label for MAX_DAYS_ANALYZED to show ≥MAX
current_ticks = ax.get_xticks()
current_labels = [str(int(t)) if int(t) != MAX_DAYS_ANALYZED else f'≥{MAX_DAYS_ANALYZED}' for t in current_ticks]
ax.set_xticks(current_ticks)
ax.set_xticklabels(current_labels)

# 5) PICU admissions per patient distribution (in analysis cohort)
ax = axes[1, 1]
admissions_per_patient = df_analysis.groupby('PAT_ID').apply(
    lambda x: x.groupby(['CSN', 'admission_number_this_csn']).ngroups
)
admission_dist = admissions_per_patient.value_counts().sort_index()
ax.bar(admission_dist.index, admission_dist.values, color='steelblue', alpha=0.7)
ax.set_xlabel('Number of PICU Admissions per Patient')
ax.set_ylabel('Number of Patients')
ax.set_title('PICU Admissions per Patient\n(In Analysis Cohort)')
ax.grid(True, alpha=0.3)

# 6) Impact of filters waterfall (admission counts)
ax = axes[1, 2]
admissions_to_exclude = set(all_admissions_data.loc[all_admissions_data['exclude_7day_gap'], 'picu_admission_id'])
all_picu = len(all_admissions_data)
after_gap = len(all_admissions_data.loc[~all_admissions_data['exclude_7day_gap']])
after_los = len(all_admissions_data.loc[(~all_admissions_data['exclude_7day_gap']) &
                                        (all_admissions_data['los_days_calc'] >= MIN_LOS_DAYS)])
final = n_picu_admissions_analyzed

categories = ['All PICU\nAdmissions', '7-Day Gap\nFilter', 'LOS ≥ 7\nFilter', 'Final\nCohort']
values = [all_picu, after_gap - all_picu, after_los - after_gap, final]
colors_wf = ['lightblue', 'lightcoral', 'lightyellow', 'darkgreen']

# Calculate positions for waterfall
positions = [0]
cumulative = all_picu
for i in range(1, len(values)-1):
    positions.append(cumulative)
    cumulative += values[i]
positions.append(0)  # Final bar starts at 0

# Draw bars
for i, (cat, val, color, pos) in enumerate(zip(categories, values, colors_wf, positions)):
    if i == 0:  # First bar
        ax.bar(i, val, bottom=0, color=color, alpha=0.8)
        ax.text(i, val/2, f'{val:,}', ha='center', va='center', fontweight='bold')
    elif i == len(categories) - 1:  # Final bar
        ax.bar(i, val, bottom=0, color=color, alpha=0.8)
        ax.text(i, val/2, f'{val:,}', ha='center', va='center', fontweight='bold')
    else:  # Middle delta bars
        height = abs(val)
        bottom = pos + val if val < 0 else pos
        ax.bar(i, height, bottom=bottom, color=color, alpha=0.8)
        ax.text(i, bottom + height/2, f'{val:,}', ha='center', va='center', fontweight='bold')

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories)
ax.set_ylabel('Number of PICU Admissions')
ax.set_title('Filter Impact on Cohort Size')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# ===========================
# SUMMARY STAT TABLE
# ===========================
summary_stats = pd.DataFrame({
    'Cohort': ['All PICU Admissions', 'After 7-Day Gap Filter', 'Analysis Cohort (Gap + LOS ≥ 7)'],
    'N_PICU_Admissions': [
        len(all_admissions_data),
        after_gap,
        n_picu_admissions_analyzed
    ],
    'N_CSNs': [
        all_admissions_data['CSN'].nunique(),
        all_admissions_data.loc[~all_admissions_data['exclude_7day_gap'], 'CSN'].nunique(),
        n_csns_analyzed
    ],
    'N_Unique_Patients': [
        all_admissions_data['PAT_ID'].nunique(),
        all_admissions_data.loc[~all_admissions_data['exclude_7day_gap'], 'PAT_ID'].nunique(),
        n_patients_analyzed
    ],
    'Mean_LOS': [
        f"{all_admissions_data['los_days_calc'].mean():.2f} days",
        f"{all_admissions_data.loc[~all_admissions_data['exclude_7day_gap'], 'los_days_calc'].mean():.2f} days",
        f"{(days_per_admission.mean() if len(days_per_admission) else 0):.2f} days analyzed"
    ],
    'Total_Patient_Days': [
        all_patient_days,
        all_admissions_data.loc[~all_admissions_data['exclude_7day_gap'], 'los_days_calc'].sum(),
        analyzed_days
    ]
})

print("COHORT SUMMARY STATS")
print("*** This analysis includes ALL PICU admissions (not just first within CSN) ***")
print("*** 7-day gap filter ensures independent PICU episodes ***")
display(summary_stats.round(2))

print("KEY POINTS ABOUT THE DATASET")
print(f"- {n_picu_admissions_analyzed} PICU admissions analyzed")
print(f"- From {n_csns_analyzed} hospital encounters and {n_patients_analyzed} unique patients")
print(f"- Excludes PICU admissions within 7 days of previous PICU discharge")
print(f"- Includes multiple PICU admissions per hospital stay if 7+ days apart")
print(f"- Analyzes days 1-{MAX_DAYS_ANALYZED} of each PICU admission")

# ===========================
# FILTER IMPACT BY ADMISSION #
# ===========================
print("FILTER IMPACT ANALYSIS")
gap_impact = []
for num in sorted(all_admissions_data['admission_number_this_csn'].unique()):
    admissions_at_num = all_admissions_data.loc[all_admissions_data['admission_number_this_csn'] == num]
    if len(admissions_at_num) > 0:
        excluded = admissions_at_num['exclude_7day_gap'].sum()
        total = len(admissions_at_num)
        gap_impact.append({
            'PICU_Admission_Number': num,
            'Total': total,
            'Excluded_by_Gap': int(excluded),
            'Percent_Excluded': (excluded/total*100 if total > 0 else 0)
        })
gap_impact = pd.DataFrame(gap_impact)
print("\n7-Day Gap Filter Impact by PICU Admission Number:")
display(gap_impact.round(2))

# CSNs with multiple PICU admissions in analysis
csns_with_multiple = df_analysis.groupby('CSN')['admission_number_this_csn'].max()
multi_picu_csns = (csns_with_multiple > 1).sum()
print(f"\nHospital encounters (CSNs) with multiple PICU admissions:")
print(f"- CSNs with 1 PICU admission: {(csns_with_multiple == 1).sum()}")
print(f"- CSNs with 2+ PICU admissions: {multi_picu_csns}")
print(f"- Maximum PICU admissions in one CSN: {csns_with_multiple.max()}")

print("\n" + "="*80)
print("FINAL DATASET FOR DOWNSTREAM ANALYSIS")
print("="*80)
print(f"DataFrame: df_analysis")
print(f"PICU Admissions: {n_picu_admissions_analyzed}")
print(f"Hospital encounters (CSNs): {n_csns_analyzed}")
print(f"Unique patients: {n_patients_analyzed}")
print(f"Patient-days: {len(df_analysis)} (days 1-{MAX_DAYS_ANALYZED} for each PICU admission)")
print("\nThis dataset includes:")
print("- ALL PICU admissions with LOS ≥ 7 days")
print("- Excludes admissions within 7 days of previous PICU discharge")
print("- Includes multiple PICU admissions per hospital stay (if eligible)")
print(f"- Only days 1-{MAX_DAYS_ANALYZED} of each PICU admission")
print("="*80)



# In[6]:


# ============================================================
# TOTAL LENGTH OF STAY VISUALIZATION FOR ANALYSIS COHORT
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

# Get eligible admissions (those in the analysis cohort)
eligible_admissions = all_admissions_data.loc[
    (~all_admissions_data['exclude_7day_gap']) &
    (all_admissions_data['los_days_calc'] >= MIN_LOS_DAYS)
].copy()

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ===========================
# 1. HISTOGRAM OF LOS DISTRIBUTION (CAPPED AT 56)
# ===========================
ax = axes[0]
los_data = eligible_admissions['los_days_calc']

# Create data for histogram with values capped for display
los_display = los_data.copy()
los_display[los_display > MAX_DAYS_ANALYZED] = MAX_DAYS_ANALYZED

# Create bins from 7 to 56
bins = list(range(MIN_LOS_DAYS, MAX_DAYS_ANALYZED + 2))  # 7 to 57 to include 56

# Create histogram
n, bins_out, patches = ax.hist(los_display, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')

# Color the last bar differently (>56 days)
if len(patches) > 0:
    patches[-1].set_facecolor('#d62728')
    patches[-1].set_alpha(0.8)

# Add statistics lines (using original uncapped data)
ax.axvline(los_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {los_data.mean():.1f} days')
ax.axvline(los_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {los_data.median():.0f} days')

# Formatting
ax.set_xlabel('Length of Stay (days)', fontsize=12)
ax.set_ylabel('Number of PICU Admissions', fontsize=12)
ax.set_title(f'LOS Distribution - Analysis Cohort\n(n={len(eligible_admissions)} eligible PICU admissions)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Modify x-axis labels to show ">56" for the last bin
current_ticks = ax.get_xticks()
new_labels = []
for tick in current_ticks:
    if tick == MAX_DAYS_ANALYZED:
        new_labels.append(f'≥{MAX_DAYS_ANALYZED}')
    else:
        new_labels.append(str(int(tick)) if tick.is_integer() else str(tick))
ax.set_xticklabels(new_labels, rotation=45, ha='right')

# Add text box with statistics
admissions_exceeding_56 = (los_data > MAX_DAYS_ANALYZED).sum()
pct_exceeding = (admissions_exceeding_56 / len(los_data) * 100)

stats_text = f'Total Admissions: {len(eligible_admissions)}\n' \
             f'Mean LOS: {los_data.mean():.1f} days\n' \
             f'Median LOS: {los_data.median():.0f} days\n' \
             f'Admissions ≥{MAX_DAYS_ANALYZED} days: {admissions_exceeding_56} ({pct_exceeding:.1f}%)'
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top', horizontalalignment='right', fontsize=10)

# ===========================
# 2. LOS CATEGORIES BREAKDOWN
# ===========================
ax = axes[1]

# Define LOS categories
los_categories = pd.cut(los_data, 
                        bins=[0, 14, 28, 56, np.inf],
                        labels=['7-14 days', '15-28 days', '29-56 days', '>56 days'],
                        include_lowest=False)

# Count by category
category_counts = los_categories.value_counts()
category_pct = (category_counts / len(los_data) * 100).round(1)

# Create bar plot
colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
bars = ax.bar(range(len(category_counts)), category_counts.values, 
              color=colors, alpha=0.7, edgecolor='black', width=0.6)

# Add value labels on bars
for i, (count, pct) in enumerate(zip(category_counts.values, category_pct.values)):
    ax.text(i, count + 0.5, f'{count}\n({pct}%)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Formatting
ax.set_xticks(range(len(category_counts)))
ax.set_xticklabels(category_counts.index, rotation=0)
ax.set_xlabel('Length of Stay Category', fontsize=12)
ax.set_ylabel('Number of PICU Admissions', fontsize=12)
ax.set_title('LOS Distribution by Category', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Add note about analysis window
ax.text(0.5, -0.12, f'Note: Analysis limited to days 1-{MAX_DAYS_ANALYZED} per admission', 
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

# Overall title
fig.suptitle('TOTAL LENGTH OF STAY ANALYSIS - ELIGIBLE PICU ADMISSIONS', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.show()

# ===========================
# PRINT DETAILED STATISTICS
# ===========================
print("\n" + "="*60)
print("TOTAL LENGTH OF STAY STATISTICS - ANALYSIS COHORT")
print("="*60)
print(f"Number of eligible PICU admissions: {len(eligible_admissions)}")
print(f"Total patient-days (uncapped): {los_data.sum():,}")
print(f"Total patient-days analyzed (capped at {MAX_DAYS_ANALYZED}): {analyzed_days:,}")

print("\nDescriptive Statistics:")
print(f"  Mean LOS: {los_data.mean():.2f} days")
print(f"  Median LOS: {los_data.median():.0f} days")
print(f"  Standard Deviation: {los_data.std():.2f} days")
print(f"  Min LOS: {los_data.min():.0f} days")
print(f"  Max LOS: {los_data.max():.0f} days")

print("\nLOS Category Distribution:")
for category in category_counts.index:
    count = category_counts[category]
    pct = category_pct[category]
    print(f"  {category}: {count} admissions ({pct}%)")

print(f"\nAdmissions exceeding {MAX_DAYS_ANALYZED}-day analysis window:")
print(f"  {admissions_exceeding_56} admissions ({pct_exceeding:.1f}%)")
print(f"  Total days beyond day {MAX_DAYS_ANALYZED}: {days_after_56:,}")
# ===========================
# DESCRIPTIVE STATISTICS TABLE
# ===========================
import pandas as pd
from IPython.display import display

# Calculate statistics
q1 = los_data.quantile(0.25)
q3 = los_data.quantile(0.75)
iqr = q3 - q1

# Create main descriptive statistics table - transposed format
descriptive_stats_dict = {
    'Number of Admissions': f'{len(eligible_admissions):,}',
    'Mean LOS': f'{los_data.mean():.2f}',
    'Median LOS': f'{los_data.median():.0f}',
    'Standard Deviation': f'{los_data.std():.2f}',
    'IQR': f'{iqr:.1f}',
    'Min LOS': f'{los_data.min():.0f}',
    'Max LOS': f'{los_data.max():.0f}',
    f'Total Patient-Days': f'{los_data.sum():,}',
    f'Patient-Days Analyzed (capped at {MAX_DAYS_ANALYZED})': f'{analyzed_days:,}',
    f'Admissions ≥{MAX_DAYS_ANALYZED} days': f'{admissions_exceeding_56:,}',
    f'% Admissions ≥{MAX_DAYS_ANALYZED} days': f'{pct_exceeding:.1f}%'
}

# Create DataFrame with statistics as columns
descriptive_stats = pd.DataFrame([descriptive_stats_dict])

# Display the main statistics table
print("\n" + "="*60)
print("TOTAL LENGTH OF STAY STATISTICS - ANALYSIS COHORT")
print("="*60)
print("\nDescriptive Statistics:")
display(descriptive_stats)
print("="*60)


# ## Measure 1 - Cumulative Unique Providers
# 

# In[ ]:





# In[7]:


# ============================================================================
# MEASURE 1: CUMULATIVE UNIQUE PROVIDERS
# ============================================================================

print("\n" + "="*80)
print("MEASURE 1: CUMULATIVE UNIQUE PROVIDERS")
print("="*80)

# Checkpoint configuration
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = CHECKPOINT_DIR / "measure1_cumulative_providers.pkl"
FORCE_RECOMPUTE = False

if CHECKPOINT_FILE.exists() and not FORCE_RECOMPUTE:
    print(f"Loading cached data from {CHECKPOINT_FILE}...")
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint_data = pickle.load(f)
        df_cumulative = checkpoint_data['df_cumulative']
        df_cumulative_all = checkpoint_data.get('df_cumulative_all')
        computation_timestamp = checkpoint_data.get('timestamp', 'unknown')
    print(f"Loaded checkpoint from {computation_timestamp}")
    print(f"Cumulative data shape: {df_cumulative.shape}")
    
else:
    print("Computing cumulative unique providers...")
    
    cumulative_primary = []
    unique_admissions = df_primary.groupby(['CSN', 'admission_number_this_csn']).size()
    
    print("Processing primary team data...")
    for (csn, adm), group in tqdm(df_primary.groupby(['CSN', 'admission_number_this_csn']), 
                                   total=len(unique_admissions), 
                                   desc="Primary team"):
        group = group.sort_values('access_date')
        for idx, row in group.iterrows():
            day_num = row['days_since_admission']
            
            mask = (df_primary['CSN'] == csn) & \
                   (df_primary['admission_number_this_csn'] == adm) & \
                   (df_primary['days_since_admission'] <= day_num)
            
            subset = df_primary[mask]
            
            result = {
                'CSN': csn,
                'admission_number_this_csn': adm,
                'days_since_admission': day_num,
                'access_date': row['access_date']
            }
            
            # Original roles (keep for backward compatibility)
            for role in ['nurse', 'attending']:  # Note: removed 'frontline' here
                role_col = f"{role}_user_id"
                if role_col in subset.columns:
                    unique_providers = subset[role_col].dropna().nunique()
                    result[f"cumulative_{role}"] = unique_providers
                else:
                    result[f"cumulative_{role}"] = 0
            
            # Handle frontline with subcategories
            if 'frontline_user_id' in subset.columns:
                # All frontline
                result['cumulative_frontline'] = subset['frontline_user_id'].dropna().nunique()
                
                # Frontline residents
                resident_mask = subset['frontline_subcategory'] == 'frontline_resident'
                result['cumulative_frontline_resident'] = subset.loc[resident_mask, 'frontline_user_id'].dropna().nunique()
                
                # Frontline APP/Hospitalist
                app_hosp_mask = subset['frontline_subcategory'] == 'frontline_app_or_hospitalist'
                result['cumulative_frontline_app_or_hospitalist'] = subset.loc[app_hosp_mask, 'frontline_user_id'].dropna().nunique()
            else:
                result['cumulative_frontline'] = 0
                result['cumulative_frontline_resident'] = 0
                result['cumulative_frontline_app_or_hospitalist'] = 0
            
            # All primary team members
            all_primary = pd.concat([
                subset[f"{role}_user_id"].dropna() 
                for role in ['nurse', 'frontline', 'attending']
                if f"{role}_user_id" in subset.columns
            ])
            result['cumulative_all_primary'] = all_primary.nunique() if len(all_primary) > 0 else 0
            
            cumulative_primary.append(result)
    
    df_cumulative = pd.DataFrame(cumulative_primary)
    
    cumulative_all = []
    unique_audit_admissions = df_audit.groupby(['CSN', 'admission_number_this_csn']).size()
    
    print("Processing audit log data...")
    for (csn, adm), group in tqdm(df_audit.groupby(['CSN', 'admission_number_this_csn']), 
                                   total=len(unique_audit_admissions), 
                                   desc="Audit logs"):
        start_date = df_primary[(df_primary['CSN'] == csn) & 
                                (df_primary['admission_number_this_csn'] == adm)]['access_date'].min()
        
        if pd.isna(start_date):
            continue
        
        group = group.sort_values('access_date')
        
        for date in group['access_date'].unique():
            days_since = (date - start_date).days
            if days_since > 56:  # Match the x-axis limit
                continue
                
            mask = (group['access_date'] <= date)
            unique_all = group[mask]['user_id'].nunique()
            
            cumulative_all.append({
                'CSN': csn,
                'admission_number_this_csn': adm,
                'days_since_admission': days_since,
                'cumulative_all_providers': unique_all
            })
    
    df_cumulative_all = pd.DataFrame(cumulative_all)
    
    df_cumulative = df_cumulative.merge(
        df_cumulative_all[['CSN', 'admission_number_this_csn', 'days_since_admission', 'cumulative_all_providers']],
        on=['CSN', 'admission_number_this_csn', 'days_since_admission'],
        how='left'
    )
    
    print(f"Saving checkpoint to {CHECKPOINT_FILE}...")
    checkpoint_data = {
        'df_cumulative': df_cumulative,
        'df_cumulative_all': df_cumulative_all,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print("Checkpoint saved successfully!")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Cumulative Unique Providers Over Stay', fontsize=16, fontweight='bold')

plot_cols = ['cumulative_nurse', 'cumulative_frontline', 
             'cumulative_attending', 'cumulative_all_primary', 'cumulative_all_providers']
plot_titles = ['Nurses', 'Frontline', 'Attending', 'Primary Team', 'Broader Team']

# Define fixed y-axis limits for each subplot
y_limits = {
    'cumulative_nurse': 40,
    'cumulative_frontline': 40,
    'cumulative_attending': 40,
    'cumulative_all_primary': 100,
    'cumulative_all_providers': 1000
}

for idx, (col, title) in enumerate(zip(plot_cols, plot_titles)):
    ax = axes[idx // 3, idx % 3]
    
    # Plot individual trajectories
    for (csn, adm), group in df_cumulative.groupby(['CSN', 'admission_number_this_csn']):
        if len(group) > 1:
            ax.plot(group['days_since_admission'], group[col], alpha=0.01, color='gray')
    
    # Special handling for Frontline plot - show three median lines
    if col == 'cumulative_frontline':
        # Frontline (All) - use primary color
        median_all = df_cumulative.groupby('days_since_admission')['cumulative_frontline'].median()
        ax.plot(median_all.index, median_all.values, color=COLORS['secondary'], 
                linewidth=2.5, label='All Frontline', linestyle='-')
        
        # Frontline (Resident) - use a distinct color
        median_resident = df_cumulative.groupby('days_since_admission')['cumulative_frontline_resident'].median()
        ax.plot(median_resident.index, median_resident.values, color='darkblue', 
                linewidth=2, label='Resident', linestyle=':')
        
        # Frontline (APP/Hospitalist) - use another distinct color
        median_app_hosp = df_cumulative.groupby('days_since_admission')['cumulative_frontline_app_or_hospitalist'].median()
        ax.plot(median_app_hosp.index, median_app_hosp.values, color='darkgreen', 
                linewidth=2, label='APP/Hospitalist', linestyle=':')
    else:
        # Standard single median line for other plots
        median_by_day = df_cumulative.groupby('days_since_admission')[col].median()
        ax.plot(median_by_day.index, median_by_day.values, color=COLORS['secondary'], 
                linewidth=2, label='Median')
        
    ax.set_xlim(0, 56)
    ax.set_ylim(0, y_limits[col])
    
    ax.set_xlabel('Days Since Admission')
    ax.set_ylabel('Cumulative Unique Providers')
    ax.set_title(title)
    ax.legend(loc='upper left', frameon=True, fancybox=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
save_and_show_plot(fig, 'measure1_cumulative_providers')


# In[ ]:





# ## Measure 2: Team Stability

# In[ ]:





# ### Computation

# In[8]:


# ============================================================================
# MEASURE 2: TEAM STABILITY - COMPUTATION
# ============================================================================

print("\n" + "="*80)
print("MEASURE 2: TEAM STABILITY - COMPUTATION")
print("="*80)

# Checkpoint configuration
CHECKPOINT_FILE_M2 = CHECKPOINT_DIR / "measure2_stability.pkl"
FORCE_RECOMPUTE_M2 = False

if CHECKPOINT_FILE_M2.exists() and not FORCE_RECOMPUTE_M2:
    print(f"\nLoading cached data from {CHECKPOINT_FILE_M2}...")
    with open(CHECKPOINT_FILE_M2, 'rb') as f:
        checkpoint_data = pickle.load(f)
        df_stability = checkpoint_data['df_stability']
        computation_timestamp = checkpoint_data.get('timestamp', 'unknown')
    print(f"✓ Loaded checkpoint from {computation_timestamp}")
    print(f"  Stability data shape: {df_stability.shape}")
else:
    print("\nComputing all stability metrics...")
    print("⏱️  This will take some time for large datasets...\n")
    
    # Start with df_analysis
    df_stability = df_analysis.copy()
    
    # ------------------------------------------------------------------
    # SECTION 1A: INDIVIDUAL PROVIDER STABILITY
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("SECTION 1A: INDIVIDUAL PROVIDER STABILITY")
    print("-"*80)
    
    roles = ['nurse', 'frontline', 'attending']
    windows = [1, 3, 7, 14]
    
    # Create has_lookback columns once (same for all roles)
    print("\n📊 Creating lookback availability flags...")
    for window in windows:
        has_col = f'has_lookback_{window}d'
        df_stability[has_col] = df_stability['days_since_admission'] >= window
        
        # Report exclusions for first window
        if window == 1:
            excluded = (~df_stability[has_col]).sum()
            total = len(df_stability)
            pct = (excluded / total * 100)
            print(f"     ⚠️  Window {window}d: Excluded {excluded:,} patient-days ({pct:.1f}%) due to insufficient lookback")
    
    # Regular stability
    print("\n📊 Computing regular stability metrics...")
    for role in roles:
        for window in windows:
            print(f"\n  → {role.capitalize()} {window}-day stability...")
            result = calculate_individual_stability(df_stability, role, window)
            # Only join the stability column, not has_lookback (already exists)
            df_stability = df_stability.join(result[[f'S_{role}_{window}d']])
    
    # Streak stability
    print("\n📊 Computing streak stability metrics...")
    for role in roles:
        print(f"\n  → {role.capitalize()} streak...")
        result = calculate_individual_streak(df_stability, role)
        df_stability = df_stability.join(result)
    
    # Calculate mean individual stability (combined metric)
    print("\n📊 Computing mean individual stability (combined)...")
    for window in windows:
        cols = [f'S_{role}_{window}d' for role in roles]
        df_stability[f'S_individual_mean_{window}d'] = df_stability[cols].mean(axis=1)
    
    # Mean streak
    streak_cols = [f'S_{role}_streak' for role in roles]
    df_stability['S_individual_mean_streak'] = df_stability[streak_cols].mean(axis=1)
    
    print("\n✓ Section 1A complete! (20 metrics: 15 individual + 5 mean)")
    
    # ------------------------------------------------------------------
    # SECTION 1B: DYADIC STABILITY
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("SECTION 1B: DYADIC STABILITY")
    print("-"*80)
    
    dyads = [('nurse', 'frontline'), ('nurse', 'attending'), ('frontline', 'attending')]
    
    # Regular stability
    print("\n📊 Computing dyadic stability metrics...")
    for pair in dyads:
        for window in windows:
            print(f"\n  → {pair[0].capitalize()}-{pair[1].capitalize()} {window}-day stability...")
            result = calculate_dyadic_stability(df_stability, pair, window)
            pair_name = f"{pair[0][0].upper()}{pair[1][0].upper()}"
            # Only join the stability column
            df_stability = df_stability.join(result[[f'S_{pair_name}_{window}d']])
    
    # Streak stability
    print("\n📊 Computing dyadic streak metrics...")
    for pair in dyads:
        print(f"\n  → {pair[0].capitalize()}-{pair[1].capitalize()} streak...")
        result = calculate_dyadic_streak(df_stability, pair)
        df_stability = df_stability.join(result)
    
    print("\n✓ Section 1B complete! (15 metrics)")
    
    # ------------------------------------------------------------------
    # SECTION 1C: TEAM STABILITY
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("SECTION 1C: TEAM STABILITY")
    print("-"*80)
    
    # Regular stability
    print("\n📊 Computing team stability metrics...")
    for window in windows:
        print(f"\n  → Team {window}-day stability...")
        result = calculate_team_stability(df_stability, window)
        # Only join the stability column
        df_stability = df_stability.join(result[[f'S_team_{window}d']])
    
    # Streak stability
    print(f"\n  → Team streak...")
    result = calculate_team_streak(df_stability)
    df_stability = df_stability.join(result)
    
    print("\n✓ Section 1C complete! (5 metrics)")
    
    # Save checkpoint
    print(f"\n💾 Saving checkpoint to {CHECKPOINT_FILE_M2}...")
    checkpoint_data = {
        'df_stability': df_stability,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(CHECKPOINT_FILE_M2, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print("✓ Checkpoint saved successfully!")

print("\n" + "="*80)
print("✓ ALL STABILITY METRICS COMPUTED (40 total)")
print("="*80)


# ### Summary Stats

# In[ ]:





# In[12]:


# ============================================================================
# MEASURE 2: STABILITY - SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("MEASURE 2: STABILITY - SUMMARY STATISTICS")
print("="*80)

roles = ['nurse', 'frontline', 'attending', 'individual_mean']
windows = [1, 3, 7, 14]

# ------------------------------------------------------------------
# Display 1A: Individual Stability Summary
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 1A: INDIVIDUAL PROVIDER STABILITY")
print("="*70)

for window in windows:
    print(f"\n{'─'*70}")
    print(f"Individual Stability ({window}-day window)")
    print(f"{'─'*70}")
    
    summary_data = []
    for role in roles:
        col = f'S_{role}_{window}d'
        if col in df_stability.columns:
            data = df_stability[col].dropna()
            if len(data) > 0:
                summary_data.append({
                    'Role': role.replace('_', ' ').title(),
                    'Mean': f"{data.mean():.3f}",
                    'Median': f"{data.median():.3f}",
                    'SD': f"{data.std():.3f}",
                    'Min': f"{data.min():.3f}",
                    'Max': f"{data.max():.3f}",
                    'Q25': f"{data.quantile(0.25):.3f}",
                    'Q75': f"{data.quantile(0.75):.3f}",
                    'N': f"{len(data):,}"
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        display(summary_df)

# Streak summary
print(f"\n{'─'*70}")
print(f"Individual Consecutive Streak Duration")
print(f"{'─'*70}")

summary_data = []
for role in roles:
    col = f'S_{role}_streak'
    if col in df_stability.columns:
        data = df_stability[col].dropna()
        if len(data) > 0:
            summary_data.append({
                'Role': role.replace('_', ' ').title(),
                'Mean': f"{data.mean():.2f}",
                'Median': f"{data.median():.0f}",
                'SD': f"{data.std():.2f}",
                'Min': f"{data.min():.0f}",
                'Max': f"{data.max():.0f}",
                'Q25': f"{data.quantile(0.25):.0f}",
                'Q75': f"{data.quantile(0.75):.0f}",
                'N': f"{len(data):,}"
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# ------------------------------------------------------------------
# Display 1B: Dyadic Stability Summary
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 1B: DYADIC STABILITY")
print("="*70)

dyad_labels = {'NF': 'Nurse-Frontline', 'NA': 'Nurse-Attending', 'FA': 'Frontline-Attending'}

for window in windows:
    print(f"\n{'─'*70}")
    print(f"Dyadic Stability ({window}-day window)")
    print(f"{'─'*70}")
    
    summary_data = []
    for pair_code, pair_label in dyad_labels.items():
        col = f'S_{pair_code}_{window}d'
        if col in df_stability.columns:
            data = df_stability[col].dropna()
            if len(data) > 0:
                summary_data.append({
                    'Dyad': pair_label,
                    'Mean': f"{data.mean():.3f}",
                    'Median': f"{data.median():.3f}",
                    'SD': f"{data.std():.3f}",
                    'Min': f"{data.min():.3f}",
                    'Max': f"{data.max():.3f}",
                    'Q25': f"{data.quantile(0.25):.3f}",
                    'Q75': f"{data.quantile(0.75):.3f}",
                    'N': f"{len(data):,}"
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        display(summary_df)

# Streak summary
print(f"\n{'─'*70}")
print(f"Dyadic Consecutive Streak Duration")
print(f"{'─'*70}")

summary_data = []
for pair_code, pair_label in dyad_labels.items():
    col = f'S_{pair_code}_streak'
    if col in df_stability.columns:
        data = df_stability[col].dropna()
        if len(data) > 0:
            summary_data.append({
                'Dyad': pair_label,
                'Mean': f"{data.mean():.2f}",
                'Median': f"{data.median():.0f}",
                'SD': f"{data.std():.2f}",
                'Min': f"{data.min():.0f}",
                'Max': f"{data.max():.0f}",
                'Q25': f"{data.quantile(0.25):.0f}",
                'Q75': f"{data.quantile(0.75):.0f}",
                'N': f"{len(data):,}"
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# ------------------------------------------------------------------
# Display 1C: Team Stability Summary
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 1C: TEAM STABILITY")
print("="*70)

print(f"\n{'─'*70}")
print(f"Team Stability (All Windows)")
print(f"{'─'*70}")

summary_data = []
for window in windows:
    col = f'S_team_{window}d'
    if col in df_stability.columns:
        data = df_stability[col].dropna()
        if len(data) > 0:
            summary_data.append({
                'Window': f'{window}-day',
                'Mean': f"{data.mean():.3f}",
                'Median': f"{data.median():.3f}",
                'SD': f"{data.std():.3f}",
                'Min': f"{data.min():.3f}",
                'Max': f"{data.max():.3f}",
                'Q25': f"{data.quantile(0.25):.3f}",
                'Q75': f"{data.quantile(0.75):.3f}",
                'N': f"{len(data):,}"
            })

col = 'S_team_streak'
if col in df_stability.columns:
    data = df_stability[col].dropna()
    if len(data) > 0:
        summary_data.append({
            'Window': 'Streak',
            'Mean': f"{data.mean():.2f}",
            'Median': f"{data.median():.0f}",
            'SD': f"{data.std():.2f}",
            'Min': f"{data.min():.0f}",
            'Max': f"{data.max():.0f}",
            'Q25': f"{data.quantile(0.25):.0f}",
            'Q75': f"{data.quantile(0.75):.0f}",
            'N': f"{len(data):,}"
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# Export stability summary to CSV
print("\n💾 Exporting stability summary statistics to CSV...")
stability_summary_file = DATA_DIR / "stability_summary_statistics.csv"
all_summary = []

# Individual
for window in windows:
    for role in roles:
        col = f'S_{role}_{window}d'
        if col in df_stability.columns:
            data = df_stability[col].dropna()
            if len(data) > 0:
                all_summary.append({
                    'Level': 'Individual',
                    'Unit': role.replace('_', ' ').title(),
                    'Metric': f'{window}-day',
                    'Mean': data.mean(),
                    'Median': data.median(),
                    'SD': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Q25': data.quantile(0.25),
                    'Q75': data.quantile(0.75),
                    'N': len(data)
                })

# Dyadic
for window in windows:
    for pair_code, pair_label in dyad_labels.items():
        col = f'S_{pair_code}_{window}d'
        if col in df_stability.columns:
            data = df_stability[col].dropna()
            if len(data) > 0:
                all_summary.append({
                    'Level': 'Dyadic',
                    'Unit': pair_label,
                    'Metric': f'{window}-day',
                    'Mean': data.mean(),
                    'Median': data.median(),
                    'SD': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Q25': data.quantile(0.25),
                    'Q75': data.quantile(0.75),
                    'N': len(data)
                })

# Team
for window in windows:
    col = f'S_team_{window}d'
    if col in df_stability.columns:
        data = df_stability[col].dropna()
        if len(data) > 0:
            all_summary.append({
                'Level': 'Team',
                'Unit': 'Complete Team',
                'Metric': f'{window}-day',
                'Mean': data.mean(),
                'Median': data.median(),
                'SD': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Q25': data.quantile(0.25),
                'Q75': data.quantile(0.75),
                'N': len(data)
            })

summary_export_df = pd.DataFrame(all_summary)
summary_export_df.to_csv(stability_summary_file, index=False)
print(f"✓ Saved to: {stability_summary_file}")


# ### Visuals

# In[13]:


# ============================================================================
# VISUALIZATIONS - SETUP
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create day bins for temporal analysis
print("\nPreparing data for temporal visualizations...")
bins = [0, 8, 16, 24, 32, 100]
bin_labels = ['1-8', '9-16', '17-24', '25-32', '33+']
df_familiarity['day_bin'] = pd.cut(df_familiarity['days_since_admission'], 
                                   bins=bins, labels=bin_labels, right=False)

# Color definitions
role_colors = {
    'nurse': COLORS['palette'][0], 
    'frontline': COLORS['palette'][1], 
    'attending': COLORS['palette'][2], 
    'individual_mean': COLORS['palette'][5]
}

dyad_colors = {
    'NF': COLORS['palette'][0], 
    'NA': COLORS['palette'][1], 
    'FA': COLORS['palette'][2]
}

context_colors = {
    'C1': COLORS['quinary'], 
    'C2': COLORS['quaternary'], 
    'C3': COLORS['senary']
}

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

print("✓ Data prepared for visualization")

# ============================================================================
# STABILITY VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("PART 1: STABILITY VISUALIZATIONS")
print("="*80)

# ------------------------------------------------------------------
# INDIVIDUAL STABILITY PLOTS
# ------------------------------------------------------------------
print("\nGenerating individual stability plots...")

# Plot 1A-1: Mean Individual Stability Bar Plot (Like Legacy Code)
print("  → Plot 1A-1: Mean individual stability comparison (bar plot)...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Mean Individual Provider Stability Across Time Windows', 
             fontsize=14, fontweight='bold')

x_positions = np.arange(len(windows))
width = 0.2

for i, role in enumerate(['nurse', 'frontline', 'attending']):
    means = []
    for window in windows:
        col = f'S_{role}_{window}d'
        if col in df_familiarity.columns:
            means.append(df_familiarity[col].mean())
        else:
            means.append(0)
    
    ax.bar(x_positions + i*width, means, width, 
           label=role.capitalize(), color=role_colors[role], alpha=0.7)

ax.set_xlabel('Time Window', fontsize=12)
ax.set_ylabel('Mean Stability Score', fontsize=12)
ax.set_xticks(x_positions + width)
ax.set_xticklabels([f'{w}-day' for w in windows])
ax.legend(loc='upper left')
ax.set_ylim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1A_1_mean_individual_bars')

# Plot 1A-2: Distribution Grid
print("  → Plot 1A-2: Distribution grid...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Individual Provider Stability - Distribution by Window', 
             fontsize=16, fontweight='bold')

for idx, window in enumerate(windows):
    ax = axes[idx // 2, idx % 2]
    
    for role in ['nurse', 'frontline', 'attending']:
        col = f'S_{role}_{window}d'
        if col in df_familiarity.columns:
            data = df_familiarity[col].dropna()
            ax.hist(data, bins=30, alpha=0.4, label=role.capitalize(), 
                   color=role_colors[role], edgecolor='black')
    
    ax.set_xlabel('Stability Score [0,1]', fontsize=10)
    ax.set_ylabel('Count of Patient-Days', fontsize=10)
    ax.set_title(f'{window}-Day Window', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1A_2_distribution_grid')

# Plot 1A-3: Streak Distribution
print("  → Plot 1A-3: Streak distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Individual Provider Consecutive Streak Distribution', 
             fontsize=14, fontweight='bold')

for role in ['nurse', 'frontline', 'attending']:
    col = f'S_{role}_streak'
    if col in df_familiarity.columns:
        data = df_familiarity[col].dropna()
        p95 = data.quantile(0.95)
        data_capped = data.clip(upper=p95)
        
        ax.hist(data_capped, bins=30, alpha=0.4, label=role.capitalize(),
               color=role_colors[role], edgecolor='black')

ax.set_xlabel('Consecutive Days', fontsize=12)
ax.set_ylabel('Count of Patient-Days', fontsize=12)
ax.legend(loc='upper right')

# Add truncation indicator
max_val = max([df_familiarity[f'S_{role}_streak'].max() 
              for role in ['nurse', 'frontline', 'attending']
              if f'S_{role}_streak' in df_familiarity.columns])
p95_val = max([df_familiarity[f'S_{role}_streak'].quantile(0.95) 
              for role in ['nurse', 'frontline', 'attending']
              if f'S_{role}_streak' in df_familiarity.columns])
if max_val > p95_val:
    add_truncation_indicator(ax, df_familiarity['S_nurse_streak'], percentile=95)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1A_3_streak_distribution')

# Plot 1A-4: Temporal Evolution (one plot with 4 lines for windows)
print("  → Plot 1A-4: Temporal evolution...")
for role in ['nurse', 'frontline', 'attending', 'individual_mean']:
    fig, ax = plt.subplots(figsize=(10, 6))
    role_label = role.replace('_', ' ').title()
    fig.suptitle(f'{role_label} Stability Over Admission Length', 
                 fontsize=14, fontweight='bold')
    
    for window in windows:
        col = f'S_{role}_{window}d'
        if col in df_familiarity.columns:
            means = df_familiarity.groupby('day_bin')[col].mean()
            ax.plot(range(len(means)), means.values, marker='o', linewidth=2,
                   label=f'{window}-day', alpha=0.8)
    
    ax.set_xlabel('Days Since Admission', fontsize=12)
    ax.set_ylabel('Mean Stability Score', fontsize=12)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='best', title='Window')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    safe_role = role.replace('/', '_')
    save_and_show_plot(fig, PLOTS_DIR / f'stability_1A_4_temporal_{safe_role}')

print("✓ Individual stability plots complete!")

# ------------------------------------------------------------------
# DYADIC STABILITY PLOTS
# ------------------------------------------------------------------
print("\nGenerating dyadic stability plots...")

# Plot 1B-1: Distribution Grid
print("  → Plot 1B-1: Distribution grid...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Dyadic Stability - Distribution by Window', 
             fontsize=16, fontweight='bold')

for idx, window in enumerate(windows):
    ax = axes[idx // 2, idx % 2]
    
    for dyad_code in ['NF', 'NA', 'FA']:
        col = f'S_{dyad_code}_{window}d'
        if col in df_familiarity.columns:
            data = df_familiarity[col].dropna()
            ax.hist(data, bins=30, alpha=0.4, label=dyad_labels[dyad_code],
                   color=dyad_colors[dyad_code], edgecolor='black')
    
    ax.set_xlabel('Stability Score [0,1]', fontsize=10)
    ax.set_ylabel('Count of Patient-Days', fontsize=10)
    ax.set_title(f'{window}-Day Window', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1B_1_distribution_grid')

# Plot 1B-2: Streak Distribution
print("  → Plot 1B-2: Streak distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Dyadic Consecutive Streak Distribution', 
             fontsize=14, fontweight='bold')

for dyad_code in ['NF', 'NA', 'FA']:
    col = f'S_{dyad_code}_streak'
    if col in df_familiarity.columns:
        data = df_familiarity[col].dropna()
        p95 = data.quantile(0.95)
        data_capped = data.clip(upper=p95)
        
        ax.hist(data_capped, bins=30, alpha=0.4, label=dyad_labels[dyad_code],
               color=dyad_colors[dyad_code], edgecolor='black')

ax.set_xlabel('Consecutive Days', fontsize=12)
ax.set_ylabel('Count of Patient-Days', fontsize=12)
ax.legend(loc='upper right')

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1B_2_streak_distribution')

# Plot 1B-3: Temporal Evolution (one per dyad)
print("  → Plot 1B-3: Temporal evolution...")
for dyad_code, dyad_label in dyad_labels.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'{dyad_label} Stability Over Admission Length', 
                 fontsize=14, fontweight='bold')
    
    for window in windows:
        col = f'S_{dyad_code}_{window}d'
        if col in df_familiarity.columns:
            means = df_familiarity.groupby('day_bin')[col].mean()
            ax.plot(range(len(means)), means.values, marker='o', linewidth=2,
                   label=f'{window}-day', alpha=0.8)
    
    ax.set_xlabel('Days Since Admission', fontsize=12)
    ax.set_ylabel('Mean Stability Score', fontsize=12)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='best', title='Window')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_and_show_plot(fig, PLOTS_DIR / f'stability_1B_3_temporal_{dyad_code}')

print("✓ Dyadic stability plots complete!")

# ------------------------------------------------------------------
# TEAM STABILITY PLOTS
# ------------------------------------------------------------------
print("\nGenerating team stability plots...")

# Plot 1C-1: Team Stability Bar Plot (Individuals next to each other - like legacy)
print("  → Plot 1C-1: Team stability comparison (bar plot)...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Complete Team Stability Across Time Windows', 
             fontsize=14, fontweight='bold')

x_positions = np.arange(len(windows))
width = 0.8

# Just team values
team_means = []
for window in windows:
    col = f'S_team_{window}d'
    if col in df_familiarity.columns:
        team_means.append(df_familiarity[col].mean())
    else:
        team_means.append(0)

ax.bar(x_positions, team_means, width, 
       color=COLORS['palette'][3], alpha=0.7, label='Complete Team')

ax.set_xlabel('Time Window', fontsize=12)
ax.set_ylabel('Mean Stability Score', fontsize=12)
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{w}-day' for w in windows])
ax.legend(loc='upper left')
ax.set_ylim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1C_1_team_bars')

# Plot 1C-2: Distribution Grid
print("  → Plot 1C-2: Distribution grid...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Team Stability - Distribution by Window', 
             fontsize=16, fontweight='bold')

for idx, window in enumerate(windows):
    ax = axes[idx // 2, idx % 2]
    
    col = f'S_team_{window}d'
    if col in df_familiarity.columns:
        data = df_familiarity[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color=COLORS['palette'][3],
               edgecolor='black')
    
    ax.set_xlabel('Stability Score [0,1]', fontsize=10)
    ax.set_ylabel('Count of Patient-Days', fontsize=10)
    ax.set_title(f'{window}-Day Window', fontsize=12)
    ax.set_xlim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1C_2_distribution_grid')

# Plot 1C-3: Streak Distribution
print("  → Plot 1C-3: Streak distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Team Consecutive Streak Distribution', 
             fontsize=14, fontweight='bold')

col = 'S_team_streak'
if col in df_familiarity.columns:
    data = df_familiarity[col].dropna()
    p95 = data.quantile(0.95)
    data_capped = data.clip(upper=p95)
    
    ax.hist(data_capped, bins=30, alpha=0.7, color=COLORS['palette'][3],
           edgecolor='black')
    
    if data.max() > p95:
        add_truncation_indicator(ax, data, percentile=95)

ax.set_xlabel('Consecutive Days', fontsize=12)
ax.set_ylabel('Count of Patient-Days', fontsize=12)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1C_3_streak_distribution')

# Plot 1C-4: Temporal Evolution
print("  → Plot 1C-4: Temporal evolution...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Complete Team Stability Over Admission Length', 
             fontsize=14, fontweight='bold')

for window in windows:
    col = f'S_team_{window}d'
    if col in df_familiarity.columns:
        means = df_familiarity.groupby('day_bin')[col].mean()
        ax.plot(range(len(means)), means.values, marker='o', linewidth=2,
               label=f'{window}-day', alpha=0.8)

ax.set_xlabel('Days Since Admission', fontsize=12)
ax.set_ylabel('Mean Stability Score', fontsize=12)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels)
ax.legend(loc='best', title='Window')
ax.set_ylim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'stability_1C_4_temporal_team')

print("✓ Team stability plots complete!")

print("\n" + "="*80)
print("✓ STABILITY VISUALIZATIONS COMPLETE")
print("="*80)


# ## Measure 3: Team Familiarity

# ### Computation

# In[ ]:


# ============================================================================
# MEASURE 3: TEAM FAMILIARITY - COMPUTATION
# ============================================================================

print("\n" + "="*80)
print("MEASURE 3: TEAM FAMILIARITY - COMPUTATION")
print("="*80)

# Checkpoint configuration
CHECKPOINT_FILE_M3 = CHECKPOINT_DIR / "measure3_familiarity.pkl"
FORCE_RECOMPUTE_M3 = False

if CHECKPOINT_FILE_M3.exists() and not FORCE_RECOMPUTE_M3:
    print(f"\nLoading cached data from {CHECKPOINT_FILE_M3}...")
    with open(CHECKPOINT_FILE_M3, 'rb') as f:
        checkpoint_data = pickle.load(f)
        df_familiarity = checkpoint_data['df_familiarity']
        computation_timestamp = checkpoint_data.get('timestamp', 'unknown')
    print(f"✓ Loaded checkpoint from {computation_timestamp}")
    print(f"  Familiarity data shape: {df_familiarity.shape}")
else:
    print("\nComputing all familiarity metrics...")
    print("⏱️  This will take some time for large datasets...\n")
    
    # Start with df_stability (which includes all base data)
    df_familiarity = df_stability.copy()
    
    # ------------------------------------------------------------------
    # SECTION 2A: DYADIC FAMILIARITY
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("SECTION 2A: DYADIC FAMILIARITY")
    print("-"*80)
    
    dyads = [('nurse', 'frontline'), ('nurse', 'attending'), ('frontline', 'attending')]
    contexts = ['C1', 'C2', 'C3']
    output_types = ['binary', 'count', 'rate']
    
    for pair in dyads:
        for context in contexts:
            for output_type in output_types:
                print(f"\n  → {pair[0].capitalize()}-{pair[1].capitalize()} {context} {output_type}...")
                result = calculate_dyadic_familiarity(df_familiarity, df_team, 
                                                     pair, context, output_type)
                df_familiarity = df_familiarity.join(result)
    
    print("\n✓ Section 2A complete! (27 metrics)")
    
    # ------------------------------------------------------------------
    # SECTION 2B: TEAM FAMILIARITY
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("SECTION 2B: TEAM FAMILIARITY")
    print("-"*80)
    
    for context in contexts:
        for output_type in output_types:
            print(f"\n  → Team {context} {output_type}...")
            result = calculate_team_familiarity(df_familiarity, df_team, 
                                              context, output_type)
            df_familiarity = df_familiarity.join(result)
    
    print("\n✓ Section 2B complete! (9 metrics)")
    
    # Save checkpoint
    print(f"\n💾 Saving checkpoint to {CHECKPOINT_FILE_M3}...")
    checkpoint_data = {
        'df_familiarity': df_familiarity,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(CHECKPOINT_FILE_M3, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print("✓ Checkpoint saved successfully!")

print("\n" + "="*80)
print("✓ ALL FAMILIARITY METRICS COMPUTED (36 total)")
print("="*80)


# ### Summary Stats

# In[ ]:


# ============================================================================
# MEASURE 3: FAMILIARITY - SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("MEASURE 3: FAMILIARITY - SUMMARY STATISTICS")
print("="*80)

# ------------------------------------------------------------------
# Display 2A: Dyadic Binary Familiarity
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 2A: DYADIC FAMILIARITY - BINARY")
print("="*70)
print("\n% of Patient-Days with Prior Collaboration")
print("-"*70)

summary_data = []
for pair_code, pair_label in dyad_labels.items():
    row_data = {'Dyad': pair_label}
    for context in ['C1', 'C2', 'C3']:
        col = f'F_{pair_code}_{context}_binary'
        if col in df_familiarity.columns:
            data = df_familiarity[col]
            pct = (data == 1).mean() * 100
            row_data[context] = f"{pct:.1f}%"
    row_data['N'] = f"{len(df_familiarity):,}"
    summary_data.append(row_data)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# ------------------------------------------------------------------
# Display 2A: Dyadic Count Familiarity
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 2A: DYADIC FAMILIARITY - COUNT")
print("="*70)
print("\nMean # of Prior Days Together (when familiarity exists)")
print("-"*70)

summary_data = []
for pair_code, pair_label in dyad_labels.items():
    row_data = {'Dyad': pair_label}
    for context in ['C1', 'C2', 'C3']:
        col_bin = f'F_{pair_code}_{context}_binary'
        col_cnt = f'F_{pair_code}_{context}_count'
        if col_bin in df_familiarity.columns and col_cnt in df_familiarity.columns:
            # Calculate mean/SD for cases where binary=1
            familiar = df_familiarity[df_familiarity[col_bin] == 1]
            if len(familiar) > 0:
                mean_val = familiar[col_cnt].mean()
                sd_val = familiar[col_cnt].std()
                row_data[context] = f"{mean_val:.1f} ({sd_val:.1f})"
                row_data[f'{context}_N'] = len(familiar)
            else:
                row_data[context] = "N/A"
                row_data[f'{context}_N'] = 0
    summary_data.append(row_data)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# ------------------------------------------------------------------
# Display 2A: Dyadic Rate Familiarity
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 2A: DYADIC FAMILIARITY - RATE")
print("="*70)
print("\nMean Proportion of Eligible Days Worked Together")
print("-"*70)

summary_data = []
for pair_code, pair_label in dyad_labels.items():
    row_data = {'Dyad': pair_label}
    for context in ['C1', 'C2', 'C3']:
        col = f'F_{pair_code}_{context}_rate'
        if col in df_familiarity.columns:
            data = df_familiarity[col]
            mean_val = data.mean()
            sd_val = data.std()
            row_data[context] = f"{mean_val:.3f} ({sd_val:.3f})"
    row_data['N'] = f"{len(df_familiarity):,}"
    summary_data.append(row_data)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# ------------------------------------------------------------------
# Display 2B: Team Familiarity Summary
# ------------------------------------------------------------------
print("\n\n" + "="*70)
print("SECTION 2B: TEAM FAMILIARITY")
print("="*70)

print(f"\n{'─'*70}")
print(f"Team Familiarity Summary")
print(f"{'─'*70}")

summary_data = []

# Binary
row_data = {'Metric': 'Binary (% = 1)'}
for context in ['C1', 'C2', 'C3']:
    col = f'F_team_{context}_binary'
    if col in df_familiarity.columns:
        data = df_familiarity[col]
        pct = (data == 1).mean() * 100
        row_data[context] = f"{pct:.1f}%"
row_data['N'] = f"{len(df_familiarity):,}"
summary_data.append(row_data)

# Count
row_data = {'Metric': 'Count (Mean ± SD)'}
for context in ['C1', 'C2', 'C3']:
    col = f'F_team_{context}_count'
    if col in df_familiarity.columns:
        data = df_familiarity[col]
        mean_val = data.mean()
        sd_val = data.std()
        row_data[context] = f"{mean_val:.1f} ± {sd_val:.1f}"
row_data['N'] = f"{len(df_familiarity):,}"
summary_data.append(row_data)

# Rate
row_data = {'Metric': 'Rate (Mean ± SD)'}
for context in ['C1', 'C2', 'C3']:
    col = f'F_team_{context}_rate'
    if col in df_familiarity.columns:
        data = df_familiarity[col]
        mean_val = data.mean()
        sd_val = data.std()
        row_data[context] = f"{mean_val:.3f} ± {sd_val:.3f}"
row_data['N'] = f"{len(df_familiarity):,}"
summary_data.append(row_data)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# Export familiarity summary to CSV
print("\n💾 Exporting familiarity summary statistics to CSV...")
familiarity_summary_file = DATA_DIR / "familiarity_summary_statistics.csv"
all_fam_summary = []

contexts = ['C1', 'C2', 'C3']
context_labels_export = {'C1': 'This Admission', 'C2': 'Previous Admissions', 'C3': 'Any Patient'}

# Dyadic familiarity
for pair_code, pair_label in dyad_labels.items():
    for context in contexts:
        for output_type in ['binary', 'count', 'rate']:
            col = f'F_{pair_code}_{context}_{output_type}'
            if col in df_familiarity.columns:
                data = df_familiarity[col]
                if output_type == 'binary':
                    value = (data == 1).mean() * 100
                    all_fam_summary.append({
                        'Level': 'Dyadic',
                        'Unit': pair_label,
                        'Context': context_labels_export[context],
                        'Metric': 'Binary (%)',
                        'Value': value
                    })
                else:
                    all_fam_summary.append({
                        'Level': 'Dyadic',
                        'Unit': pair_label,
                        'Context': context_labels_export[context],
                        'Metric': f'{output_type.capitalize()} (Mean)',
                        'Value': data.mean()
                    })

# Team familiarity
for context in contexts:
    for output_type in ['binary', 'count', 'rate']:
        col = f'F_team_{context}_{output_type}'
        if col in df_familiarity.columns:
            data = df_familiarity[col]
            if output_type == 'binary':
                value = (data == 1).mean() * 100
                all_fam_summary.append({
                    'Level': 'Team',
                    'Unit': 'Complete Team',
                    'Context': context_labels_export[context],
                    'Metric': 'Binary (%)',
                    'Value': value
                })
            else:
                all_fam_summary.append({
                    'Level': 'Team',
                    'Unit': 'Complete Team',
                    'Context': context_labels_export[context],
                    'Metric': f'{output_type.capitalize()} (Mean)',
                    'Value': data.mean()
                })

fam_export_df = pd.DataFrame(all_fam_summary)
fam_export_df.to_csv(familiarity_summary_file, index=False)
print(f"✓ Saved to: {familiarity_summary_file}")


# ### Visuals

# In[ ]:


# ============================================================================
# FAMILIARITY VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("PART 2: FAMILIARITY VISUALIZATIONS")
print("="*80)

context_labels_plot = {'C1': 'This Admission', 'C2': 'Previous Admissions', 'C3': 'Any Patient'}

# ------------------------------------------------------------------
# DYADIC & TEAM FAMILIARITY - COMBINED BAR PLOTS
# ------------------------------------------------------------------
print("\nGenerating familiarity comparison bar plots...")

# Plot 2A-1: Binary Familiarity by Context (Dyadic + Team)
print("  → Plot 2A-1: Binary familiarity by context...")
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Binary Familiarity by Context\n(% of Patient-Days with Prior Collaboration)', 
             fontsize=14, fontweight='bold')

plot_data = []
for dyad_code in ['NF', 'NA', 'FA']:
    for context in contexts:
        col = f'F_{dyad_code}_{context}_binary'
        if col in df_familiarity.columns:
            data = df_familiarity[col]
            pct = (data == 1).mean() * 100
            plot_data.append({
                'Unit': dyad_labels[dyad_code],
                'Context': context_labels_plot[context],
                'Percentage': pct
            })

# Add team
for context in contexts:
    col = f'F_team_{context}_binary'
    if col in df_familiarity.columns:
        data = df_familiarity[col]
        pct = (data == 1).mean() * 100
        plot_data.append({
            'Unit': 'Team',
            'Context': context_labels_plot[context],
            'Percentage': pct
        })

if plot_data:
    plot_df = pd.DataFrame(plot_data)
    
    units = ['Nurse-Frontline', 'Nurse-Attending', 'Frontline-Attending', 'Team']
    x = np.arange(len(units))
    width = 0.25
    
    for i, context in enumerate(contexts):
        context_label = context_labels_plot[context]
        context_data = plot_df[plot_df['Context'] == context_label]
        percentages = [context_data[context_data['Unit'] == unit]['Percentage'].values[0]
                      if len(context_data[context_data['Unit'] == unit]) > 0 else 0
                      for unit in units]
        
        ax.bar(x + i*width, percentages, width, label=context_label,
              color=context_colors[context], alpha=0.7)
    
    ax.set_xlabel('Dyad / Team', fontsize=12)
    ax.set_ylabel('% of Patient-Days', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(units, rotation=15, ha='right')
    ax.legend(loc='upper right', title='Context')
    ax.set_ylim(0, 100)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'familiarity_2A_1_binary_by_context')

# Plot 2A-2: Count Familiarity by Context
print("  → Plot 2A-2: Count familiarity by context...")
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Count Familiarity by Context\n(Mean # of Prior Days | given familiarity exists)', 
             fontsize=14, fontweight='bold')

plot_data = []
for dyad_code in ['NF', 'NA', 'FA']:
    for context in contexts:
        col_bin = f'F_{dyad_code}_{context}_binary'
        col_cnt = f'F_{dyad_code}_{context}_count'
        if col_bin in df_familiarity.columns and col_cnt in df_familiarity.columns:
            familiar = df_familiarity[df_familiarity[col_bin] == 1]
            if len(familiar) > 0:
                mean_val = familiar[col_cnt].mean()
                plot_data.append({
                    'Unit': dyad_labels[dyad_code],
                    'Context': context_labels_plot[context],
                    'Mean_Count': mean_val
                })

# Add team
for context in contexts:
    col_bin = f'F_team_{context}_binary'
    col_cnt = f'F_team_{context}_count'
    if col_bin in df_familiarity.columns and col_cnt in df_familiarity.columns:
        familiar = df_familiarity[df_familiarity[col_bin] == 1]
        if len(familiar) > 0:
            mean_val = familiar[col_cnt].mean()
            plot_data.append({
                'Unit': 'Team',
                'Context': context_labels_plot[context],
                'Mean_Count': mean_val
            })

if plot_data:
    plot_df = pd.DataFrame(plot_data)
    
    units = ['Nurse-Frontline', 'Nurse-Attending', 'Frontline-Attending', 'Team']
    x = np.arange(len(units))
    width = 0.25
    
    for i, context in enumerate(contexts):
        context_label = context_labels_plot[context]
        context_data = plot_df[plot_df['Context'] == context_label]
        counts = [context_data[context_data['Unit'] == unit]['Mean_Count'].values[0]
                 if len(context_data[context_data['Unit'] == unit]) > 0 else 0
                 for unit in units]
        
        ax.bar(x + i*width, counts, width, label=context_label,
              color=context_colors[context], alpha=0.7)
    
    ax.set_xlabel('Dyad / Team', fontsize=12)
    ax.set_ylabel('Mean # of Prior Days', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(units, rotation=15, ha='right')
    ax.legend(loc='upper right', title='Context')

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'familiarity_2A_2_count_by_context')

# Plot 2A-3: Rate Familiarity by Context
print("  → Plot 2A-3: Rate familiarity by context...")
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Rate Familiarity by Context\n(Mean proportion of eligible days worked together)', 
             fontsize=14, fontweight='bold')

plot_data = []
for dyad_code in ['NF', 'NA', 'FA']:
    for context in contexts:
        col = f'F_{dyad_code}_{context}_rate'
        if col in df_familiarity.columns:
            data = df_familiarity[col]
            mean_val = data.mean()
            plot_data.append({
                'Unit': dyad_labels[dyad_code],
                'Context': context_labels_plot[context],
                'Mean_Rate': mean_val
            })

# Add team
for context in contexts:
    col = f'F_team_{context}_rate'
    if col in df_familiarity.columns:
        data = df_familiarity[col]
        mean_val = data.mean()
        plot_data.append({
            'Unit': 'Team',
            'Context': context_labels_plot[context],
            'Mean_Rate': mean_val
        })

if plot_data:
    plot_df = pd.DataFrame(plot_data)
    
    units = ['Nurse-Frontline', 'Nurse-Attending', 'Frontline-Attending', 'Team']
    x = np.arange(len(units))
    width = 0.25
    
    for i, context in enumerate(contexts):
        context_label = context_labels_plot[context]
        context_data = plot_df[plot_df['Context'] == context_label]
        rates = [context_data[context_data['Unit'] == unit]['Mean_Rate'].values[0]
                if len(context_data[context_data['Unit'] == unit]) > 0 else 0
                for unit in units]
        
        ax.bar(x + i*width, rates, width, label=context_label,
              color=context_colors[context], alpha=0.7)
    
    ax.set_xlabel('Dyad / Team', fontsize=12)
    ax.set_ylabel('Mean Rate [0,1]', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(units, rotation=15, ha='right')
    ax.legend(loc='upper right', title='Context')
    ax.set_ylim(0, 1)

plt.tight_layout()
save_and_show_plot(fig, PLOTS_DIR / 'familiarity_2A_3_rate_by_context')

print("✓ Main familiarity comparison plots complete!")

# ------------------------------------------------------------------
# TEMPORAL FAMILIARITY PLOTS
# ------------------------------------------------------------------
print("\nGenerating temporal familiarity plots...")

# Binary over time (one per context)
for context in contexts:
    print(f"  → Plotting binary temporal evolution for {context}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Binary Familiarity Over Admission - {context_labels_plot[context]}', 
                 fontsize=14, fontweight='bold')
    
    for dyad_code in ['NF', 'NA', 'FA']:
        col = f'F_{dyad_code}_{context}_binary'
        if col in df_familiarity.columns:
            percentages = df_familiarity.groupby('day_bin')[col].apply(lambda x: (x==1).mean() * 100)
            ax.plot(range(len(percentages)), percentages.values, marker='o', linewidth=2,
                   label=dyad_labels[dyad_code], color=dyad_colors[dyad_code])
    
    ax.set_xlabel('Days Since Admission', fontsize=12)
    ax.set_ylabel('% with Prior Collaboration', fontsize=12)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='best')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    save_and_show_plot(fig, PLOTS_DIR / f'familiarity_2A_4_binary_temporal_{context}')

print("✓ Familiarity visualizations complete!")

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS COMPLETE!")
print("="*80)

