#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import difflib
import glob
import warnings
from tqdm.notebook import tqdm
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

"""
Optimized Asymmetric Threshold of Inactivity Sensitivity Analysis
----------------------------------------------------------------

OPTIMIZATIONS IMPLEMENTED:
1. Parallel processing of parameter combinations using multiprocessing
2. Load input files once and reuse them in memory
3. Eliminate intermediate CSV file writes during parameter sweep

SETUP INSTRUCTIONS:
1. Create the following directories in the same folder as this script:
   - inputs/        Place all raw CSV files here (e.g., "username_date.csv")
   - ground_truth/  Place all manually labeled files here (must be named "username_date_MANUALLYLABELED.csv")
   - results/       This is where output files will be saved (will be created if it doesn't exist)

2. Run this script. Results will be saved to the results/ directory.
"""

# Set plot style for better notebook display
plt.style.use('seaborn-v0_8')
sns.set_context("notebook", font_scale=1.2)

################################################################################
# Task Dictionary from Original Code
# ###############################################################################

task_dictionary = {
  "Chart_Review": [
    "Chart","Review","Chart Review","Timeline","IP Summary","Web","Resources","Web Resources",
    "Intake/Output","Intake","Output","Avatar","Growth","Growth Chart","Infant Development","Sidebar",
    "Sidebar Summary","Synopsis","BPA Review","Episodes of Care","Health Maintenance","Immunizations",
    "MAR","Medications","History","Implants","Review Flowsheets","Graphs","Images","Media","Manager",
    "Media Manager","Document List","Scans","View Scans","Document List","Request Outside","Doc Flowsheets",
    "Gender and Sexual Orientation","Problem","Problem List","Rounding","Annotated Images","Enter/Edit Results",
    "Communications","Additional Tools","Clinical Calculator","References","Acquire","HIM","List Disclosures",
    "Quick Disclosure","Endo/Diab","ROP SmartForm","Sedation","Antimicrobial Stewards","Asthma Action Plan",
    "Connect to Video","Education","Newborn ROP Screen","Occluded Vessels","Patient Goals","Prep for Surgery",
    "Restraint Note","SureScripts","Univ Protocol","WARM Scoring","CHOIR","Ehlers-Danlos Checklist","NM Tools",
    "Research Studies","Suspected child abuse","Asthma Pathway","Asthma","Autopsy","Bass Center","Behavior",
    "Behavior Plan","Dialysis","Neonatal MDR"
  ],
  "In_Basket": ["In Basket", "Send Message"],
  "Login": ["Login"],
  "Navigation": ["Inpatient Provider Dashboard","Provider","Dashboard","Welcome","Patient Lists"],
  "Note_Entry": ["Procedures","Charge","Note Entry","Charge Capture","Edit Note","Charge Capture Goals of Care"],
  "Note_Review": ["Notes","My Notes Settings","Note Review"],
  "Order_Entry": ["Orders","ADT Navigators","Order","Order Review","Finish Order Reconciliation","Order Sets",
                  "Admit","Discharge","Order Entry"],
  "Other": ["NaN"],
  "Results_Review": ["Results","Review","Results Review"]
}

# Flags to indicate active/inactive state
active = True
inactive = False

################################################################################
# Helper Functions 
# ###############################################################################

def get_task_dictionary_match(text: str) -> str:
    """Fuzzy-match a text against task_dictionary keys."""
    best_key = "Other"
    best_score = 0.0
    for key, synonyms in task_dictionary.items():
        for val in synonyms:
            score = difflib.SequenceMatcher(None, text.lower(), val.lower()).ratio()
            if score > best_score:
                best_score = score
                best_key = key
    return best_key if best_score > 0.8 else "Other"


def get_task(i, df):
    """
    Determine the 'title' for row i based on page_title / sidebar.
    If page_title is "Other" but sidebar has a recognized match, we pick that.
    """
    page_title = str(df["page_title"].iloc[i])
    sidebar    = str(df["sidebar"].iloc[i])
    matched_pt = get_task_dictionary_match(page_title)
    matched_sb = get_task_dictionary_match(sidebar)
    if matched_pt == "Other" and matched_sb != "Other":
        return matched_sb
    return matched_pt


def get_state_from_mse(mse_val, threshold=0.1) -> bool:
    """Return True (active) if MSE exceeds threshold."""
    return mse_val > threshold


def next_frames_titles_same(i, data, title_change_threshold):
    """Check if the next title_change_threshold rows have the same page_title."""
    if i + title_change_threshold > len(data):
        return False
    subset = set(data["page_title"].iloc[i : i + title_change_threshold])
    return len(subset) == 1


def enough_rows(i, data, threshold):
    """Return True if there are at least 'threshold' rows starting from index i."""
    return i + threshold <= len(data)


def get_prelogin_count(df, threshold=0.1, n_frames=10) -> int:
    """
    Count consecutive frames from the start where %change <= threshold.
    Stop at n_frames or on the first frame with MSE > threshold.
    Returns the number of frames to force as 'Other' in the prelogin block.
    """
    count = 0
    for i in range(len(df)):
        mse_val = df["%change from previous frame"].iloc[i]
        if mse_val <= threshold:
            count += 1
            if count == n_frames:
                break
        else:
            break
    return count

################################################################################
# Modified Consolidation Function - Returns DataFrame Instead of Writing File
# ###############################################################################

def consolidate_with_asymmetric_thresholds_memory(
    df,  # Now takes DataFrame directly instead of file path
    threshold=0.2,
    title_change_threshold=2,
    active_to_inactive_threshold=5,
    inactive_to_active_threshold=5,
    n_prelogin=5
):
    """
    Modified consolidation function that returns DataFrame instead of writing to file.
    This eliminates file I/O during parameter sweeps.
    """
    if df.empty:
        return pd.DataFrame(columns=["time_start", "Activity", "Active_seconds", "Inactive_seconds"])

    # Convert timestamps to integer seconds (make a copy to avoid modifying original)
    df = df.copy()
    df["time_stamp"] = pd.to_timedelta(df["time_stamp"]).dt.total_seconds().astype(int)

    # Handle prelogin block
    pre_count = get_prelogin_count(df, threshold=threshold, n_frames=n_prelogin)
    out_rows = []
    if pre_count > 0:
        start_time = df["time_stamp"].iloc[0]
        out_rows.append([start_time, "Other", 0, pre_count])

    # Process remaining frames
    df_main = df.iloc[pre_count:].reset_index(drop=True)
    if df_main.empty:
        return pd.DataFrame(out_rows, columns=["time_start", "Activity", "Active_seconds", "Inactive_seconds"])

    # Initialize variables
    prev_title = "Other"
    prev_state = inactive
    consecutive = 0
    active_time = 0
    inactive_time = 0
    start_time = df_main["time_stamp"].iloc[0]
    first_frame = True
    login_mode = False
    forced_login_done = False

    for i in range(len(df_main)):
        curr_time = df_main["time_stamp"].iloc[i]
        curr_title = get_task(i, df_main)
        curr_mse = df_main["%change from previous frame"].iloc[i]
        curr_state = get_state_from_mse(curr_mse, threshold=threshold)

        # Handle first frame
        if first_frame:
            if curr_state == active:
                prev_title = "Login"
                login_mode = True
                forced_login_done = True
            else:
                prev_title = curr_title if curr_title != "Other" else "Other"
                login_mode = False
            consecutive = 1
            prev_state = curr_state
            start_time = curr_time
            first_frame = False
            continue

        # Force first active frame to be Login
        if (not login_mode) and (curr_state == active) and (not forced_login_done):
            if consecutive > 0:
                if prev_state == active:
                    active_time += consecutive
                else:
                    inactive_time += consecutive
                out_rows.append([start_time, prev_title, active_time, inactive_time])
            forced_login_done = True
            login_mode = True
            prev_title = "Login"
            start_time = curr_time
            consecutive = 1
            active_time = 0
            inactive_time = 0
            prev_state = curr_state
            continue

        # Handle login mode transitions
        if login_mode:
            if curr_title != "Other":
                if consecutive > 0:
                    if prev_state == active:
                        active_time += consecutive
                    else:
                        inactive_time += consecutive
                    out_rows.append([start_time, "Login", active_time, inactive_time])
                login_mode = False
                prev_title = curr_title
                start_time = curr_time
                consecutive = 1
                active_time = 0
                inactive_time = 0
                prev_state = curr_state
                continue
            else:
                curr_title = "Login"

        # Handle task changes
        task_change = False
        if curr_title != prev_title:
            if prev_title == "Login" or curr_title == "Login":
                if consecutive > 0:
                    if prev_state == active:
                        active_time += consecutive
                    else:
                        inactive_time += consecutive
                    out_rows.append([start_time, prev_title, active_time, inactive_time])
                task_change = True
                active_time, inactive_time = 0, 0
                start_time = curr_time
                consecutive = 1
                if curr_state == active and curr_title == "Login":
                    login_mode = True
                    prev_title = "Login"
                else:
                    login_mode = False
                    prev_title = curr_title
            else:
                if enough_rows(i, df_main, title_change_threshold):
                    if next_frames_titles_same(i, df_main, title_change_threshold):
                        if consecutive > 0:
                            if prev_state == active:
                                active_time += consecutive
                            else:
                                inactive_time += consecutive
                            out_rows.append([start_time, prev_title, active_time, inactive_time])
                        task_change = True
                        active_time, inactive_time = 0, 0
                        start_time = curr_time
                        consecutive = 1
                    else:
                        curr_title = prev_title

        if not task_change:
            # Check for active/inactive state changes within same task
            if curr_state != prev_state:
                # Use different thresholds based on transition direction
                threshold_to_use = active_to_inactive_threshold if prev_state == active else inactive_to_active_threshold
                
                if enough_rows(i, df_main, threshold_to_use):
                    next_states = df_main["%change from previous frame"].iloc[i : i + threshold_to_use]
                    all_inactive = all(ns <= threshold for ns in next_states)
                    all_active = all(ns > threshold for ns in next_states)
                    if all_inactive or all_active:
                        if prev_state == active:
                            active_time += consecutive
                        else:
                            inactive_time += consecutive
                        consecutive = 1
                    else:
                        curr_state = prev_state
                        consecutive += 1
                else:
                    curr_state = prev_state
                    consecutive += 1
            else:
                consecutive += 1

        prev_title = curr_title
        prev_state = curr_state

    # Finalize the last block
    if prev_state == active:
        active_time += consecutive
    else:
        inactive_time += consecutive
    out_rows.append([start_time, prev_title, active_time, inactive_time])
    out_rows.append([curr_time, 'END', 0, 0])

    # Return the consolidated DataFrame
    return pd.DataFrame(out_rows, columns=["time_start", "Activity", "Active_seconds", "Inactive_seconds"])

################################################################################
# Evaluation Functions (unchanged)
# ###############################################################################

def expand_to_seconds(df):
    """
    Expands a dataframe with segments to a second-by-second timeline with Activity and State.
    Handles different column naming conventions.
    """
    # Normalize column names (different versions use different names)
    if 'title' in df.columns and 'Activity' not in df.columns:
        df = df.rename(columns={'title': 'Activity'})
    if 'time_active' in df.columns and 'Active_seconds' not in df.columns:
        df = df.rename(columns={'time_active': 'Active_seconds'})
    if 'time_inactive' in df.columns and 'Inactive_seconds' not in df.columns:
        df = df.rename(columns={'time_inactive': 'Inactive_seconds'})
    
    # Get total duration
    if df.empty:
        return pd.DataFrame(columns=['second', 'Activity', 'State'])
    
    total_duration = 0
    for _, row in df.iterrows():
        # Ensure all values are numeric and handle any NaN values
        time_start = float(row['time_start']) if not pd.isna(row['time_start']) else 0
        active_secs = float(row['Active_seconds']) if not pd.isna(row['Active_seconds']) else 0 
        inactive_secs = float(row['Inactive_seconds']) if not pd.isna(row['Inactive_seconds']) else 0
        
        end_time = time_start + active_secs + inactive_secs
        if end_time > total_duration:
            total_duration = end_time
    
    # Create a second-by-second dataframe - ensure total_duration is an integer
    total_duration_int = int(np.ceil(total_duration))
    seconds_df = pd.DataFrame({'second': range(total_duration_int)})
    seconds_df['Activity'] = None
    seconds_df['State'] = None
    
    # Fill in activity and state for each second
    for _, row in df.iterrows():
        # Ensure all values are numeric
        start = int(row['time_start']) if not pd.isna(row['time_start']) else 0
        activity = row['Activity']
        active_secs = int(row['Active_seconds']) if not pd.isna(row['Active_seconds']) else 0
        inactive_secs = int(row['Inactive_seconds']) if not pd.isna(row['Inactive_seconds']) else 0
        
        # Active seconds
        active_end = min(start + active_secs, len(seconds_df))
        active_range = range(start, active_end)
        seconds_df.loc[seconds_df['second'].isin(active_range), 'Activity'] = activity
        seconds_df.loc[seconds_df['second'].isin(active_range), 'State'] = 'Active'
        
        # Inactive seconds
        inactive_start = start + active_secs
        inactive_end = min(inactive_start + inactive_secs, len(seconds_df))
        inactive_range = range(inactive_start, inactive_end)
        seconds_df.loc[seconds_df['second'].isin(inactive_range), 'Activity'] = activity
        seconds_df.loc[seconds_df['second'].isin(inactive_range), 'State'] = 'Inactive'
    
    return seconds_df

# Modified evaluation function to include signed error
def evaluate_prediction_memory(pred_df, gt_df):
    """Evaluate prediction DataFrame against ground truth DataFrame (both in memory)."""
    
    # Skip 'END' rows if they exist
    pred_df = pred_df[pred_df['Activity'] != 'END'].copy() if 'Activity' in pred_df.columns else pred_df
    gt_df = gt_df[gt_df['Activity'] != 'END'].copy() if 'Activity' in gt_df.columns else gt_df
    
    # Expand to second-by-second
    try:
        pred_sec = expand_to_seconds(pred_df)
        gt_sec = expand_to_seconds(gt_df)
    except Exception as e:
        print(f"[ERROR] Failed to expand to seconds: {str(e)}")
        return None
    
    # Ensure same length
    min_len = min(len(pred_sec), len(gt_sec))
    pred_sec = pred_sec.iloc[:min_len].copy()
    gt_sec = gt_sec.iloc[:min_len].copy()
    
    # Drop rows where either Activity or State is None
    mask = (~pred_sec['Activity'].isna()) & (~gt_sec['Activity'].isna()) & \
           (~pred_sec['State'].isna()) & (~gt_sec['State'].isna())
    pred_sec = pred_sec[mask].copy()
    gt_sec = gt_sec[mask].copy()
    
    if len(pred_sec) == 0:
        print("[ERROR] No valid data to compare after filtering")
        return None
    
    # Calculate activity accuracy
    activity_accuracy = accuracy_score(gt_sec['Activity'], pred_sec['Activity'])
    
    # Calculate state accuracy
    state_accuracy = accuracy_score(gt_sec['State'], pred_sec['State'])
    
    # Calculate combined accuracy
    combined = gt_sec['Activity'] + '_' + gt_sec['State']
    pred_combined = pred_sec['Activity'] + '_' + pred_sec['State']
    combined_accuracy = accuracy_score(combined, pred_combined)
    
    # Calculate MAPE and MPE for active time
    active_gt = gt_df['Active_seconds'].sum()
    active_pred = pred_df['Active_seconds'].sum()
    
    if active_gt > 0:
        active_time_mape = abs(active_gt - active_pred) / active_gt * 100
        # NEW: Calculate signed percentage error
        active_time_mpe = (active_pred - active_gt) / active_gt * 100
    else:
        active_time_mape = 0
        active_time_mpe = 0
    
    # Count task changes
    gt_task_changes = sum(1 for i in range(1, len(gt_df)) if gt_df['Activity'].iloc[i] != gt_df['Activity'].iloc[i-1])
    pred_task_changes = sum(1 for i in range(1, len(pred_df)) if pred_df['Activity'].iloc[i] != pred_df['Activity'].iloc[i-1])
    
    # Task changes accuracy
    max_changes = max(gt_task_changes, pred_task_changes)
    task_changes_accuracy = abs(gt_task_changes - pred_task_changes) / max_changes if max_changes > 0 else 1.0
    
    return {
        'activity_accuracy': activity_accuracy,
        'state_accuracy': state_accuracy,
        'combined_accuracy': combined_accuracy,
        'active_time_mape': active_time_mape,
        'active_time_mpe': active_time_mpe,  # NEW: Signed error
        'gt_task_changes': gt_task_changes,
        'pred_task_changes': pred_task_changes,
        'task_changes_accuracy': task_changes_accuracy
    }



################################################################################
# Parallel Processing Functions
# ###############################################################################

def process_single_parameter_combination(args):
    """
    Worker function to process a single parameter combination across all files.
    This function will be called in parallel by multiprocessing.Pool.
    """
    ati, ita, file_data_list, threshold, title_change, prelogin = args
    
    results = []
    
    for file_info in file_data_list:
        input_df = file_info['input_df']
        gt_df = file_info['gt_df']
        base_name = file_info['base_name']
        
        try:
            # Run consolidation in memory
            pred_df = consolidate_with_asymmetric_thresholds_memory(
                input_df,
                threshold=threshold,
                title_change_threshold=title_change,
                active_to_inactive_threshold=ati,
                inactive_to_active_threshold=ita,
                n_prelogin=prelogin
            )
            
            # Evaluate in memory
            metrics = evaluate_prediction_memory(pred_df, gt_df)
            
            if metrics is not None:
                # Add metadata to results
                metrics.update({
                    'file': base_name,
                    'active_to_inactive_threshold': ati,
                    'inactive_to_active_threshold': ita
                })
                results.append(metrics)
            else:
                print(f"[WARNING] Failed to evaluate {base_name} with ATI={ati}, ITA={ita}")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {base_name} with ATI={ati}, ITA={ita}: {str(e)}")
            continue
    
    return results

def load_all_files(input_dir, ground_truth_dir):
    """
    Load all input and ground truth files into memory once.
    Returns a list of dictionaries containing the loaded data.
    """
    # Convert to absolute paths
    input_dir = os.path.abspath(input_dir)
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    
    # Find input CSVs
    input_csvs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.lower().endswith(".csv")]
    
    print(f"Found {len(input_csvs)} input CSV files")
    
    # Load valid file pairs
    file_data_list = []
    for input_csv in input_csvs:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        gt_file = f"{base_name}_MANUALLYLABELED.csv"
        gt_path = os.path.join(ground_truth_dir, gt_file)
        
        if os.path.exists(gt_path):
            try:
                # Load input file
                input_df = pd.read_csv(input_csv)
                if input_df.empty:
                    print(f"[WARNING] Empty input file: {input_csv}")
                    continue
                
                # Load ground truth file
                gt_df = pd.read_csv(gt_path)
                if gt_df.empty:
                    print(f"[WARNING] Empty ground truth file: {gt_path}")
                    continue
                
                file_data_list.append({
                    'input_df': input_df,
                    'gt_df': gt_df,
                    'base_name': base_name
                })
                
            except Exception as e:
                print(f"[ERROR] Failed to load files for {base_name}: {str(e)}")
                continue
        else:
            print(f"[WARNING] No matching ground truth file for {input_csv}")
    
    print(f"Successfully loaded {len(file_data_list)} valid input/ground truth pairs")
    return file_data_list

################################################################################
# Optimized Main Analysis Function
# ###############################################################################

def run_asymmetric_threshold_analysis_optimized(
    input_dir="inputs", 
    ground_truth_dir="ground_truth", 
    output_dir="results_asymmetric_optimized",
    n_processes=None
):
    """
    Optimized version of the asymmetric threshold analysis with:
    1. Parallel processing of parameter combinations
    2. Files loaded once and kept in memory
    3. No intermediate file writes during parameter sweep
    """
    # Set number of processes (default to CPU count - 1)
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"Using {n_processes} parallel processes")
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # OPTIMIZATION 1: Load all files once into memory
    print("Loading all files into memory...")
    file_data_list = load_all_files(input_dir, ground_truth_dir)
    
    if not file_data_list:
        print("[ERROR] No valid input/ground truth pairs found")
        return None
    
    # Fixed parameters (as requested)
    threshold = 0.3
    title_change = 2
    prelogin = 3
    
    # Parameters to vary
    active_to_inactive_thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 45, 60]
    inactive_to_active_thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 45, 60]
    
    # Generate parameter combinations
    params_grid = list(product(active_to_inactive_thresholds, inactive_to_active_thresholds))
    print(f"Testing {len(params_grid)} parameter combinations")
    
    # OPTIMIZATION 2: Prepare arguments for parallel processing
    # Each worker gets: (ati, ita, file_data_list, threshold, title_change, prelogin)
    worker_args = [(ati, ita, file_data_list, threshold, title_change, prelogin) 
                   for ati, ita in params_grid]
    
    # OPTIMIZATION 3: Parallel processing of parameter combinations
    print("Starting parallel processing...")
    all_results = []
    
    with mp.Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        for param_results in tqdm(pool.imap(process_single_parameter_combination, worker_args), 
                                  total=len(worker_args), 
                                  desc="Processing parameter combinations"):
            all_results.extend(param_results)
    
    print(f"Collected {len(all_results)} total results")
    
    # OPTIMIZATION 4: Write final results only (no intermediate files)
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, "asymmetric_results_optimized_2.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved final results to {results_path}")
        
        # Analyze results
        analyze_asymmetric_thresholds(results_df, output_dir)
        
        return results_df
    else:
        print("[ERROR] No results were generated.")
        return None

################################################################################
# Analysis Functions (unchanged from original)
# ###############################################################################

def analyze_asymmetric_thresholds(results_df, output_dir):
    """Analyze the asymmetric threshold results to determine which effect dominates."""
    # Calculate average metrics for each parameter combination
    param_metrics = results_df.groupby(['active_to_inactive_threshold', 'inactive_to_active_threshold'])['active_time_mape'].mean().reset_index()
    
    # Create heatmap of MAPE by threshold combinations
    plt.figure(figsize=(10, 8))
    pivot = param_metrics.pivot(index='active_to_inactive_threshold', 
                              columns='inactive_to_active_threshold', 
                              values='active_time_mape')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis_r')  # viridis_r because lower MAPE is better
    plt.title('Active Time MAPE (%) by Threshold Combinations')
    plt.xlabel('Inactive-to-Active Threshold (seconds)')
    plt.ylabel('Active-to-Inactive Threshold (seconds)')
    plt.savefig(os.path.join(output_dir, "asymmetric_heatmap_optimized.png"))
    plt.show()
    
    # Find optimal combination
    best_idx = param_metrics['active_time_mape'].idxmin()
    best_params = param_metrics.iloc[best_idx]
    
    print(f"Optimal threshold combination:")
    print(f"Active-to-Inactive: {best_params['active_to_inactive_threshold']} seconds")
    print(f"Inactive-to-Active: {best_params['inactive_to_active_threshold']} seconds")
    print(f"MAPE: {best_params['active_time_mape']:.2f}%")
    
    # Analyze effect of each threshold independently
    ati_effect = param_metrics.groupby('active_to_inactive_threshold')['active_time_mape'].mean().reset_index()
    ita_effect = param_metrics.groupby('inactive_to_active_threshold')['active_time_mape'].mean().reset_index()
    
    print("\nEffect of Active-to-Inactive Threshold on MAPE:")
    print(ati_effect)
    
    print("\nEffect of Inactive-to-Active Threshold on MAPE:")
    print(ita_effect)
    
    # Compare marginal effects
    ati_range = ati_effect['active_time_mape'].max() - ati_effect['active_time_mape'].min()
    ita_range = ita_effect['active_time_mape'].max() - ita_effect['active_time_mape'].min()
    
    print(f"\nMarginal effect range:")
    print(f"Active-to-Inactive: {ati_range:.1f}% MAPE change")
    print(f"Inactive-to-Active: {ita_range:.1f}% MAPE change")
    
    dominant_effect = "Active-to-Inactive" if ati_range > ita_range else "Inactive-to-Active"
    print(f"\nDominant effect: {dominant_effect} threshold")
    
    # Plot effects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Active-to-Inactive threshold effect
    ax1.plot(ati_effect['active_to_inactive_threshold'], ati_effect['active_time_mape'], marker='o', linewidth=2)
    ax1.set_title('Effect of Active-to-Inactive Threshold')
    ax1.set_xlabel('Threshold (seconds)')
    ax1.set_ylabel('Average MAPE (%)')
    ax1.grid(True)
    
    # Inactive-to-Active threshold effect
    ax2.plot(ita_effect['inactive_to_active_threshold'], ita_effect['active_time_mape'], marker='o', linewidth=2)
    ax2.set_title('Effect of Inactive-to-Active Threshold')
    ax2.set_xlabel('Threshold (seconds)')
    ax2.set_ylabel('Average MAPE (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_effects_optimized.png"))
    plt.show()
    
    # Save detailed results to CSV
    ati_effect.to_csv(os.path.join(output_dir, "active_to_inactive_effect_optimized.csv"), index=False)
    ita_effect.to_csv(os.path.join(output_dir, "inactive_to_active_effect_optimized.csv"), index=False)
    
    print("\n===== OPTIMIZED ANALYSIS SUMMARY =====")
    print("The study explored different thresholds for active-to-inactive vs inactive-to-active transitions.")
    print(f"Optimal combination: Active-to-Inactive = {best_params['active_to_inactive_threshold']}s, Inactive-to-Active = {best_params['inactive_to_active_threshold']}s")
    print(f"The {dominant_effect} threshold has a stronger impact on accuracy.")
    
    return {
        'optimal_params': {
            'active_to_inactive': best_params['active_to_inactive_threshold'],
            'inactive_to_active': best_params['inactive_to_active_threshold'],
            'mape': best_params['active_time_mape']
        },
        'dominant_effect': dominant_effect,
        'ati_range': ati_range,
        'ita_range': ita_range
    }

################################################################################
# Main Execution
# ###############################################################################

if __name__ == "__main__":
    # Run the optimized asymmetric threshold analysis
    print("Starting optimized asymmetric threshold analysis...")
    results_df = run_asymmetric_threshold_analysis_optimized()
    print(results_df)
    if results_df is not None:
        print(f"Analysis completed successfully with {len(results_df)} total results")
    else:
        print("Analysis failed to generate results")
