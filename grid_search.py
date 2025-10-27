#!/usr/bin/env python3
"""
Grid Search for EHR Activity Detection Parameters

This script performs a sensitivity analysis to find the optimal parameters
for the threshold of inactivity model.

Required directory structure:
- inputs/                   # Place model outputs here
- ground_truth/             # Place manually labeled files here

The script will save results to:
- results/                  # Main output directory
- results/parameter_search_results.csv  # Final results CSV
"""

import os
import sys
import shutil
import time
import argparse
import pandas as pd
import numpy as np
from itertools import product
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import difflib
import glob
import warnings
from tqdm import tqdm
import yaml

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_context("notebook", font_scale=1.2)


################################################################################
# Task Dictionary from Original Code
################################################################################

task_dictionary = {
    "Chart_Review": [
        "Chart", "Review", "Chart Review", "Timeline", "IP Summary", "Web", "Resources", "Web Resources",
        "Intake/Output", "Intake", "Output", "Avatar", "Growth", "Growth Chart", "Infant Development", "Sidebar",
        "Sidebar Summary", "Synopsis", "BPA Review", "Episodes of Care", "Health Maintenance", "Immunizations",
        "MAR", "Medications", "History", "Implants", "Review Flowsheets", "Graphs", "Images", "Media", "Manager",
        "Media Manager", "Document List", "Scans", "View Scans", "Document List", "Request Outside", "Doc Flowsheets",
        "Gender and Sexual Orientation", "Problem", "Problem List", "Rounding", "Annotated Images", "Enter/Edit Results",
        "Communications", "Additional Tools", "Clinical Calculator", "References", "Acquire", "HIM", "List Disclosures",
        "Quick Disclosure", "Endo/Diab", "ROP SmartForm", "Sedation", "Antimicrobial Stewards", "Asthma Action Plan",
        "Connect to Video", "Education", "Newborn ROP Screen", "Occluded Vessels", "Patient Goals", "Prep for Surgery",
        "Restraint Note", "SureScripts", "Univ Protocol", "WARM Scoring", "CHOIR", "Ehlers-Danlos Checklist", "NM Tools",
        "Research Studies", "Suspected child abuse", "Asthma Pathway", "Asthma", "Autopsy", "Bass Center", "Behavior",
        "Behavior Plan", "Dialysis", "Neonatal MDR"
    ],
    "In_Basket": ["In Basket", "Send Message"],
    "Login": ["Login"],
    "Navigation": ["Inpatient Provider Dashboard", "Provider", "Dashboard", "Welcome", "Patient Lists"],
    "Note_Entry": ["Procedures", "Charge", "Note Entry", "Charge Capture", "Edit Note", "Charge Capture Goals of Care"],
    "Note_Review": ["Notes", "My Notes Settings", "Note Review"],
    "Order_Entry": ["Orders", "ADT Navigators", "Order", "Order Review", "Finish Order Reconciliation", "Order Sets",
                    "Admit", "Discharge", "Order Entry"],
    "Other": ["NaN"],
    "Results_Review": ["Results", "Review", "Results Review"]
}

# Flags to indicate active/inactive state
active = True
inactive = False


################################################################################
# Consolidation Functions
################################################################################

def get_task_dictionary_match(text: str) -> str:
    """
    Fuzzy-match a text against task_dictionary keys.

    Parameters:
    -----------
    text : str
        Text to match against task dictionary

    Returns:
    --------
    str : Best matching task category
    """
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

    Parameters:
    -----------
    i : int
        Row index
    df : DataFrame
        Input dataframe

    Returns:
    --------
    str : Determined task name
    """
    page_title = str(df["page_title"].iloc[i])
    sidebar = str(df["sidebar"].iloc[i])
    matched_pt = get_task_dictionary_match(page_title)
    matched_sb = get_task_dictionary_match(sidebar)
    if matched_pt == "Other" and matched_sb != "Other":
        return matched_sb
    return matched_pt


def get_state_from_mse(mse_val, threshold=0.1) -> bool:
    """
    Return True (active) if MSE exceeds threshold.

    Parameters:
    -----------
    mse_val : float
        MSE value
    threshold : float
        Threshold value

    Returns:
    --------
    bool : True if active, False if inactive
    """
    return mse_val > threshold


def next_frames_titles_same(i, data, title_change_threshold):
    """
    Check if the next title_change_threshold rows have the same page_title.

    Parameters:
    -----------
    i : int
        Current row index
    data : DataFrame
        Input dataframe
    title_change_threshold : int
        Number of frames to check

    Returns:
    --------
    bool : True if all titles are the same
    """
    subset = set(data["page_title"].iloc[i: i + title_change_threshold])
    return len(subset) == 1


def enough_rows(i, data, threshold):
    """
    Return True if there are at least 'threshold' rows starting from index i.

    Parameters:
    -----------
    i : int
        Starting index
    data : DataFrame
        Input dataframe
    threshold : int
        Number of rows needed

    Returns:
    --------
    bool : True if enough rows exist
    """
    return i + threshold <= len(data)


def get_prelogin_count(df, threshold=0.1, n_frames=10) -> int:
    """
    Count consecutive frames from the start where %change <= threshold.
    Stop at n_frames or on the first frame with MSE > threshold.
    Returns the number of frames to force as 'Other' in the prelogin block.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    threshold : float
        MSE threshold
    n_frames : int
        Maximum number of frames to check

    Returns:
    --------
    int : Number of prelogin frames
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


def consolidate(
    input_csv,
    threshold=0.1,
    title_change_threshold=3,
    state_change_threshold=3,
    n_prelogin=10
):
    """
    Consolidate video frames into activity blocks.

    1) Force the first n_prelogin frames with MSE <= threshold as one 'Other' block.
    2) Then process the remaining frames with modified logic:
         - The very first time an active frame is encountered, force its block to be 'Login'.
         - Once that login block is started, it will continue as 'Login' as long as the
           identified title is "Other". When a new identified title (not "Other") occurs,
           finalize the login block and start a new block.
         - If the next block is also 'Login', merge it.
    3) Save final results in ./consolidated/<basename>_parsed.csv, ending with an 'END' row.

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    threshold : float
        MSE threshold for activity detection
    title_change_threshold : int
        Number of frames required to confirm a title change
    state_change_threshold : int
        Number of frames required to confirm a state change
    n_prelogin : int
        Number of prelogin frames to force as 'Other'
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"[WARNING] No data in {input_csv}. Cannot consolidate.")
        return

    # Convert "0:00:09" timestamps to integer seconds.
    df["time_stamp"] = pd.to_timedelta(df["time_stamp"]).dt.total_seconds().astype(int)

    # A) Prelogin Block: Force up to n_prelogin frames (with MSE <= threshold) as "Other"
    pre_count = get_prelogin_count(df, threshold=threshold, n_frames=n_prelogin)
    out_rows = []
    if pre_count > 0:
        start_time = df["time_stamp"].iloc[0]
        # Label the prelogin block as "Other" with 0 active time and pre_count frames of inactivity.
        out_rows.append([start_time, "Other", 0, pre_count])

    # Slice remaining frames.
    df_main = df.iloc[pre_count:].reset_index(drop=True)
    if df_main.empty:
        final_df = pd.DataFrame(out_rows, columns=["time_start", "title", "time_active", "time_inactive"])
        _save_consolidated_csv(input_csv, final_df)
        return

    # B) Main Consolidation Logic with First Active Frame Forced as Login
    prev_title = "Other"  # Initialize as "Other"
    prev_state = inactive
    consecutive = 0
    active_time = 0
    inactive_time = 0
    start_time = df_main["time_stamp"].iloc[0]
    first_frame = True
    login_mode = False      # Indicates if we are in a Login block.
    forced_login_done = False  # To force login only for the very first active instance.

    for i in range(len(df_main)):
        curr_time = df_main["time_stamp"].iloc[i]
        curr_title = get_task(i, df_main)
        curr_mse = df_main["%change from previous frame"].iloc[i]
        curr_state = get_state_from_mse(curr_mse, threshold=threshold)

        # --- Handle the very first frame ---
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

        # --- Force the first active frame (if not already forced) to be Login ---
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

        # --- If in login mode, check for transition out ---
        if login_mode:
            # If the identified title is not "Other", that signals a new block.
            if curr_title != "Other":
                # Finalize the current Login block.
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
                # While in login mode and title is "Other", force it to remain "Login".
                curr_title = "Login"

        # --- Normal task-change handling ---
        task_change = False
        if curr_title != prev_title:
            # If either previous or current title is "Login", finalize immediately.
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
                # Set login_mode based on current frame.
                if curr_state == active and curr_title == "Login":
                    login_mode = True
                    prev_title = "Login"
                else:
                    login_mode = False
                    prev_title = curr_title
            else:
                # Use standard title-change threshold logic.
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
            # Check for active/inactive state changes within the same block.
            if curr_state != prev_state:
                if enough_rows(i, df_main, state_change_threshold):
                    next_states = df_main["%change from previous frame"].iloc[i: i + state_change_threshold]
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

    # Finalize the last block.
    if prev_state == active:
        active_time += consecutive
    else:
        inactive_time += consecutive
    out_rows.append([start_time, prev_title, active_time, inactive_time])
    out_rows.append([curr_time, 'END', 0, 0])

    # C) Save the Consolidated CSV
    final_df = pd.DataFrame(out_rows, columns=["time_start", "Activity", "Active_seconds", "Inactive_seconds"])
    _save_consolidated_csv(input_csv, final_df)


def _save_consolidated_csv(input_csv, out_df):
    """
    Helper to save final CSV in ./consolidated/ named <basename>_parsed.csv.

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    out_df : DataFrame
        Consolidated dataframe to save
    """
    base_dir = os.getcwd()
    consolidated_dir = os.path.join(base_dir, "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv = os.path.join(consolidated_dir, f"{base_name}_parsed.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Consolidation done => {out_csv}")


################################################################################
# Evaluation Functions
################################################################################

def expand_to_seconds(df):
    """
    Expands a dataframe with segments to a second-by-second timeline with Activity and State.
    Handles different column naming conventions.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe with consolidated activities

    Returns:
    --------
    DataFrame : Second-by-second timeline
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


def run_consolidation(input_csv, output_dir, params):
    """
    Run consolidation with given parameters and return path to output.

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    output_dir : str
        Base output directory
    params : tuple
        (threshold, title_change, state_change, prelogin)

    Returns:
    --------
    str : Path to consolidated output file, or None if failed
    """
    threshold, title_change, state_change, prelogin = params

    # Ensure we're working with absolute paths
    input_csv = os.path.abspath(input_csv)
    output_dir = os.path.abspath(output_dir)

    # Create param-specific output directory
    param_str = f"t{threshold}_tc{title_change}_sc{state_change}_pl{prelogin}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)

    # Save current directory
    orig_dir = os.getcwd()

    try:
        # Change to parameter directory so consolidated output goes there
        os.chdir(param_dir)

        # Create consolidated directory if it doesn't exist
        os.makedirs("consolidated", exist_ok=True)

        # Create temporary inputs directory and copy the input file there
        os.makedirs("inputs", exist_ok=True)
        input_copy = os.path.join(param_dir, "inputs", os.path.basename(input_csv))

        try:
            shutil.copy2(input_csv, input_copy)
        except Exception:
            # If copy fails, return None
            os.chdir(orig_dir)
            return None

        if not os.path.exists(input_copy):
            os.chdir(orig_dir)
            return None

        # Run consolidation (this will save to ./consolidated/<basename>_parsed.csv)
        try:
            consolidate(input_copy, threshold=threshold,
                        title_change_threshold=title_change,
                        state_change_threshold=state_change,
                        n_prelogin=prelogin)
        except Exception:
            # If consolidation fails, return None
            os.chdir(orig_dir)
            return None

        # Get the output path
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        out_path = os.path.join(param_dir, "consolidated", f"{base_name}_parsed.csv")

        if not os.path.exists(out_path):
            os.chdir(orig_dir)
            return None

        return out_path

    finally:
        # Change back to original directory
        os.chdir(orig_dir)


def evaluate_prediction(pred_path, gt_file, ground_truth_dir):
    """
    Evaluate prediction against ground truth.

    Parameters:
    -----------
    pred_path : str
        Path to prediction CSV file
    gt_file : str
        Name of ground truth file
    ground_truth_dir : str
        Directory containing ground truth files

    Returns:
    --------
    dict : Evaluation metrics, or None if evaluation failed
    """
    # Ensure absolute paths
    pred_path = os.path.abspath(pred_path)
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    gt_path = os.path.join(ground_truth_dir, gt_file)

    # Load prediction
    try:
        if not os.path.exists(pred_path):
            return None
        pred_df = pd.read_csv(pred_path)
    except Exception:
        return None

    # Load ground truth
    try:
        if not os.path.exists(gt_path):
            return None
        gt_df = pd.read_csv(gt_path)
    except Exception:
        return None

    # Skip 'END' rows if they exist
    pred_df = pred_df[pred_df['Activity'] != 'END'].copy() if 'Activity' in pred_df.columns else pred_df
    gt_df = gt_df[gt_df['Activity'] != 'END'].copy() if 'Activity' in gt_df.columns else gt_df

    # Expand to second-by-second
    try:
        pred_sec = expand_to_seconds(pred_df)
        gt_sec = expand_to_seconds(gt_df)
    except Exception:
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
        return None

    # Calculate activity accuracy
    activity_accuracy = accuracy_score(gt_sec['Activity'], pred_sec['Activity'])

    # Calculate state accuracy
    state_accuracy = accuracy_score(gt_sec['State'], pred_sec['State'])

    # Calculate combined accuracy
    combined = gt_sec['Activity'] + '_' + gt_sec['State']
    pred_combined = pred_sec['Activity'] + '_' + pred_sec['State']
    combined_accuracy = accuracy_score(combined, pred_combined)

    return {
        'activity_accuracy': activity_accuracy,
        'state_accuracy': state_accuracy,
        'combined_accuracy': combined_accuracy,
    }


################################################################################
# Grid Search Functions
################################################################################

def grid_search(input_csvs, ground_truth_dir, output_dir, params_grid):
    """
    Perform grid search over parameter space.

    Parameters:
    -----------
    input_csvs : list
        List of paths to input CSV files
    ground_truth_dir : str
        Directory containing ground truth files
    output_dir : str
        Base output directory
    params_grid : list
        List of parameter tuples to test

    Returns:
    --------
    DataFrame : Results dataframe with all evaluations
    """
    all_results = []

    # Convert paths to absolute paths to avoid directory issues
    input_csvs = [os.path.abspath(csv) for csv in input_csvs]
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    output_dir = os.path.abspath(output_dir)

    # Use tqdm for the outer loop
    pbar_params = tqdm(params_grid, desc="Parameter combinations", leave=True)

    # For each parameter combination
    for params in pbar_params:
        t, tc, sc, pl = params
        pbar_params.set_description(f"Testing t={t}, tc={tc}, sc={sc}, pl={pl}")

        file_metrics = []

        # Use tqdm for the inner loop but hide it by default
        pbar_files = tqdm(input_csvs, desc="Processing files", leave=False)

        # Test on each input file
        for input_csv in pbar_files:
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(input_csv))[0]
            pbar_files.set_description(f"Processing {base_name}")

            # Find matching ground truth file
            gt_file = f"{base_name}_MANUALLYLABELED.csv"
            gt_path = os.path.join(ground_truth_dir, gt_file)

            if not os.path.exists(gt_path):
                continue

            # Run consolidation
            try:
                pred_path = run_consolidation(input_csv, output_dir, params)

                if not pred_path or not os.path.exists(pred_path):
                    continue

                # Evaluate against ground truth
                metrics = evaluate_prediction(pred_path, gt_file, ground_truth_dir)

                if metrics is None:
                    continue

                metrics['file'] = base_name
                file_metrics.append(metrics)

            except Exception:
                # Swallow exceptions to avoid breaking the loop
                continue

        # Calculate average metrics across all files
        if file_metrics:
            # Add to results
            for fm in file_metrics:
                all_results.append({
                    **fm,
                    'threshold': t,
                    'title_change_threshold': tc,
                    'state_change_threshold': sc,
                    'n_prelogin': pl
                })

    # Convert to dataframe
    results_df = pd.DataFrame(all_results)

    return results_df


def analyze_results(results_df, output_dir):
    """
    Analyze results and identify best parameters.

    Parameters:
    -----------
    results_df : DataFrame
        Results from grid search
    output_dir : str
        Output directory for saving plots

    Returns:
    --------
    dict : Best parameter values
    """
    # Group by parameter combinations
    param_metrics = results_df.groupby([
        'threshold', 'title_change_threshold',
        'state_change_threshold', 'n_prelogin'
    ])[['activity_accuracy', 'state_accuracy', 'combined_accuracy']].mean().reset_index()

    # Find best parameters
    best_row = param_metrics.loc[param_metrics['combined_accuracy'].idxmax()]

    best_params = {
        'threshold': best_row['threshold'],
        'title_change_threshold': int(best_row['title_change_threshold']),
        'state_change_threshold': int(best_row['state_change_threshold']),
        'n_prelogin': int(best_row['n_prelogin'])
    }

    print("\n===== BEST PARAMETERS =====")
    print(f"MSE Threshold: {best_params['threshold']}")
    print(f"Task Change Threshold: {best_params['title_change_threshold']}")
    print(f"State Change Threshold: {best_params['state_change_threshold']}")
    print(f"Prelogin Frames: {best_params['n_prelogin']}")
    print(f"Combined Accuracy: {best_row['combined_accuracy']:.4f}")

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot parameter effects
    plot_parameter_effects(param_metrics, plots_dir)

    return best_params


def plot_parameter_effects(param_metrics, plots_dir):
    """
    Plot the effect of each parameter on the combined accuracy.

    Parameters:
    -----------
    param_metrics : DataFrame
        Aggregated metrics by parameter combination
    plots_dir : str
        Directory to save plots
    """
    # Plot effect of threshold
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='threshold', y='combined_accuracy', data=param_metrics)
    plt.title('Effect of MSE Threshold on Combined Accuracy')
    plt.xlabel('MSE Threshold')
    plt.ylabel('Combined Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'threshold_effect.png'))
    plt.close()

    # Plot effect of title_change_threshold
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='title_change_threshold', y='combined_accuracy', data=param_metrics)
    plt.title('Effect of Task Change Threshold on Combined Accuracy')
    plt.xlabel('Task Change Threshold')
    plt.ylabel('Combined Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task_change_effect.png'))
    plt.close()

    # Plot effect of state_change_threshold
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='state_change_threshold', y='combined_accuracy', data=param_metrics)
    plt.title('Effect of State Change Threshold on Combined Accuracy')
    plt.xlabel('State Change Threshold')
    plt.ylabel('Combined Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'state_change_effect.png'))
    plt.close()

    # Plot effect of n_prelogin
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_prelogin', y='combined_accuracy', data=param_metrics)
    plt.title('Effect of Prelogin Frames on Combined Accuracy')
    plt.xlabel('Prelogin Frames')
    plt.ylabel('Combined Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prelogin_effect.png'))
    plt.close()

    print(f"Plots saved to {plots_dir}")


def create_final_outputs(input_csvs, output_dir, best_params):
    """
    Create final consolidated outputs using best parameters.

    Parameters:
    -----------
    input_csvs : list
        List of input CSV files
    output_dir : str
        Output directory
    best_params : dict
        Best parameter values
    """
    print("\n===== CREATING FINAL OUTPUTS =====")

    params = (
        best_params['threshold'],
        best_params['title_change_threshold'],
        best_params['state_change_threshold'],
        best_params['n_prelogin']
    )

    param_str = f"t{params[0]}_tc{params[1]}_sc{params[2]}_pl{params[3]}"
    source_dir = os.path.join(output_dir, param_str, "consolidated")

    if not os.path.exists(source_dir):
        print(f"Warning: Source directory not found: {source_dir}")
        return

    # Copy final outputs to main results directory
    final_dir = os.path.join(output_dir, "final_outputs")
    os.makedirs(final_dir, exist_ok=True)

    for input_csv in input_csvs:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        source_file = os.path.join(source_dir, f"{base_name}_parsed.csv")
        dest_file = os.path.join(final_dir, f"{base_name}_final.csv")

        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"Created {dest_file}")


def load_config(config_path):
    """
    Load configuration from YAML file.

    Parameters:
    -----------
    config_path : str
        Path to config.yaml file

    Returns:
    --------
    dict : Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading config file: {e}")
        return {}


def run_grid_search(
    input_dir="inputs",
    ground_truth_dir="ground_truth",
    output_dir="results",
    mse_thresholds=None,
    task_change_thresholds=None,
    state_change_thresholds=None,
    prelogin_frames=None,
    config=None
):
    """
    Run the complete grid search analysis.

    Parameters:
    -----------
    input_dir : str
        Directory containing input CSV files
    ground_truth_dir : str
        Directory containing ground truth files
    output_dir : str
        Directory to save results
    mse_thresholds : list
        List of MSE threshold values to test (default: Paper Table 1 values)
    task_change_thresholds : list
        List of task change threshold values to test (default: Paper Table 1 values)
    state_change_thresholds : list
        List of state change threshold values to test (threshold of inactivity from Paper Table 1)
    prelogin_frames : list
        List of prelogin frame values to test
    config : dict
        Configuration dictionary from config.yaml

    Returns:
    --------
    dict : Results including best parameters and full results dataframe
    """
    # Use Paper Table 1 values if not specified
    if mse_thresholds is None:
        mse_thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    if task_change_thresholds is None:
        task_change_thresholds = [1, 2, 3, 4, 5]

    if state_change_thresholds is None:
        # threshold_of_inactivity from Paper Table 1
        state_change_thresholds = [1, 2, 3, 4, 5, 6, 7]

    if prelogin_frames is None:
        prelogin_frames = [5, 10, 15]

    # Override with config values if provided
    if config:
        if 'mse_thresholds' in config:
            mse_thresholds = config['mse_thresholds']
        if 'task_change_thresholds' in config:
            task_change_thresholds = config['task_change_thresholds']
        if 'state_change_thresholds' in config:
            state_change_thresholds = config['state_change_thresholds']
        if 'prelogin_frames' in config:
            prelogin_frames = config['prelogin_frames']
        if 'input_dir' in config:
            input_dir = config['input_dir']
        if 'ground_truth_dir' in config:
            ground_truth_dir = config['ground_truth_dir']
        if 'output_dir' in config:
            output_dir = config['output_dir']

    # Convert to absolute paths
    input_dir = os.path.abspath(input_dir)
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    output_dir = os.path.abspath(output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check for valid directories
    for dir_name, dir_path in [("input", input_dir), ("ground truth", ground_truth_dir)]:
        if not os.path.exists(dir_path):
            print(f"Error: {dir_name} directory does not exist: {dir_path}")
            return None
        if len(os.listdir(dir_path)) == 0:
            print(f"Warning: {dir_name} directory is empty: {dir_path}")

    # Find input CSVs and ground truth files
    input_csvs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(".csv")]
    gt_files = [f for f in os.listdir(ground_truth_dir)
                if f.endswith("_MANUALLYLABELED.csv")]

    print(f"Found {len(input_csvs)} input CSV files and {len(gt_files)} ground truth files")

    if not input_csvs:
        print("Error: No input CSV files found.")
        return None

    if not gt_files:
        print("Error: No ground truth files found.")
        return None

    # Generate parameter grid
    params_grid = list(product(mse_thresholds, task_change_thresholds,
                               state_change_thresholds, prelogin_frames))

    print(f"\n===== PARAMETER RANGES (from Paper Table 1) =====")
    print(f"MSE Thresholds: {mse_thresholds}")
    print(f"Task Change Thresholds: {task_change_thresholds}")
    print(f"Threshold of Inactivity (State Change): {state_change_thresholds}")
    print(f"Prelogin Frames: {prelogin_frames}")
    print(f"\nTotal parameter combinations: {len(params_grid)}")

    # Run grid search
    print("\n===== STARTING GRID SEARCH =====")
    results_df = grid_search(input_csvs, ground_truth_dir, output_dir, params_grid)

    if results_df.empty:
        print("Error: No results generated.")
        return None

    # Save results
    results_path = os.path.join(output_dir, "parameter_search_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Analyze results
    best_params = analyze_results(results_df, output_dir)

    # Create final outputs
    create_final_outputs(input_csvs, output_dir, best_params)

    print("\n===== GRID SEARCH COMPLETE =====")

    return {
        'best_params': best_params,
        'results_df': results_df,
        'results_path': results_path
    }


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description='Grid search for EHR activity detection parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters (Paper Table 1 values)
  python grid_search.py

  # Run with custom config file
  python grid_search.py --config my_config.yaml

  # Run with custom directories
  python grid_search.py --input-dir data/inputs --ground-truth-dir data/gt --output-dir results

  # Run with specific parameter ranges
  python grid_search.py --mse-thresholds 0.1 0.2 0.3 --task-change-thresholds 2 3 4

Default parameter ranges (from Paper Table 1):
  - MSE Threshold: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  - Task Change Threshold: [1, 2, 3, 4, 5]
  - Threshold of Inactivity: [1, 2, 3, 4, 5, 6, 7]
  - Prelogin Frames: [5, 10, 15]
        """
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml file')
    parser.add_argument('--input-dir', type=str, default='inputs',
                        help='Directory containing input CSV files (default: inputs)')
    parser.add_argument('--ground-truth-dir', type=str, default='ground_truth',
                        help='Directory containing ground truth files (default: ground_truth)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--mse-thresholds', type=float, nargs='+', default=None,
                        help='MSE threshold values to test (default: Paper Table 1 values)')
    parser.add_argument('--task-change-thresholds', type=int, nargs='+', default=None,
                        help='Task change threshold values to test (default: Paper Table 1 values)')
    parser.add_argument('--state-change-thresholds', type=int, nargs='+', default=None,
                        help='State change (threshold of inactivity) values to test (default: Paper Table 1 values)')
    parser.add_argument('--prelogin-frames', type=int, nargs='+', default=None,
                        help='Prelogin frame values to test (default: [5, 10, 15])')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")

    # Run grid search
    result = run_grid_search(
        input_dir=args.input_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        mse_thresholds=args.mse_thresholds,
        task_change_thresholds=args.task_change_thresholds,
        state_change_thresholds=args.state_change_thresholds,
        prelogin_frames=args.prelogin_frames,
        config=config
    )

    if result:
        print("\nGrid search completed successfully!")
        print(f"Results saved to: {result['results_path']}")
        sys.exit(0)
    else:
        print("\nGrid search failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
