#!/usr/bin/env python3
"""
Bayesian Optimization for EHR Activity Detection Parameters

This script performs Bayesian optimization to find optimal parameters for the
threshold of inactivity model using two different objective functions:
1. Active time MAPE from ground truth
2. Task change sensitivity (transition detection F1 score)

Usage:
    python bayesian_optimization.py [options]

Example:
    python bayesian_optimization.py --input-dir inputs --gt-dir ground_truth --n-calls 50
    python bayesian_optimization.py --config config.yaml
"""

# Import required libraries
import os
import sys
import shutil
import pandas as pd
import numpy as np
from itertools import product
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import difflib
import glob
import warnings
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Additional imports for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_context("notebook", font_scale=1.2)

################################################################################
# Task Dictionary from Original Code
################################################################################

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
# Consolidation Functions from reconsolidate.ipynb
################################################################################

def get_task_dictionary_match(text: str) -> str:
    """
    Fuzzy-match a text against task_dictionary keys.

    Args:
        text: The text to match against task dictionary

    Returns:
        The best matching task key or "Other" if no good match found
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

    Args:
        i: Row index
        df: DataFrame containing page_title and sidebar columns

    Returns:
        Task name for the given row
    """
    page_title = str(df["page_title"].iloc[i])
    sidebar    = str(df["sidebar"].iloc[i])
    matched_pt = get_task_dictionary_match(page_title)
    matched_sb = get_task_dictionary_match(sidebar)
    if matched_pt == "Other" and matched_sb != "Other":
        return matched_sb
    return matched_pt


def get_state_from_mse(mse_val, threshold=0.1) -> bool:
    """
    Return True (active) if MSE exceeds threshold.

    Args:
        mse_val: MSE value to check
        threshold: Threshold for activity detection

    Returns:
        True if active, False if inactive
    """
    return mse_val > threshold


def next_frames_titles_same(i, data, title_change_threshold):
    """
    Check if the next title_change_threshold rows have the same page_title.

    Args:
        i: Starting row index
        data: DataFrame to check
        title_change_threshold: Number of frames to check

    Returns:
        True if all titles are the same
    """
    subset = set(data["page_title"].iloc[i : i + title_change_threshold])
    return len(subset) == 1


def enough_rows(i, data, threshold):
    """
    Return True if there are at least 'threshold' rows starting from index i.

    Args:
        i: Starting row index
        data: DataFrame to check
        threshold: Minimum number of rows required

    Returns:
        True if enough rows exist
    """
    return i + threshold <= len(data)


def get_prelogin_count(df, threshold=0.1, n_frames=10) -> int:
    """
    Count consecutive frames from the start where %change <= threshold.

    Args:
        df: DataFrame to analyze
        threshold: MSE threshold
        n_frames: Maximum number of frames to check

    Returns:
        Number of frames to force as 'Other' in the prelogin block
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
    title_change_threshold=5,
    state_change_threshold=5,
    n_prelogin=10
):
    """
    Consolidate EHR activity data into activity blocks.

    This function processes raw EHR activity data and consolidates it into
    coherent activity blocks with active and inactive time periods.

    Args:
        input_csv: Path to input CSV file
        threshold: MSE threshold for activity detection
        title_change_threshold: Number of frames to confirm title change
        state_change_threshold: Number of frames to confirm state change
        n_prelogin: Number of frames to force as prelogin block
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"[WARNING] No data in {input_csv}. Cannot consolidate.")
        return

    # Convert "0:00:09" timestamps to integer seconds.
    df["time_stamp"] = pd.to_timedelta(df["time_stamp"]).dt.total_seconds().astype(int)

    # ---------------------------------------------------------------------
    # A) Prelogin Block: Force up to n_prelogin frames (with MSE <= threshold) as "Other"
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # B) Main Consolidation Logic with First Active Frame Forced as Login
    # ---------------------------------------------------------------------
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
        curr_time  = df_main["time_stamp"].iloc[i]
        curr_title = get_task(i, df_main)
        curr_mse   = df_main["%change from previous frame"].iloc[i]
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
                    next_states = df_main["%change from previous frame"].iloc[i : i + state_change_threshold]
                    all_inactive = all(ns <= threshold for ns in next_states)
                    all_active   = all(ns > threshold  for ns in next_states)
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

    # ---------------------------------------------------------------------
    # C) Save the Consolidated CSV
    # ---------------------------------------------------------------------
    final_df = pd.DataFrame(out_rows, columns=["time_start", "Activity", "Active_seconds", "Inactive_seconds"])
    _save_consolidated_csv(input_csv, final_df)


def _save_consolidated_csv(input_csv, out_df, verbose=False):
    """
    Helper to save final CSV in ./consolidated/ named <basename>_parsed.csv.

    Args:
        input_csv: Path to input CSV file
        out_df: DataFrame to save
        verbose: Whether to print output message
    """
    base_dir = os.getcwd()
    consolidated_dir = os.path.join(base_dir, "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv = os.path.join(consolidated_dir, f"{base_name}_parsed.csv")
    out_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"[INFO] Consolidation done => {out_csv}")


################################################################################
# Functions for Second-by-Second Timeline and Evaluation
################################################################################

def expand_to_seconds(df):
    """
    Expands a dataframe with segments to a second-by-second timeline with Activity and State.

    Args:
        df: DataFrame with activity segments

    Returns:
        DataFrame with second-by-second timeline
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

################################################################################
# Time-based and Transition Detection Metrics
################################################################################

def match_tasks(df_pred, df_gt, task_col="Activity"):
    """
    Match tasks between prediction and ground truth DataFrames.

    Args:
        df_pred: Prediction DataFrame
        df_gt: Ground truth DataFrame
        task_col: Column name for activity/task

    Returns:
        Merged DataFrame with prediction and ground truth metrics
    """
    df_pred_agg = (
        df_pred
        .groupby(task_col, as_index=False)
        .agg({"Active_seconds":"sum", "Inactive_seconds":"sum"})
    )
    df_pred_agg["Total_seconds"] = df_pred_agg["Active_seconds"] + df_pred_agg["Inactive_seconds"]
    df_pred_agg = df_pred_agg.rename(columns={
        "Active_seconds":"pred_active",
        "Inactive_seconds":"pred_inactive",
        "Total_seconds":"pred_total"
    })

    df_gt_agg = (
        df_gt
        .groupby(task_col, as_index=False)
        .agg({"Active_seconds":"sum", "Inactive_seconds":"sum"})
    )
    df_gt_agg["Total_seconds"] = df_gt_agg["Active_seconds"] + df_gt_agg["Inactive_seconds"]
    df_gt_agg = df_gt_agg.rename(columns={
        "Active_seconds":"gt_active",
        "Inactive_seconds":"gt_inactive",
        "Total_seconds":"gt_total"
    })

    df_merged = pd.merge(df_pred_agg, df_gt_agg, on=task_col, how="outer")

    for c in ["pred_active","pred_inactive","pred_total","gt_active","gt_inactive","gt_total"]:
        df_merged[c] = df_merged[c].fillna(0)

    return df_merged

def compute_time_deviation_metrics(df, task_col="Activity"):
    """
    Compute MAE, MAPE for total time, active time, inactive time across tasks.

    Args:
        df: DataFrame with matched tasks
        task_col: Column name for activity/task

    Returns:
        Dictionary with computed metrics
    """
    df_out = df.copy()

    # active (our primary metric of interest)
    df_out["gt_active_eps"] = df_out["gt_active"].replace({0:1e-9})
    df_out["abs_error_active"] = (df_out["pred_active"] - df_out["gt_active"]).abs()
    df_out["pct_error_active"] = df_out["abs_error_active"] / df_out["gt_active_eps"]
    mae_active = df_out["abs_error_active"].mean()
    mape_active = df_out["pct_error_active"].mean()*100

    # total and inactive metrics
    df_out["gt_total_eps"] = df_out["gt_total"].replace({0:1e-9})
    df_out["abs_error_total"] = (df_out["pred_total"] - df_out["gt_total"]).abs()
    df_out["pct_error_total"] = df_out["abs_error_total"]/df_out["gt_total_eps"]
    mae_total = df_out["abs_error_total"].mean()
    mape_total = df_out["pct_error_total"].mean()*100

    df_out["gt_inactive_eps"] = df_out["gt_inactive"].replace({0:1e-9})
    df_out["abs_error_inactive"] = (df_out["pred_inactive"] - df_out["gt_inactive"]).abs()
    df_out["pct_error_inactive"] = df_out["abs_error_inactive"] / df_out["gt_inactive_eps"]
    mae_inactive = df_out["abs_error_inactive"].mean()
    mape_inactive = df_out["pct_error_inactive"].mean()*100

    metric_summary = {
        "MAE_Total": mae_total,
        "MAPE_Total": mape_total,
        "MAE_Active": mae_active,
        "MAPE_Active": mape_active,
        "MAE_Inactive": mae_inactive,
        "MAPE_Inactive": mape_inactive
    }

    return metric_summary

def compute_transition_detection_metrics(df_pred, df_gt, task_col="Activity"):
    """
    Evaluate transitions between tasks by comparing (Task_i -> Task_i+1) patterns.

    Args:
        df_pred: Prediction DataFrame
        df_gt: Ground truth DataFrame
        task_col: Column name for activity/task

    Returns:
        Dictionary with transition detection metrics
    """
    # Build transitions from GT
    transitions_gt = []
    for i in range(len(df_gt)-1):
        curr_act = df_gt[task_col].iloc[i]
        next_act = df_gt[task_col].iloc[i+1]
        transitions_gt.append((curr_act, next_act))

    # Build transitions from Pred
    transitions_pred = []
    for i in range(len(df_pred)-1):
        curr_act = df_pred[task_col].iloc[i]
        next_act = df_pred[task_col].iloc[i+1]
        transitions_pred.append((curr_act, next_act))

    set_gt = set(transitions_gt)
    set_pred = set(transitions_pred)

    tp = len(set_gt.intersection(set_pred))
    fp = len(set_pred - set_gt)
    fn = len(set_gt - set_pred)

    precision = tp / (tp + fp) if (tp+fp)>0 else 0
    recall = tp / (tp + fn) if (tp+fn)>0 else 0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0

    results = {
        "Transition_Precision": precision,
        "Transition_Recall": recall,
        "Transition_F1": f1,
        "TP_transitions": tp,
        "FP_transitions": fp,
        "FN_transitions": fn
    }
    return results

################################################################################
# Bayesian Optimization Functions
################################################################################

def evaluate_params_active_time_mape(input_csvs, ground_truth_dir, output_dir, params):
    """
    Run consolidation with given parameters and evaluate active time MAPE.

    Args:
        input_csvs: List of input CSV file paths
        ground_truth_dir: Path to ground truth directory
        output_dir: Path to output directory
        params: Tuple of (threshold, title_change, state_change, prelogin)

    Returns:
        MAPE value (for minimization in Bayesian optimization)
    """
    threshold, title_change, state_change, prelogin = params

    # Ensure we're working with absolute paths
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    output_dir = os.path.abspath(output_dir)

    # Create param-specific output directory
    param_str = f"t{threshold:.3f}_tc{title_change}_sc{state_change}_pl{prelogin}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)

    file_metrics = []

    # Find valid pairs of input and ground truth files
    valid_pairs = []
    for input_csv in input_csvs:
        input_csv = os.path.abspath(input_csv)
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        gt_file = f"{base_name}_MANUALLYLABELED.csv"
        gt_path = os.path.join(ground_truth_dir, gt_file)

        if os.path.exists(gt_path):
            valid_pairs.append((input_csv, gt_path, base_name))

    # Process each valid pair without inner progress bar
    for input_csv, gt_path, base_name in valid_pairs:
        # Save current directory
        orig_dir = os.getcwd()

        try:
            # Change to parameter directory for output
            os.chdir(param_dir)

            # Create consolidated directory
            os.makedirs("consolidated", exist_ok=True)

            # Create inputs directory and copy file
            os.makedirs("inputs", exist_ok=True)
            input_copy = os.path.join("inputs", os.path.basename(input_csv))
            shutil.copy2(input_csv, input_copy)

            # Run consolidation
            consolidate(
                input_copy,
                threshold=threshold,
                title_change_threshold=title_change,
                state_change_threshold=state_change,
                n_prelogin=prelogin
            )

            # Get output path
            out_path = os.path.join("consolidated", f"{base_name}_parsed.csv")

            if not os.path.exists(out_path):
                os.chdir(orig_dir)
                continue

            # Evaluate against ground truth
            try:
                # Load prediction and ground truth
                pred_df = pd.read_csv(out_path)
                gt_df = pd.read_csv(gt_path)

                # Skip 'END' rows if they exist
                pred_df = pred_df[pred_df['Activity'] != 'END'].copy() if 'Activity' in pred_df.columns else pred_df
                gt_df = gt_df[gt_df['Activity'] != 'END'].copy() if 'Activity' in gt_df.columns else gt_df

                # Match tasks and compute time-based metrics
                merged_df = match_tasks(pred_df, gt_df, task_col="Activity")
                time_metrics = compute_time_deviation_metrics(merged_df)

                # Add to metrics
                file_metrics.append({
                    'file': base_name,
                    'MAPE_Active': time_metrics['MAPE_Active'],
                    'MAE_Active': time_metrics['MAE_Active']
                })

            except Exception as e:
                tqdm.write(f"Error evaluating {base_name}: {str(e)}")

        finally:
            # Change back to original directory
            os.chdir(orig_dir)

    # Calculate average MAPE across files
    if not file_metrics:
        # Return a very high value if evaluation failed
        return 1000.0

    mean_mape_active = np.mean([m['MAPE_Active'] for m in file_metrics])

    # For Bayesian optimization, we want to minimize MAPE
    return mean_mape_active

def evaluate_params_task_change(input_csvs, ground_truth_dir, output_dir, params):
    """
    Run consolidation with given parameters and evaluate task change sensitivity.

    Args:
        input_csvs: List of input CSV file paths
        ground_truth_dir: Path to ground truth directory
        output_dir: Path to output directory
        params: Tuple of (threshold, title_change, state_change, prelogin)

    Returns:
        Negative F1 score (for minimization in Bayesian optimization)
    """
    threshold, title_change, state_change, prelogin = params

    # Ensure we're working with absolute paths
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    output_dir = os.path.abspath(output_dir)

    # Create param-specific output directory
    param_str = f"t{threshold:.3f}_tc{title_change}_sc{state_change}_pl{prelogin}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)

    file_metrics = []

    # Find valid pairs of input and ground truth files
    valid_pairs = []
    for input_csv in input_csvs:
        input_csv = os.path.abspath(input_csv)
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        gt_file = f"{base_name}_MANUALLYLABELED.csv"
        gt_path = os.path.join(ground_truth_dir, gt_file)

        if os.path.exists(gt_path):
            valid_pairs.append((input_csv, gt_path, base_name))

    # Process each valid pair without inner progress bar
    for input_csv, gt_path, base_name in valid_pairs:
        # Save current directory
        orig_dir = os.getcwd()

        try:
            # Change to parameter directory for output
            os.chdir(param_dir)

            # Create consolidated directory
            os.makedirs("consolidated", exist_ok=True)

            # Create inputs directory and copy file
            os.makedirs("inputs", exist_ok=True)
            input_copy = os.path.join("inputs", os.path.basename(input_csv))
            shutil.copy2(input_csv, input_copy)

            # Run consolidation
            consolidate(
                input_copy,
                threshold=threshold,
                title_change_threshold=title_change,
                state_change_threshold=state_change,
                n_prelogin=prelogin
            )

            # Get output path
            out_path = os.path.join("consolidated", f"{base_name}_parsed.csv")

            if not os.path.exists(out_path):
                os.chdir(orig_dir)
                continue

            # Evaluate against ground truth
            try:
                # Load prediction and ground truth
                pred_df = pd.read_csv(out_path)
                gt_df = pd.read_csv(gt_path)

                # Skip 'END' rows if they exist
                pred_df = pred_df[pred_df['Activity'] != 'END'].copy() if 'Activity' in pred_df.columns else pred_df
                gt_df = gt_df[gt_df['Activity'] != 'END'].copy() if 'Activity' in gt_df.columns else gt_df

                # Compute transition detection metrics
                transition_metrics = compute_transition_detection_metrics(pred_df, gt_df)

                # Add to metrics
                file_metrics.append({
                    'file': base_name,
                    'Transition_F1': transition_metrics['Transition_F1'],
                    'Transition_Precision': transition_metrics['Transition_Precision'],
                    'Transition_Recall': transition_metrics['Transition_Recall']
                })

            except Exception as e:
                tqdm.write(f"Error evaluating {base_name}: {str(e)}")

        finally:
            # Change back to original directory
            os.chdir(orig_dir)

    # Calculate average F1 score across files
    if not file_metrics:
        # Return a very low value if evaluation failed
        return -0.0

    mean_f1 = np.mean([m['Transition_F1'] for m in file_metrics])

    # For Bayesian optimization, we want to minimize negative F1 score (maximize F1)
    return -mean_f1

def run_dual_bayesian_optimization(input_dir="inputs", ground_truth_dir="ground_truth",
                                  output_dir="results", n_calls=15, random_state=42):
    """
    Run two separate Bayesian optimizations for parameter tuning.

    This function runs two optimizations:
    1. Optimize for active time MAPE (Mean Absolute Percentage Error)
    2. Optimize for task change sensitivity (transition F1 score)

    Args:
        input_dir: Directory containing input CSV files
        ground_truth_dir: Directory containing ground truth CSV files
        output_dir: Directory to save output files
        n_calls: Number of iterations for each Bayesian optimization
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing results from both optimizations
    """
    # Convert to absolute paths to avoid directory issues
    input_dir = os.path.abspath(input_dir)
    ground_truth_dir = os.path.abspath(ground_truth_dir)
    output_dir = os.path.abspath(output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each optimization
    active_time_dir = os.path.join(output_dir, "active_time_mape")
    task_change_dir = os.path.join(output_dir, "task_change")
    os.makedirs(active_time_dir, exist_ok=True)
    os.makedirs(task_change_dir, exist_ok=True)

    # Create input and ground_truth dirs if they don't exist
    for dir_name in [input_dir, ground_truth_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")

    # Check if dirs are empty and provide instructions
    if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
        print(f"Warning: The '{input_dir}' directory is empty or doesn't exist.")
        print("Please add your model output CSV files to this directory.")
        print("These should have columns: time_stamp, page_title, sidebar, ...")
        return None

    if not os.path.exists(ground_truth_dir) or len(os.listdir(ground_truth_dir)) == 0:
        print(f"Warning: The '{ground_truth_dir}' directory is empty or doesn't exist.")
        print("Please add your ground truth files to this directory.")
        print("These should be named <filename>_MANUALLYLABELED.csv")
        return None

    # Find input CSVs
    input_csvs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(".csv")]

    print(f"Found {len(input_csvs)} input CSV files")

    # Find ground truth files
    gt_files = [f for f in os.listdir(ground_truth_dir)
               if f.endswith("_MANUALLYLABELED.csv")]

    print(f"Found {len(gt_files)} ground truth files")

    if not gt_files:
        print("Error: No ground truth files found.")
        return None

    # Define the search space for parameters
    space = [
        Real(0.05, 0.5, name='threshold'),
        Integer(1, 10, name='title_change_threshold'),
        Integer(1, 10, name='state_change_threshold'),
        Integer(1, 20, name='n_prelogin')
    ]

    ############################################################################
    # 1. Run Bayesian Optimization for Active Time MAPE
    ############################################################################
    print("\n========== STARTING OPTIMIZATION FOR ACTIVE TIME MAPE ==========\n")

    # Define the objective function for active time MAPE
    @use_named_args(space)
    def objective_active_time_mape(threshold, title_change_threshold, state_change_threshold, n_prelogin):
        # Create a parent directory for this optimization run
        opt_dir = os.path.join(active_time_dir, "bayesian_opt")
        os.makedirs(opt_dir, exist_ok=True)

        # Evaluate the parameters
        params = (threshold, title_change_threshold, state_change_threshold, n_prelogin)
        mape = evaluate_params_active_time_mape(input_csvs, ground_truth_dir, opt_dir, params)

        # Output result without disrupting progress bar
        tqdm.write(f"Active MAPE - Tested: threshold={threshold:.3f}, "
              f"title_change={title_change_threshold}, "
              f"state_change={state_change_threshold}, "
              f"prelogin={n_prelogin}, "
              f"MAPE={mape:.4f}")

        return mape

    # Run the Bayesian optimization for active time MAPE with progress bar
    with tqdm(total=n_calls, desc="Active MAPE Optimization") as pbar:
        def callback_mape(res):
            pbar.update(1)

        result_active_mape = gp_minimize(
            objective_active_time_mape,
            space,
            n_calls=n_calls,
            random_state=random_state,
            verbose=False,
            n_initial_points=5,
            callback=callback_mape
        )

    # Get the best parameters for active time MAPE
    best_threshold_mape = result_active_mape.x[0]
    best_title_change_mape = result_active_mape.x[1]
    best_state_change_mape = result_active_mape.x[2]
    best_prelogin_mape = result_active_mape.x[3]
    best_mape = result_active_mape.fun

    print("\n========== OPTIMAL PARAMETERS (ACTIVE TIME MAPE) ==========")
    print(f"Threshold: {best_threshold_mape:.3f}")
    print(f"Title change threshold: {best_title_change_mape}")
    print(f"State change threshold: {best_state_change_mape}")
    print(f"Prelogin frames: {best_prelogin_mape}")
    print(f"Active time MAPE: {best_mape:.4f}")

    # Create a results directory for active time MAPE
    mape_results_dir = os.path.join(active_time_dir, "optimization_results")
    os.makedirs(mape_results_dir, exist_ok=True)

    # Save active time MAPE results
    import joblib
    joblib.dump(result_active_mape, os.path.join(mape_results_dir, 'optimization_result.pkl'))

    # Save parameter combinations and scores for active time MAPE
    iterations_mape = []
    for i, (params, value) in enumerate(zip(result_active_mape.x_iters, result_active_mape.func_vals)):
        iterations_mape.append({
            'iteration': i,
            'threshold': params[0],
            'title_change_threshold': params[1],
            'state_change_threshold': params[2],
            'n_prelogin': params[3],
            'mape_active': value
        })

    iterations_mape_df = pd.DataFrame(iterations_mape)
    iterations_mape_df.to_csv(os.path.join(mape_results_dir, 'optimization_iterations.csv'), index=False)

    # Save summary results to text file
    with open(os.path.join(mape_results_dir, 'optimization_summary.txt'), 'w') as f:
        f.write("BAYESIAN OPTIMIZATION RESULTS - ACTIVE TIME MAPE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of iterations: {n_calls}\n")
        f.write(f"Random state: {random_state}\n\n")
        f.write("OPTIMAL PARAMETERS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Threshold: {best_threshold_mape:.3f}\n")
        f.write(f"Title change threshold: {best_title_change_mape}\n")
        f.write(f"State change threshold: {best_state_change_mape}\n")
        f.write(f"Prelogin frames: {best_prelogin_mape}\n\n")
        f.write(f"Best Active time MAPE: {best_mape:.4f}\n")

    ############################################################################
    # 2. Run Bayesian Optimization for Task Change Sensitivity
    ############################################################################
    print("\n========== STARTING OPTIMIZATION FOR TASK CHANGE SENSITIVITY ==========\n")

    # Define the objective function for task change sensitivity
    @use_named_args(space)
    def objective_task_change(threshold, title_change_threshold, state_change_threshold, n_prelogin):
        # Create a parent directory for this optimization run
        opt_dir = os.path.join(task_change_dir, "bayesian_opt")
        os.makedirs(opt_dir, exist_ok=True)

        # Evaluate the parameters
        params = (threshold, title_change_threshold, state_change_threshold, n_prelogin)
        neg_f1 = evaluate_params_task_change(input_csvs, ground_truth_dir, opt_dir, params)

        # Output result without disrupting progress bar
        tqdm.write(f"Task Change - Tested: threshold={threshold:.3f}, "
              f"title_change={title_change_threshold}, "
              f"state_change={state_change_threshold}, "
              f"prelogin={n_prelogin}, "
              f"F1={-neg_f1:.4f}")

        return neg_f1

    # Run the Bayesian optimization for task change sensitivity with progress bar
    with tqdm(total=n_calls, desc="Task Change Optimization") as pbar:
        def callback_task(res):
            pbar.update(1)

        result_task_change = gp_minimize(
            objective_task_change,
            space,
            n_calls=n_calls,
            random_state=random_state,
            verbose=False,
            n_initial_points=5,
            callback=callback_task
        )

    # Get the best parameters for task change sensitivity
    best_threshold_task = result_task_change.x[0]
    best_title_change_task = result_task_change.x[1]
    best_state_change_task = result_task_change.x[2]
    best_prelogin_task = result_task_change.x[3]
    best_f1 = -result_task_change.fun  # Convert negative back to positive F1

    print("\n========== OPTIMAL PARAMETERS (TASK CHANGE SENSITIVITY) ==========")
    print(f"Threshold: {best_threshold_task:.3f}")
    print(f"Title change threshold: {best_title_change_task}")
    print(f"State change threshold: {best_state_change_task}")
    print(f"Prelogin frames: {best_prelogin_task}")
    print(f"Task change F1 score: {best_f1:.4f}")

    # Create a results directory for task change sensitivity
    task_results_dir = os.path.join(task_change_dir, "optimization_results")
    os.makedirs(task_results_dir, exist_ok=True)

    # Save task change sensitivity results
    joblib.dump(result_task_change, os.path.join(task_results_dir, 'optimization_result.pkl'))

    # Save parameter combinations and scores for task change sensitivity
    iterations_task = []
    for i, (params, value) in enumerate(zip(result_task_change.x_iters, result_task_change.func_vals)):
        iterations_task.append({
            'iteration': i,
            'threshold': params[0],
            'title_change_threshold': params[1],
            'state_change_threshold': params[2],
            'n_prelogin': params[3],
            'f1_score': -value  # Convert negative back to positive F1
        })

    iterations_task_df = pd.DataFrame(iterations_task)
    iterations_task_df.to_csv(os.path.join(task_results_dir, 'optimization_iterations.csv'), index=False)

    # Save summary results to text file
    with open(os.path.join(task_results_dir, 'optimization_summary.txt'), 'w') as f:
        f.write("BAYESIAN OPTIMIZATION RESULTS - TASK CHANGE SENSITIVITY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of iterations: {n_calls}\n")
        f.write(f"Random state: {random_state}\n\n")
        f.write("OPTIMAL PARAMETERS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Threshold: {best_threshold_task:.3f}\n")
        f.write(f"Title change threshold: {best_title_change_task}\n")
        f.write(f"State change threshold: {best_state_change_task}\n")
        f.write(f"Prelogin frames: {best_prelogin_task}\n\n")
        f.write(f"Best Task change F1 score: {best_f1:.4f}\n")

    ############################################################################
    # 3. Create final output directories with best parameters from both runs
    ############################################################################

    # Create final output directory for active time MAPE
    mape_final_dir = os.path.join(active_time_dir, "final")
    os.makedirs(mape_final_dir, exist_ok=True)
    os.makedirs(os.path.join(mape_final_dir, "consolidated"), exist_ok=True)

    # Create final output directory for task change sensitivity
    task_final_dir = os.path.join(task_change_dir, "final")
    os.makedirs(task_final_dir, exist_ok=True)
    os.makedirs(os.path.join(task_final_dir, "consolidated"), exist_ok=True)

    # Create final outputs with best parameters for active time MAPE
    print("\nCreating final outputs with best parameters for active time MAPE...")
    mape_best_params = {
        'threshold': best_threshold_mape,
        'title_change_threshold': best_title_change_mape,
        'state_change_threshold': best_state_change_mape,
        'n_prelogin': best_prelogin_mape
    }

    # Create final outputs with best parameters for task change sensitivity
    print("\nCreating final outputs with best parameters for task change sensitivity...")
    task_best_params = {
        'threshold': best_threshold_task,
        'title_change_threshold': best_title_change_task,
        'state_change_threshold': best_state_change_task,
        'n_prelogin': best_prelogin_task
    }

    # Save current directory
    orig_dir = os.getcwd()

    # Process each input file for MAPE optimization results with progress bar
    mape_files = [os.path.splitext(os.path.basename(csv))[0] for csv in input_csvs]
    for base_name in tqdm(mape_files, desc="Creating MAPE optimized outputs"):
        input_csv = os.path.join(input_dir, f"{base_name}.csv")
        if not os.path.exists(input_csv):
            continue

        try:
            # Change to MAPE final directory
            os.chdir(mape_final_dir)

            # Create inputs directory
            os.makedirs("inputs", exist_ok=True)
            input_copy = os.path.join("inputs", f"{base_name}.csv")
            shutil.copy2(input_csv, input_copy)

            # Run consolidation with best MAPE parameters
            consolidate(
                input_copy,
                threshold=mape_best_params['threshold'],
                title_change_threshold=mape_best_params['title_change_threshold'],
                state_change_threshold=mape_best_params['state_change_threshold'],
                n_prelogin=mape_best_params['n_prelogin']
            )

            # Copy to main output directory
            consolidated_path = os.path.join("consolidated", f"{base_name}_parsed.csv")
            if os.path.exists(consolidated_path):
                shutil.copy2(
                    consolidated_path,
                    os.path.join(orig_dir, active_time_dir, f"{base_name}_final_mape.csv")
                )
        except Exception as e:
            tqdm.write(f"  Error processing {base_name} for MAPE: {str(e)}")
        finally:
            os.chdir(orig_dir)

    # Process each input file for task change optimization results with progress bar
    task_files = [os.path.splitext(os.path.basename(csv))[0] for csv in input_csvs]
    for base_name in tqdm(task_files, desc="Creating task change optimized outputs"):
        input_csv = os.path.join(input_dir, f"{base_name}.csv")
        if not os.path.exists(input_csv):
            continue

        try:
            # Change to task change final directory
            os.chdir(task_final_dir)

            # Create inputs directory
            os.makedirs("inputs", exist_ok=True)
            input_copy = os.path.join("inputs", f"{base_name}.csv")
            shutil.copy2(input_csv, input_copy)

            # Run consolidation with best task change parameters
            consolidate(
                input_copy,
                threshold=task_best_params['threshold'],
                title_change_threshold=task_best_params['title_change_threshold'],
                state_change_threshold=task_best_params['state_change_threshold'],
                n_prelogin=task_best_params['n_prelogin']
            )

            # Copy to main output directory
            consolidated_path = os.path.join("consolidated", f"{base_name}_parsed.csv")
            if os.path.exists(consolidated_path):
                shutil.copy2(
                    consolidated_path,
                    os.path.join(orig_dir, task_change_dir, f"{base_name}_final_task.csv")
                )
        except Exception as e:
            tqdm.write(f"  Error processing {base_name} for task change: {str(e)}")
        finally:
            os.chdir(orig_dir)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Active time MAPE results: {active_time_dir}")
    print(f"  - Task change results: {task_change_dir}")

    # Return results for further analysis
    return {
        'active_time_mape': {
            'best_params': mape_best_params,
            'mape_value': best_mape,
            'iterations': iterations_mape_df,
            'result': result_active_mape
        },
        'task_change': {
            'best_params': task_best_params,
            'f1_score': best_f1,
            'iterations': iterations_task_df,
            'result': result_task_change
        }
    }


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for EHR Activity Detection Parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python bayesian_optimization.py

  # Run with custom directories and iterations
  python bayesian_optimization.py --input-dir data/inputs --gt-dir data/ground_truth --n-calls 50

  # Run with configuration file
  python bayesian_optimization.py --config config.yaml

  # Run with specific output directory
  python bayesian_optimization.py --output-dir my_results --n-calls 100
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='inputs',
        help='Directory containing input CSV files (default: inputs)'
    )

    parser.add_argument(
        '--gt-dir',
        type=str,
        default='ground_truth',
        help='Directory containing ground truth CSV files (default: ground_truth)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save output files (default: results)'
    )

    parser.add_argument(
        '--n-calls',
        type=int,
        default=15,
        help='Number of Bayesian optimization iterations (default: 15)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    return parser.parse_args()


def main():
    """
    Main function to run Bayesian optimization from command line.
    """
    args = parse_arguments()

    # If config file is provided, load it and override defaults
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file '{args.config}' not found.")
            sys.exit(1)

        config = load_config(args.config)

        # Override arguments with config values
        input_dir = config.get('input_dir', args.input_dir)
        gt_dir = config.get('ground_truth_dir', args.gt_dir)
        output_dir = config.get('output_dir', args.output_dir)
        n_calls = config.get('n_calls', args.n_calls)
        random_state = config.get('random_state', args.random_state)
    else:
        # Use command line arguments
        input_dir = args.input_dir
        gt_dir = args.gt_dir
        output_dir = args.output_dir
        n_calls = args.n_calls
        random_state = args.random_state

    # Print configuration
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION FOR EHR ACTIVITY DETECTION")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Ground truth directory: {gt_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of iterations: {n_calls}")
    print(f"  Random state: {random_state}")
    print("=" * 60 + "\n")

    # Run the optimization
    results = run_dual_bayesian_optimization(
        input_dir=input_dir,
        ground_truth_dir=gt_dir,
        output_dir=output_dir,
        n_calls=n_calls,
        random_state=random_state
    )

    if results is None:
        print("\nOptimization failed. Please check the error messages above.")
        sys.exit(1)

    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()
