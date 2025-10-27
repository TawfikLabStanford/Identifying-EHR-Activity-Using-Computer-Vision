#!/usr/bin/env python
# coding: utf-8

# ## By Individual Video

# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

##############################################################################
# 1) Helper Functions
##############################################################################
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute MAPE, handling zeros in y_true by ignoring them.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    non_zero_mask = (y_true != 0)
    if np.sum(non_zero_mask) == 0:
        return 0.0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def match_transitions_with_tolerance(gt_times, pred_times, delta=2):
    """
    Match predicted transitions to ground truth with tolerance window.

    For each ground truth transition time, any predicted transition time
    within [t - delta, t + delta] is considered a match.

    Parameters:
        gt_times: Ground truth transition times
        pred_times: Predicted transition times
        delta (int): Tolerance in frames (default: 2, per paper Section 3.2.2)

    Returns:
        int: Count of matched transitions
    """
    matched_count = 0
    used_pred = set()
    for t_gt in gt_times:
        low_bound = t_gt - delta
        high_bound = t_gt + delta
        candidates = [
            t_pred for t_pred in pred_times
            if (low_bound <= t_pred <= high_bound) and (t_pred not in used_pred)
        ]
        if candidates:
            matched_count += 1
            used_pred.add(candidates[0])
    return matched_count

def detect_transitions_blocks(df):
    """
    Detect transitions between rows based on the 'Activity' column.
    Returns a list of time_start values for rows where the task changes.
    """
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]["Activity"] != df.iloc[i - 1]["Activity"]:
            transitions.append(df.iloc[i]["time_start"])
    return transitions

def compute_time_by_task(df):
    """
    Compute the total time per task by summing Active_seconds and Inactive_seconds.
    Returns a dict with task names as keys.
    """
    df = df.copy()
    df["Total_time"] = df["Active_seconds"] + df["Inactive_seconds"]
    return df.groupby("Activity")["Total_time"].sum().to_dict()

def compute_active_inactive(df):
    """
    Compute total active and inactive time.
    """
    total_active = df["Active_seconds"].sum()
    total_inactive = df["Inactive_seconds"].sum()
    return total_active, total_inactive

##############################################################################
# 2) Evaluation Function for a Single Video
##############################################################################
def evaluate_video(video_id, gt_file, pred_file, tolerance=1):
    print(f"\nProcessing video: {video_id}")
    
    # Read CSVs (assuming tab-delimited; adjust delimiter if needed)
    df_gt = pd.read_csv(gt_file)
    df_pred = pd.read_csv(pred_file)
    
    # Fill missing numeric values for seconds with 0 and convert to numeric
    for col in ["Active_seconds", "Inactive_seconds"]:
        df_gt[col] = pd.to_numeric(df_gt[col], errors="coerce").fillna(0)
        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce").fillna(0)
    # Ensure time_start is numeric
    df_gt["time_start"] = pd.to_numeric(df_gt["time_start"], errors="coerce").fillna(0)
    df_pred["time_start"] = pd.to_numeric(df_pred["time_start"], errors="coerce").fillna(0)
    
    # Create results folders if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Save and display descriptive statistics
    gt_desc = df_gt.describe(include="all")
    pred_desc = df_pred.describe(include="all")
    gt_desc.to_csv(f"results/{video_id}_GT_description.csv")
    pred_desc.to_csv(f"results/{video_id}_Pred_description.csv")
    
    print("\nGround Truth Descriptive Statistics:")
    display(gt_desc)
    
    print("\nPrediction Descriptive Statistics:")
    display(pred_desc)
    
    # Transition detection
    gt_transitions = detect_transitions_blocks(df_gt)
    pred_transitions = detect_transitions_blocks(df_pred)
    matched_count = match_transitions_with_tolerance(gt_transitions, pred_transitions, delta=tolerance)
    gt_transition_count = len(gt_transitions)
    pred_transition_count = len(pred_transitions)
    transition_sensitivity = matched_count / gt_transition_count if gt_transition_count > 0 else 0
    
    transitions_df = pd.DataFrame({
        "Metric": ["GT transitions", "Pred transitions", "Matched transitions", "Transition Sensitivity"],
        "Value": [gt_transition_count, pred_transition_count, matched_count, transition_sensitivity]
    })
    transitions_df.to_csv(f"results/{video_id}_transitions.csv", index=False)
    
    print("\nTransition Metrics:")
    display(transitions_df)
    
    # Time per task
    gt_time_task = compute_time_by_task(df_gt)
    pred_time_task = compute_time_by_task(df_pred)
    tasks_union = set(gt_time_task.keys()).union(set(pred_time_task.keys()))
    time_task_rows = []
    ape_list = []
    for t in sorted(tasks_union):
        t_gt = gt_time_task.get(t, 0)
        t_pred = pred_time_task.get(t, 0)
        if t_gt > 0:
            ape_list.append(abs(t_gt - t_pred) / t_gt)
        time_task_rows.append({"Task": t, "GT_time": t_gt, "Pred_time": t_pred})
    time_task_df = pd.DataFrame(time_task_rows)
    overall_mape_task = np.mean(ape_list) * 100 if ape_list else 0
    time_task_df.to_csv(f"results/{video_id}_time_per_task.csv", index=False)
    
    print("\nTime per Task:")
    display(time_task_df)
    print(f"Overall MAPE for time per task: {overall_mape_task:.2f}%")
    
    # Active vs Inactive time
    gt_active, gt_inactive = compute_active_inactive(df_gt)
    pred_active, pred_inactive = compute_active_inactive(df_pred)
    ai_gt = [gt_active, gt_inactive]
    ai_pred = [pred_active, pred_inactive]
    ai_mape = mean_absolute_percentage_error(ai_gt, ai_pred)
    ai_df = pd.DataFrame({
        "Category": ["Active", "Inactive"],
        "GT": ai_gt,
        "Pred": ai_pred
    })
    ai_df.to_csv(f"results/{video_id}_active_inactive.csv", index=False)
    
    print("\nActive vs Inactive Time:")
    display(ai_df)
    print(f"Overall MAPE for active/inactive time: {ai_mape*100:.2f}%")
    
    # Overall video metrics summary
    metrics = {
        "video_id": video_id,
        "gt_blocks": len(df_gt),
        "pred_blocks": len(df_pred),
        "gt_transition_count": gt_transition_count,
        "pred_transition_count": pred_transition_count,
        "matched_transitions": matched_count,
        "transition_sensitivity": transition_sensitivity,
        "overall_mape_time_per_task": overall_mape_task,
        "active_inactive_mape": ai_mape,
        "gt_total_time": (df_gt["Active_seconds"] + df_gt["Inactive_seconds"]).sum(),
        "pred_total_time": (df_pred["Active_seconds"] + df_pred["Inactive_seconds"]).sum()
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"results/{video_id}_metrics.csv", index=False)
    
    print("\nVideo Metrics Summary:")
    display(metrics_df)
    
    # Plotting (both saving to disk and displaying inline)
    width = 0.35
    
    # Plot: Time per Task
    fig, ax = plt.subplots()
    x = np.arange(len(time_task_df))
    ax.bar(x - width/2, time_task_df["GT_time"], width, label="GT")
    ax.bar(x + width/2, time_task_df["Pred_time"], width, label="Pred")
    ax.set_xticks(x)
    ax.set_xticklabels(time_task_df["Task"], rotation=45)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Time per Task - Video {video_id}\n(MAPE: {overall_mape_task:.2f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{video_id}_time_per_task.png")
    plt.show()
    
    # Plot: Active vs Inactive
    fig, ax = plt.subplots()
    categories = ["Active", "Inactive"]
    x = np.arange(len(categories))
    ax.bar(x - width/2, ai_df["GT"], width, label="GT")
    ax.bar(x + width/2, ai_df["Pred"], width, label="Pred")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Active vs Inactive - Video {video_id}\n(MAPE: {ai_mape:.2f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{video_id}_active_inactive.png")
    plt.show()
    
    # Plot: Transition Counts
    fig, ax = plt.subplots()
    counts = [gt_transition_count, pred_transition_count, matched_count]
    labels = ["GT Transitions", "Pred Transitions", "Matched Transitions"]
    ax.bar(labels, counts, color=["blue", "orange", "green"])
    ax.set_ylabel("Count")
    ax.set_title(f"Transition Counts - Video {video_id}")
    plt.tight_layout()
    plt.savefig(f"results/plots/{video_id}_transitions.png")
    plt.show()
    
    # Append video_id to time_task_df for task-level aggregation later
    time_task_df["video_id"] = video_id
    
    return metrics, time_task_df

##############################################################################
# 3) Main Processing Function with Additional Aggregations
##############################################################################
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    folder = "consolidated"
    gt_files = glob.glob(os.path.join(folder, "*_MANUALLYLABELED.csv"))
    all_metrics = []
    all_task_frames = []  # To accumulate per-video task data
    
    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        video_id = base.replace("_MANUALLYLABELED.csv", "")
        pred_file = os.path.join(folder, f"{video_id}_parsed.csv")
        if os.path.exists(pred_file):
            metrics, task_df = evaluate_video(video_id, gt_file, pred_file, tolerance=1)
            all_metrics.append(metrics)
            all_task_frames.append(task_df)
        else:
            print(f"Prediction file for video '{video_id}' not found.")
    
    # Aggregate results across videos (video-level)
    if all_metrics:
        agg_df = pd.DataFrame(all_metrics)
        agg_df.to_csv("results/aggregated_metrics.csv", index=False)
        print("\nAggregated Metrics Across Videos:")
        display(agg_df)
        
        # Existing boxplot plots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.boxplot(y=agg_df["transition_sensitivity"], ax=ax[0])
        ax[0].set_title("Transition Sensitivity")
        sns.boxplot(y=agg_df["overall_mape_time_per_task"], ax=ax[1])
        ax[1].set_title("MAPE Time per Task")
        sns.boxplot(y=agg_df["active_inactive_mape"], ax=ax[2])
        ax[2].set_title("Active/Inactive MAPE")
        plt.tight_layout()
        plt.savefig("results/plots/aggregated_metrics.png")
        plt.show()
        
        # Additional Overall (Video-level) Aggregations:
        overall_stats = agg_df.describe()
        overall_stats.to_csv("results/aggregated_video_metrics_descriptive.csv")
        
        # Scatter plot: GT_total_time vs Pred_total_time
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=agg_df, x="gt_total_time", y="pred_total_time", ax=ax)
        ax.set_title("GT Total Time vs Pred Total Time Across Videos")
        plt.tight_layout()
        plt.savefig("results/plots/gt_vs_pred_total_time.png")
        plt.show()
        
        # Histograms for key video-level metrics
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(agg_df["transition_sensitivity"], kde=True, ax=ax[0])
        ax[0].set_title("Transition Sensitivity Distribution")
        sns.histplot(agg_df["overall_mape_time_per_task"], kde=True, ax=ax[1])
        ax[1].set_title("Overall MAPE Time per Task Distribution")
        sns.histplot(agg_df["active_inactive_mape"], kde=True, ax=ax[2])
        ax[2].set_title("Active/Inactive MAPE Distribution")
        plt.tight_layout()
        plt.savefig("results/plots/video_metrics_histograms.png")
        plt.show()
        
        print(f"Correlation between GT and Pred Total Time: {agg_df['gt_total_time'].corr(agg_df['pred_total_time']):.2f}")
    else:
        print("No videos processed.")
    
    # Additional Aggregations Per Task Across Videos:
    if all_task_frames:
        all_tasks_df = pd.concat(all_task_frames, ignore_index=True)
        # Calculate error metrics per task instance
        all_tasks_df["abs_error"] = abs(all_tasks_df["GT_time"] - all_tasks_df["Pred_time"])
        all_tasks_df["pct_error"] = np.where(all_tasks_df["GT_time"] > 0, (all_tasks_df["abs_error"] / all_tasks_df["GT_time"]) * 100, 0)
        
        # Group by Task and compute aggregated statistics on a per-instance basis
        task_agg = all_tasks_df.groupby("Task").agg(
            count=("video_id", "count"),
            mean_GT_time=("GT_time", "mean"),
            mean_Pred_time=("Pred_time", "mean"),
            std_GT_time=("GT_time", "std"),
            std_Pred_time=("Pred_time", "std"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_pct_error=("pct_error", "mean"),
            median_pct_error=("pct_error", "median")
        ).reset_index()
        
        task_agg.to_csv("results/aggregated_task_metrics.csv", index=False)
        print("\nAggregated Task Metrics Across Videos (per-instance):")
        display(task_agg)
        
        # New Aggregation: Sum total GT and Pred times per task across all videos
        aggregated_task_time = all_tasks_df.groupby("Task").agg(
            total_GT_time=("GT_time", "sum"),
            total_Pred_time=("Pred_time", "sum")
        ).reset_index()
        aggregated_task_time["aggregated_MAPE"] = aggregated_task_time.apply(
            lambda row: (abs(row["total_GT_time"] - row["total_Pred_time"]) / row["total_GT_time"] * 100)
            if row["total_GT_time"] > 0 else 0, axis=1)
        aggregated_task_time.to_csv("results/aggregated_task_mape.csv", index=False)
        
        # Plot: Aggregated MAPE per Task (across all videos)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Task", y="aggregated_MAPE", data=aggregated_task_time, ax=ax)
        ax.set_ylabel("Aggregated MAPE (%)")
        ax.set_title("Aggregated MAPE per Task Across All Videos")
        plt.tight_layout()
        plt.savefig("results/plots/aggregated_task_mape.png")
        plt.show()
        
        # Plot: Average GT vs Pred Time per Task (Bar Plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(task_agg))
        ax.bar(x - width/2, task_agg["mean_GT_time"], width, label="GT")
        ax.bar(x + width/2, task_agg["mean_Pred_time"], width, label="Pred")
        ax.set_xticks(x)
        ax.set_xticklabels(task_agg["Task"], rotation=45)
        ax.set_ylabel("Average Time (s)")
        ax.set_title("Average Time per Task Across Videos")
        ax.legend()
        plt.tight_layout()
        plt.savefig("results/plots/aggregated_time_per_task.png")
        plt.show()
        
        # Plot: Distribution of Percentage Errors per Task (Boxplot)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Task", y="pct_error", data=all_tasks_df, ax=ax)
        ax.set_title("Distribution of Percentage Error per Task Across Videos")
        ax.set_ylabel("Percentage Error (%)")
        plt.tight_layout()
        plt.savefig("results/plots/task_pct_error_boxplot.png")
        plt.show()
        
        # Plot: Scatter plot for GT vs Pred Times for each task (colored by Task)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="GT_time", y="Pred_time", hue="Task", data=all_tasks_df, ax=ax)
        min_val = all_tasks_df["GT_time"].min()
        max_val = all_tasks_df["GT_time"].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title("GT vs Pred Time per Task")
        plt.tight_layout()
        plt.savefig("results/plots/task_gt_vs_pred_scatter.png")
        plt.show()

if __name__ == "__main__":
    main()


# ## By aggregation across all videos

# In[9]:


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

##############################################################################
# Helper Functions
##############################################################################
def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAPE, handling zeros in y_true by ignoring them.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() == 0:
        return 0.0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def match_transitions_with_tolerance(gt_times: list, pred_times: list, delta: int = 2) -> int:
    """
    Match predicted transitions to ground truth with tolerance window.

    For each ground truth transition time, any predicted transition time
    within [t - delta, t + delta] is considered a match.

    Parameters:
        gt_times: Ground truth transition times
        pred_times: Predicted transition times
        delta (int): Tolerance in frames (default: 2, per paper Section 3.2.2)

    Returns:
        int: Count of matched transitions
    """
    matched_count = 0
    used_pred = set()
    for gt_time in gt_times:
        low_bound = gt_time - delta
        high_bound = gt_time + delta
        candidates = [
            pred_time for pred_time in pred_times
            if low_bound <= pred_time <= high_bound and pred_time not in used_pred
        ]
        if candidates:
            matched_count += 1
            used_pred.add(candidates[0])
    return matched_count

def detect_transitions_blocks(df: pd.DataFrame) -> list:
    """
    Detect transitions between rows based on the 'Activity' column.
    Returns a list of time_start values for rows where the task changes.
    """
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]["Activity"] != df.iloc[i - 1]["Activity"]:
            transitions.append(df.iloc[i]["time_start"])
    return transitions

def compute_time_by_task(df: pd.DataFrame) -> dict:
    """
    Compute the total time per task by summing Active_seconds and Inactive_seconds.
    Returns a dict with task names as keys mapped to total time.
    """
    df_temp = df.copy()
    df_temp["Total_time"] = df_temp["Active_seconds"] + df_temp["Inactive_seconds"]
    return df_temp.groupby("Activity")["Total_time"].sum().to_dict()

def compute_active_inactive(df: pd.DataFrame) -> tuple:
    """
    Compute total active and inactive time.
    """
    total_active = df["Active_seconds"].sum()
    total_inactive = df["Inactive_seconds"].sum()
    return total_active, total_inactive

##############################################################################
# Frame-Level Expansion and Confusion Matrix Functions
##############################################################################
def build_frame_level_labels(df: pd.DataFrame) -> dict:
    """
    Expand each block in 'df' to individual seconds: 
      (video_id, second) -> Activity.
    Returns a dict mapping (video_id, second) to activity.
    """
    label_dict = {}
    for _, row in df.iterrows():
        video_id = row["video_id"]
        activity = row["Activity"]
        start_time = int(row["time_start"])
        total_time = int(row["Active_seconds"] + row["Inactive_seconds"])
        for sec in range(start_time, start_time + total_time):
            label_dict[(video_id, sec)] = activity
    return label_dict

def compute_confusion_matrix(gt_labels: dict, pred_labels: dict, tasks_list: list) -> np.ndarray:
    """
    Build a multi-class confusion matrix with shape (N_tasks, N_tasks).
    conf_mat[i, j] = count of samples with true task i and predicted task j.
    Only compares overlapping (video_id, second) keys.
    """
    task_to_idx = {task: idx for idx, task in enumerate(tasks_list)}
    n_tasks = len(tasks_list)
    conf_mat = np.zeros((n_tasks, n_tasks), dtype=int)
    
    common_keys = set(gt_labels.keys()) & set(pred_labels.keys())
    for key in common_keys:
        gt_activity = gt_labels[key]
        pred_activity = pred_labels[key]
        if gt_activity in task_to_idx and pred_activity in task_to_idx:
            conf_mat[task_to_idx[gt_activity], task_to_idx[pred_activity]] += 1
    return conf_mat

def compute_classification_metrics_from_confusion(conf_mat: np.ndarray, tasks_list: list) -> tuple:
    """
    Compute overall accuracy and per-task precision, recall, and support from the confusion matrix.
    Returns overall_accuracy and a list of per-task metrics.
    """
    total_samples = conf_mat.sum()
    overall_accuracy = np.trace(conf_mat) / total_samples if total_samples > 0 else 0.0
    
    per_task_metrics = []
    for idx, task in enumerate(tasks_list):
        tp = conf_mat[idx, idx]
        fn = conf_mat[idx, :].sum() - tp
        fp = conf_mat[:, idx].sum() - tp
        support = conf_mat[idx, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_task_metrics.append({
            "Task": task,
            "Precision": precision,
            "Recall": recall,
            "Support": support
        })
    
    return overall_accuracy, per_task_metrics

def analyze_active_inactive_per_task_total_mape(gt_df: pd.DataFrame, pred_df: pd.DataFrame, exclude_tasks: set = None):
    """
    Compute and display active vs inactive time per task and overall MAPE.
    Saves a CSV table and a stacked bar chart.
    """
    if exclude_tasks is None:
        exclude_tasks = set()
    
    # Compute GT and Pred aggregates by Activity
    gt_group = gt_df.groupby("Activity").agg(GT_Active=("Active_seconds", "sum"),
                                             GT_Inactive=("Inactive_seconds", "sum")).reset_index()
    pred_group = pred_df.groupby("Activity").agg(Pred_Active=("Active_seconds", "sum"),
                                                 Pred_Inactive=("Inactive_seconds", "sum")).reset_index()
    
    merged = pd.merge(gt_group, pred_group, on="Activity", how="outer").fillna(0)
    merged["GT_Total"] = merged["GT_Active"] + merged["GT_Inactive"]
    merged["Pred_Total"] = merged["Pred_Active"] + merged["Pred_Inactive"]
    
    # Calculate per-task MAPE for total time
    merged["Task_time_MAPE"] = merged.apply(
        lambda row: abs(row["GT_Total"] - row["Pred_Total"]) / row["GT_Total"] * 100 if row["GT_Total"] > 0 else 0,
        axis=1
    )
    
    # Add overall 'ALL' row
    total_gt_active = merged["GT_Active"].sum()
    total_gt_inactive = merged["GT_Inactive"].sum()
    total_pred_active = merged["Pred_Active"].sum()
    total_pred_inactive = merged["Pred_Inactive"].sum()
    
    overall_gt_total = total_gt_active + total_gt_inactive
    overall_pred_total = total_pred_active + total_pred_inactive
    overall_time_mape = abs(overall_gt_total - overall_pred_total) / overall_gt_total * 100 if overall_gt_total > 0 else 0
    
    all_row = pd.DataFrame([{
        "Activity": "ALL",
        "GT_Active": total_gt_active,
        "GT_Inactive": total_gt_inactive,
        "Pred_Active": total_pred_active,
        "Pred_Inactive": total_pred_inactive,
        "GT_Total": overall_gt_total,
        "Pred_Total": overall_pred_total,
        "Task_time_MAPE": overall_time_mape
    }])
    merged = pd.concat([merged, all_row], ignore_index=True)
    
    # Save and display the table
    os.makedirs("results", exist_ok=True)
#     merged.to_csv("results/active_inactive_per_task_total_mape.csv", index=False)
#     print("\n[Active vs Inactive per Task (Total MAPE) - Table saved: results/active_inactive_per_task_total_mape.csv]")
#     display(merged)
    
    # Prepare data for plotting (excluding specified tasks and the 'ALL' row)
    plot_df = merged[~merged["Activity"].isin(exclude_tasks)].copy()
    plot_df = plot_df[plot_df["Activity"] != "ALL"]
    plot_df.sort_values(by="Activity", inplace=True)
    
    tasks_plot = plot_df["Activity"].tolist()
    gt_active_vals = plot_df["GT_Active"].values
    gt_inactive_vals = plot_df["GT_Inactive"].values
    pred_active_vals = plot_df["Pred_Active"].values
    pred_inactive_vals = plot_df["Pred_Inactive"].values
    
    x = np.arange(len(tasks_plot))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, gt_active_vals, width, label="GT Active")
    plt.bar(x - width/2, gt_inactive_vals, width, bottom=gt_active_vals, label="GT Inactive")
    plt.bar(x + width/2, pred_active_vals, width, label="Pred Active")
    plt.bar(x + width/2, pred_inactive_vals, width, bottom=pred_active_vals, label="Pred Inactive")
    
    plt.xticks(x, tasks_plot, rotation=45)
    plt.ylabel("Time (seconds)")
    plt.title("Active vs Inactive per Task\n(Total-Time MAPE only)")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("results/plots", exist_ok=True)
    plot_out = "results/plots/active_inactive_per_task_stacked_total_mape.png"
    plt.savefig(plot_out)
    plt.show()
    print(f"[Plot saved: {plot_out}]")

##############################################################################
# Main Function
##############################################################################
def main():
    # Ensure necessary folders exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    folder = "consolidated"
    gt_files = glob.glob(os.path.join(folder, "*_MANUALLYLABELED.csv"))
    
    gt_all_df = pd.DataFrame()
    pred_all_df = pd.DataFrame()
    
    # Read and concatenate all GT and corresponding Pred CSV files
    for gt_file in gt_files:
        base_name = os.path.basename(gt_file)
        video_id = base_name.replace("_MANUALLYLABELED.csv", "")
        pred_file = os.path.join(folder, f"{video_id}_parsed.csv")
        
        if os.path.exists(pred_file):
            gt_df = pd.read_csv(gt_file)
            pred_df = pd.read_csv(pred_file)
            
            # Clean and convert columns
            gt_df["Activity"] = gt_df["Activity"].fillna("Unknown").astype(str)
            pred_df["Activity"] = pred_df["Activity"].fillna("Unknown").astype(str)
            for col in ["Active_seconds", "Inactive_seconds"]:
                gt_df[col] = pd.to_numeric(gt_df[col], errors="coerce").fillna(0)
                pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").fillna(0)
            gt_df["time_start"] = pd.to_numeric(gt_df["time_start"], errors="coerce").fillna(0)
            pred_df["time_start"] = pd.to_numeric(pred_df["time_start"], errors="coerce").fillna(0)
            
            gt_df["video_id"] = video_id
            pred_df["video_id"] = video_id
            
            gt_all_df = pd.concat([gt_all_df, gt_df], ignore_index=True)
            pred_all_df = pd.concat([pred_all_df, pred_df], ignore_index=True)
        else:
            print(f"Prediction file for '{video_id}' not found.")
    
    if gt_all_df.empty or pred_all_df.empty:
        print("No data found for processing.")
        return
    
    ############################################################################
    # Dataset-Level Block Metrics: Transitions, Time, and MAPE
    ############################################################################
    # Compute transitions for GT and Pred
    all_gt_transitions = []
    all_pred_transitions = []
    for video_id, sub_gt in gt_all_df.groupby("video_id"):
        all_gt_transitions.extend(detect_transitions_blocks(sub_gt))
    for video_id, sub_pred in pred_all_df.groupby("video_id"):
        all_pred_transitions.extend(detect_transitions_blocks(sub_pred))
    
    total_gt_transitions = len(all_gt_transitions)
    total_pred_transitions = len(all_pred_transitions)
    matched_transitions = match_transitions_with_tolerance(all_gt_transitions, all_pred_transitions, delta=3)
    transition_sensitivity = matched_transitions / total_gt_transitions if total_gt_transitions > 0 else 0
    
    # Compute time per task for GT and Pred
    gt_time_per_task = compute_time_by_task(gt_all_df)
    pred_time_per_task = compute_time_by_task(pred_all_df)
    tasks_union = set(gt_time_per_task.keys()).union(set(pred_time_per_task.keys()))
    
    ape_list = []
    for task in tasks_union:
        gt_time = gt_time_per_task.get(task, 0)
        pred_time = pred_time_per_task.get(task, 0)
        if gt_time > 0:
            ape_list.append(abs(gt_time - pred_time) / gt_time)
    overall_mape_task = np.mean(ape_list) * 100 if ape_list else 0
    
    # Compute overall active and inactive times and MAPE
    total_gt_active, total_gt_inactive = compute_active_inactive(gt_all_df)
    total_pred_active, total_pred_inactive = compute_active_inactive(pred_all_df)
    ai_mape = mean_absolute_percentage_error([total_gt_active, total_gt_inactive],
                                               [total_pred_active, total_pred_inactive])
    
    total_gt_time = (gt_all_df["Active_seconds"] + gt_all_df["Inactive_seconds"]).sum()
    total_pred_time = (pred_all_df["Active_seconds"] + pred_all_df["Inactive_seconds"]).sum()
    
    # Save aggregated dataset metrics
    data_rows = [{
        "Task": "ALL",
        "Total_GT_Time": total_gt_time,
        "Total_Pred_Time": total_pred_time,
        "Total_GT_Transitions": total_gt_transitions,
        "Total_Pred_Transitions": total_pred_transitions,
        "Matched_Transitions": matched_transitions,
        "Transition_Sensitivity": transition_sensitivity,
        "MAPE_TimePerTask": overall_mape_task,
        "MAPE_ActiveInactive": ai_mape
    }]
    for task in sorted(tasks_union):
        gt_val = gt_time_per_task.get(task, 0)
        pred_val = pred_time_per_task.get(task, 0)
        task_mape = abs(gt_val - pred_val) / gt_val * 100 if gt_val > 0 else 0
        data_rows.append({
            "Task": task,
            "Total_GT_Time": gt_val,
            "Total_Pred_Time": pred_val,
            "Total_GT_Transitions": np.nan,
            "Total_Pred_Transitions": np.nan,
            "Matched_Transitions": np.nan,
            "Transition_Sensitivity": np.nan,
            "MAPE_TimePerTask": task_mape,
            "MAPE_ActiveInactive": np.nan
        })
    aggregated_df = pd.DataFrame(data_rows)
    aggregated_df.to_csv("results/aggregated_dataset_metrics.csv", index=False)
    print("Dataset-level metrics saved to 'results/aggregated_dataset_metrics.csv'.")
    
    # Filter out tasks for further analysis
    exclude_tasks = {"END", "Unknown"}
    gt_filtered = gt_all_df[~gt_all_df["Activity"].isin(exclude_tasks)]
    pred_filtered = pred_all_df[~pred_all_df["Activity"].isin(exclude_tasks)]
    
    # Active/Inactive per task summary
    gt_task_summary = gt_filtered.groupby("Activity").agg(GT_Active=("Active_seconds", "sum"),
                                                          GT_Inactive=("Inactive_seconds", "sum")).reset_index()
    pred_task_summary = pred_filtered.groupby("Activity").agg(Pred_Active=("Active_seconds", "sum"),
                                                              Pred_Inactive=("Inactive_seconds", "sum")).reset_index()
    merged_tasks = pd.merge(gt_task_summary, pred_task_summary, on="Activity", how="outer").fillna(0)
    
    summary_rows = []
    for _, row in merged_tasks.iterrows():
        task = row["Activity"]
        # Active metrics
        gt_active = row["GT_Active"]
        pred_active = row["Pred_Active"]
        diff_active = gt_active - pred_active
        mape_active = abs(diff_active) / gt_active * 100 if gt_active > 0 else 0
        summary_rows.append({
            "Task": task,
            "Category": "Active",
            "GT_Time": gt_active,
            "Pred_Time": pred_active,
            "Difference": diff_active,
            "MAPE": mape_active
        })
        # Inactive metrics
        gt_inactive = row["GT_Inactive"]
        pred_inactive = row["Pred_Inactive"]
        diff_inactive = gt_inactive - pred_inactive
        mape_inactive = abs(diff_inactive) / gt_inactive * 100 if gt_inactive > 0 else 0
        summary_rows.append({
            "Task": task,
            "Category": "Inactive",
            "GT_Time": gt_inactive,
            "Pred_Time": pred_inactive,
            "Difference": diff_inactive,
            "MAPE": mape_inactive
        })
    active_inactive_summary_df = pd.DataFrame(summary_rows)
    
    # Overall aggregated active/inactive row
    total_gt_active = merged_tasks["GT_Active"].sum()
    total_pred_active = merged_tasks["Pred_Active"].sum()
    diff_overall_active = total_gt_active - total_pred_active
    mape_overall_active = abs(diff_overall_active) / total_gt_active * 100 if total_gt_active > 0 else 0
    total_gt_inactive = merged_tasks["GT_Inactive"].sum()
    total_pred_inactive = merged_tasks["Pred_Inactive"].sum()
    diff_overall_inactive = total_gt_inactive - total_pred_inactive
    mape_overall_inactive = abs(diff_overall_inactive) / total_gt_inactive * 100 if total_gt_inactive > 0 else 0
    
    overall_rows = [
        {"Task": "ALL", "Category": "Active", "GT_Time": total_gt_active, "Pred_Time": total_pred_active,
         "Difference": diff_overall_active, "MAPE": mape_overall_active},
        {"Task": "ALL", "Category": "Inactive", "GT_Time": total_gt_inactive, "Pred_Time": total_pred_inactive,
         "Difference": diff_overall_inactive, "MAPE": mape_overall_inactive}
    ]
    active_inactive_summary_df = pd.concat([active_inactive_summary_df, pd.DataFrame(overall_rows)], ignore_index=True)
    active_inactive_summary_df.sort_values(by=["Task", "Category"], inplace=True)
    active_inactive_summary_df.to_csv("results/active_inactive_per_task_summary.csv", index=False)
    print("\n[Active vs Inactive Metrics per Task with ALL row]")
    display(active_inactive_summary_df)
    active_inactive_summary_df.to_csv('results/active_vs_inactive_all.csv')
    
    # Aggregated Time per Task Plot Data
    plot_tasks = [task for task in sorted(tasks_union) if task not in exclude_tasks]
    gt_plot_vals = [gt_time_per_task.get(task, 0) for task in plot_tasks]
    pred_plot_vals = [pred_time_per_task.get(task, 0) for task in plot_tasks]
    
    time_df = pd.DataFrame({
        "Task": plot_tasks,
        "GT_Time": gt_plot_vals,
        "Pred_Time": pred_plot_vals
    })
    time_df["Difference"] = time_df["GT_Time"] - time_df["Pred_Time"]
    time_df["MAPE"] = time_df.apply(lambda row: abs(row["GT_Time"] - row["Pred_Time"]) / row["GT_Time"] * 100 if row["GT_Time"] > 0 else 0, axis=1)
    
    overall_GT_Time = time_df["GT_Time"].sum()
    overall_Pred_Time = time_df["Pred_Time"].sum()
    overall_MAPE = abs(overall_GT_Time - overall_Pred_Time) / overall_GT_Time * 100 if overall_GT_Time > 0 else 0
    
    overall_row = pd.DataFrame([{
        "Task": "ALL",
        "GT_Time": overall_GT_Time,
        "Pred_Time": overall_Pred_Time,
        "Difference": overall_GT_Time - overall_Pred_Time,
        "MAPE": overall_MAPE
    }])
    time_df = pd.concat([time_df, overall_row], ignore_index=True)
    time_df.to_csv("results/time_per_task_plot_data_with_mape.csv", index=False)
    print("\n[Time per Task Plot Data with MAPE]")
    display(time_df)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_tasks))
    width = 0.35
    plt.bar(x - width/2, gt_plot_vals, width, label="Ground Truth", color="#ccf4a5")
    plt.bar(x + width/2, pred_plot_vals, width, label="Model Identified", color="#aacff5")
    plt.xticks(x, plot_tasks, rotation=45)
    plt.ylabel("Total Time (s)")
    plt.title(f"Aggregate Time per Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/aggregated_time_per_task.png")
    plt.show()

    # Active vs Inactive Aggregated Plot Data
    categories = ["Active", "Inactive"]
    gt_ai_values = [total_gt_active, total_gt_inactive]
    pred_ai_values = [total_pred_active, total_pred_inactive]

    ai_df = pd.DataFrame({
        "Category": categories,
        "GT_Time": gt_ai_values,
        "Pred_Time": pred_ai_values
    })
    ai_df["Difference"] = ai_df["GT_Time"] - ai_df["Pred_Time"]
    ai_df.to_csv("results/active_inactive_plot_data.csv", index=False)
    print("\n[Active vs Inactive Plot Data]")
    display(ai_df)

    plt.figure(figsize=(6, 4))
    x_ai = np.arange(len(categories))
    plt.bar(x_ai - width/2, gt_ai_values, width, label="Ground Truth", color="#ccf4a5")
    plt.bar(x_ai + width/2, pred_ai_values, width, label="Model Identified", color="#aacff5")
    plt.xticks(x_ai, categories)
    plt.ylabel("Total Time (s)")
    plt.title(f"Aggregate Active vs Inactive\nMAPE={ai_mape:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/active_inactive_aggregated.png")
    plt.show()



    # Existing transitions table
    trans_plot_df = pd.DataFrame({
        "Type": ["GT_Transitions", "Pred_Transitions", "Matched_Transitions"],
        "Value": [total_gt_transitions, total_pred_transitions, matched_transitions]
    })
    trans_plot_df.to_csv("results/transitions_plot_data.csv", index=False)
    print("\n[Transitions Plot Data]")
    display(trans_plot_df)

    plt.figure(figsize=(6, 4))
    plt.bar(trans_plot_df["Type"], trans_plot_df["Value"], color=["blue", "orange", "green"])
    plt.ylabel("Count")
    plt.title(f"Dataset-level Transitions\nSensitivity={transition_sensitivity:.2f}")
    plt.tight_layout()
    plt.savefig("results/plots/transitions_aggregated.png")
    plt.show()

    # Compute confusion matrix components
    TP = matched_transitions
    FP = total_pred_transitions - matched_transitions
    FN = total_gt_transitions - matched_transitions

    # Create a table for confusion matrix metrics
    conf_matrix_df = pd.DataFrame({
        "Metric": ["True Positives", "False Positives", "False Negatives"],
        "Count": [TP, FP, FN]
    })
    conf_matrix_df.to_csv("results/confusion_matrix_data.csv", index=False)
    print("\n[Confusion Matrix Data]")
    display(conf_matrix_df)

    # Plotting a confusion matrix heatmap (with TN set to 0 for demonstration)
    # Rows: Actual; Columns: Predicted. TN is not applicable.
    cm = np.array([[TP, FN],
                   [FP, 0]])

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Positive", "Predicted Negative"],
                yticklabels=["Actual Positive", "Actual Negative"])
    plt.title("Confusion Matrix (TN not applicable)")
    plt.tight_layout()
    plt.savefig("results/plots/confusion_matrix.png")
    plt.show()

    ############################################################################
    # Frame-Level Classification Metrics and Plots
    ############################################################################
    print("\nComputing frame-level (second-level) classification metrics...")
    gt_all_df["Total_time"] = gt_all_df["Active_seconds"] + gt_all_df["Inactive_seconds"]
    pred_all_df["Total_time"] = pred_all_df["Active_seconds"] + pred_all_df["Inactive_seconds"]
    
    gt_label_dict = build_frame_level_labels(gt_all_df)
    pred_label_dict = build_frame_level_labels(pred_all_df)
    
    tasks_frame_union = set(gt_all_df["Activity"].unique()).union(pred_all_df["Activity"].unique())
    tasks_frame_union = [task for task in tasks_frame_union if task not in exclude_tasks]
    tasks_frame_union = sorted(tasks_frame_union)
    
    if not tasks_frame_union:
        print("No valid tasks for frame-level metrics.")
        return
    
    conf_mat = compute_confusion_matrix(gt_label_dict, pred_label_dict, tasks_frame_union)
    overall_accuracy, per_task_metrics = compute_classification_metrics_from_confusion(conf_mat, tasks_frame_union)
    
    # Build frame-level metrics DataFrame
    frame_metrics_rows = []
    for metric in per_task_metrics:
        frame_metrics_rows.append({
            "Task": metric["Task"],
            "Precision": metric["Precision"],
            "Recall": metric["Recall"],
            "Support": metric["Support"]
        })
    frame_metrics_df = pd.DataFrame(frame_metrics_rows)
    frame_metrics_df.to_csv("results/frame_level_metrics.csv", index=False)
    print("\n[Frame-level Classification Metrics]")
    display(frame_metrics_df)
    
    plt.figure(figsize=(10, 6))
    x_vals = np.arange(len(frame_metrics_df.dropna(subset=["Precision"])))
    width = 0.35
    pr_filtered = frame_metrics_df.dropna(subset=["Precision"]).reset_index(drop=True)
    plt.bar(x_vals - width/2, pr_filtered["Precision"], width, label="Precision")
    plt.bar(x_vals + width/2, pr_filtered["Recall"], width, label="Recall")
    plt.xticks(x_vals, pr_filtered["Task"], rotation=45)
    plt.ylim(0, 1)
    plt.title(f"Precision & Recall per Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/frame_level_precision_recall_per_task.png")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=tasks_frame_union, yticklabels=tasks_frame_union)
    plt.xlabel("Model Identified Task")
    plt.ylabel("Ground Truth Annotation Task")
    plt.title(f"Task Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/plots/task_confusion_matrix.png")
    plt.show()
    
    conf_df = pd.DataFrame(conf_mat, index=tasks_frame_union, columns=tasks_frame_union)
    conf_df.to_csv("results/frame_level_confusion_matrix_table.csv")
    print("\n[Frame-level Confusion Matrix Table]")
    display(conf_df)
    
    analyze_active_inactive_per_task_total_mape(gt_all_df, pred_all_df, exclude_tasks={"END", "Unknown"})

if __name__ == "__main__":
    main()


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

##############################################################################
# Helper Functions for Transitions Analysis
##############################################################################
def detect_transitions_blocks(df: pd.DataFrame) -> list:
    """
    Detect transitions between rows based on the 'Activity' column.
    Returns a list of time_start values for rows where the task changes.
    """
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]["Activity"] != df.iloc[i - 1]["Activity"]:
            transitions.append(df.iloc[i]["time_start"])
    return transitions

def match_transitions_with_tolerance_detailed(gt_times: list, pred_times: list, delta: int = 2) -> (int, list):
    """
    Match predicted transitions to ground truth with tolerance window (detailed version).

    For each ground truth transition time, any predicted transition time within
    [t - delta, t + delta] is considered a match.

    Parameters:
        gt_times: Ground truth transition times
        pred_times: Predicted transition times
        delta (int): Tolerance in frames (default: 2, per paper Section 3.2.2)

    Returns:
        tuple: (matched_count, time_differences)
    """
    matched_count = 0
    used_pred = set()
    time_diffs = []
    # Sort the lists to ensure proper matching
    gt_times_sorted = sorted(gt_times)
    pred_times_sorted = sorted(pred_times)
    for gt_time in gt_times_sorted:
        low_bound = gt_time - delta
        high_bound = gt_time + delta
        # Find the first available candidate within the tolerance window
        candidates = [pt for pt in pred_times_sorted if low_bound <= pt <= high_bound and pt not in used_pred]
        if candidates:
            matched = candidates[0]
            matched_count += 1
            used_pred.add(matched)
            time_diffs.append(abs(gt_time - matched))
    return matched_count, time_diffs

##############################################################################
# Main Function for Transitions Analysis with Specificity and Visualization
##############################################################################
def main():
    # Ensure necessary folders exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    folder = "consolidated"
    gt_files = glob.glob(os.path.join(folder, "*_MANUALLYLABELED.csv"))
    
    # List to store per-video information
    video_results = []
    
    # Process each video file to extract duration and transitions
    for gt_file in gt_files:
        base_name = os.path.basename(gt_file)
        video_id = base_name.replace("_MANUALLYLABELED.csv", "")
        pred_file = os.path.join(folder, f"{video_id}_parsed.csv")
        
        if os.path.exists(pred_file):
            gt_df = pd.read_csv(gt_file)
            pred_df = pd.read_csv(pred_file)
            
            # Clean up relevant columns
            gt_df["Activity"] = gt_df["Activity"].fillna("Unknown").astype(str)
            pred_df["Activity"] = pred_df["Activity"].fillna("Unknown").astype(str)
            gt_df["time_start"] = pd.to_numeric(gt_df["time_start"], errors="coerce").fillna(0)
            pred_df["time_start"] = pd.to_numeric(pred_df["time_start"], errors="coerce").fillna(0)
            
            # Extract video duration from the row with Activity 'END'
            end_rows = gt_df[gt_df["Activity"] == "END"]
            if not end_rows.empty:
                # Assuming the first occurrence gives the duration
                video_duration = end_rows.iloc[0]["time_start"]
            else:
                # Fallback: use the maximum time_start value if no 'END' row is present
                video_duration = gt_df["time_start"].max()
            
            # Get transitions from GT and prediction
            gt_transitions = detect_transitions_blocks(gt_df)
            pred_transitions = detect_transitions_blocks(pred_df)
            
            video_results.append({
                "video_id": video_id,
                "duration": video_duration,
                "gt_transitions": gt_transitions,
                "pred_transitions": pred_transitions
            })
        else:
            print(f"Prediction file for '{video_id}' not found.")
    
    if not video_results:
        print("No videos found for analysis.")
        return
    
    # Define multiple tolerance thresholds (in seconds) to evaluate
    tolerance_thresholds = [0, 1, 2, 3]
    aggregated_results = []
    
    for delta in tolerance_thresholds:
        total_gt_transitions = 0
        total_pred_transitions = 0
        total_matched = 0
        all_time_diffs = []
        sum_TN = 0
        sum_negatives = 0  # TN + FP
        
        # Process each video's transitions
        for vid in video_results:
            gt_times = vid["gt_transitions"]
            pred_times = vid["pred_transitions"]
            duration = vid["duration"]
            
            # Compute counts
            num_gt = len(gt_times)
            num_pred = len(pred_times)
            matched_count, time_diffs = match_transitions_with_tolerance_detailed(gt_times, pred_times, delta=delta)
            
            # For specificity:
            # False Positives: predicted transitions that do not match GT transitions
            FP = num_pred - matched_count
            # Total negatives = potential seconds without transitions = duration - number of GT transitions
            negatives = duration - num_gt  
            # True Negatives = negatives - FP
            TN = negatives - FP if negatives - FP >= 0 else 0  # ensure non-negative
            
            total_gt_transitions += num_gt
            total_pred_transitions += num_pred
            total_matched += matched_count
            all_time_diffs.extend(time_diffs)
            sum_TN += TN
            sum_negatives += negatives
        
        sensitivity = total_matched / total_gt_transitions if total_gt_transitions > 0 else np.nan
        avg_time_error = np.mean(all_time_diffs) if all_time_diffs else np.nan
        FP_total = total_pred_transitions - total_matched
        specificity = (sum_TN / sum_negatives) if sum_negatives > 0 else np.nan
        
        aggregated_results.append({
            "Tolerance": delta,
            "Total_GT_Transitions": total_gt_transitions,
            "Total_Pred_Transitions": total_pred_transitions,
            "Total_Matched": total_matched,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Avg_Time_Error": avg_time_error,
            "All_Time_Diffs": all_time_diffs
        })
    
    results_df = pd.DataFrame(aggregated_results)
    results_df.to_csv("results/transitions_analysis_with_specificity.csv", index=False)
    print("Transitions analysis data (with specificity) saved to 'results/transitions_analysis_with_specificity.csv'.")
    display(results_df)
    
    # Plot 1: Matched Transitions vs. Tolerance Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["Tolerance"], results_df["Total_Matched"], marker="o", label="Matched Transitions")
    plt.plot(results_df["Tolerance"], results_df["Total_GT_Transitions"], linestyle="--", color="gray", label="Total GT Transitions")
    plt.xlabel("Tolerance (seconds)")
    plt.ylabel("Number of Transitions")
    plt.title("Matched Transitions vs. Tolerance Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/matched_transitions_vs_tolerance.png")
    plt.show()
    
    # Plot 2: Sensitivity vs. Tolerance Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["Tolerance"], results_df["Sensitivity"], marker="o", color="green")
    plt.xlabel("Tolerance (seconds)")
    plt.ylabel("Sensitivity")
    plt.title("Transition Detection Sensitivity vs. Tolerance")
    plt.tight_layout()
    plt.savefig("results/plots/sensitivity_vs_tolerance.png")
    plt.show()
    
    # Plot 3: Specificity vs. Tolerance Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["Tolerance"], results_df["Specificity"], marker="o", color="purple")
    plt.xlabel("Tolerance (seconds)")
    plt.ylabel("Specificity")
    plt.title("Transition Detection Specificity vs. Tolerance")
    plt.tight_layout()
    plt.savefig("results/plots/specificity_vs_tolerance.png")
    plt.show()
    
    # Plot 4: Average Time Error vs. Tolerance Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["Tolerance"], results_df["Avg_Time_Error"], marker="o", color="red")
    plt.xlabel("Tolerance (seconds)")
    plt.ylabel("Average Time Error (seconds)")
    plt.title("Average Time Error of Matched Transitions vs. Tolerance")
    plt.tight_layout()
    plt.savefig("results/plots/avg_time_error_vs_tolerance.png")
    plt.show()
    
    # Plot 5: Histogram of Time Errors for each Tolerance Threshold
    num_thresholds = len(tolerance_thresholds)
    fig, axs = plt.subplots(num_thresholds, 1, figsize=(8, 4 * num_thresholds), sharex=True)
    for idx, delta in enumerate(tolerance_thresholds):
        time_diffs = results_df.loc[results_df["Tolerance"] == delta, "All_Time_Diffs"].values[0]
        if time_diffs:
            axs[idx].hist(time_diffs, bins=10, color="skyblue", edgecolor="black")
            axs[idx].set_title(f"Histogram of Time Errors (Tolerance = {delta} sec)")
            axs[idx].set_xlabel("Time Error (seconds)")
            axs[idx].set_ylabel("Frequency")
        else:
            axs[idx].text(0.5, 0.5, "No matched transitions", horizontalalignment="center", verticalalignment="center")
            axs[idx].set_title(f"Tolerance = {delta} sec")
    plt.tight_layout()
    plt.savefig("results/plots/time_error_histograms.png")
    plt.show()

if __name__ == "__main__":
    main()


# ## NEW

# In[2]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################

def read_model_and_gt_files(consolidated_folder="consolidated"):
    """
    Reads all pairs of CSV files in 'consolidated_folder' that match the pattern:
      <prefix>_parsed.csv  (model output)
      <prefix>_MANUALLYLABELED.csv  (ground truth)
    Each file has columns:
       Activity, Active_seconds, Inactive_seconds [optionally Total_seconds]
    Returns a list of tuples: [(df_pred, df_gt, prefix), ...]
    """
    parsed_files = sorted(glob.glob(os.path.join(consolidated_folder, "*_parsed.csv")))
    data_pairs = []
    
    for parsed_fp in parsed_files:
        prefix = os.path.basename(parsed_fp).replace("_parsed.csv", "")
        gt_fp = os.path.join(consolidated_folder, f"{prefix}_MANUALLYLABELED.csv")
        
        if not os.path.exists(gt_fp):
            print(f"Warning: Ground truth file not found for {parsed_fp}")
            continue
        
        df_pred = pd.read_csv(parsed_fp)
        df_gt   = pd.read_csv(gt_fp)
        
        # Check for minimal required columns
        for col in ["Activity", "Active_seconds", "Inactive_seconds"]:
            if col not in df_pred.columns:
                raise ValueError(f"Missing '{col}' in {parsed_fp}")
            if col not in df_gt.columns:
                raise ValueError(f"Missing '{col}' in {gt_fp}")
        
        # If "Total_seconds" not present, create it
        if "Total_seconds" not in df_pred.columns:
            df_pred["Total_seconds"] = df_pred["Active_seconds"] + df_pred["Inactive_seconds"]
        if "Total_seconds" not in df_gt.columns:
            df_gt["Total_seconds"] = df_gt["Active_seconds"] + df_gt["Inactive_seconds"]
        
        data_pairs.append((df_pred, df_gt, prefix))
    
    return data_pairs


def match_tasks(df_pred, df_gt, task_col="Activity"):
    """
    1) Aggregate each DataFrame by Activity (sum) => one row per Activity
       (in case there are duplicates).
    2) Merge them on 'Activity'
    Returns a DataFrame with columns:
       Activity, pred_active, pred_inactive, pred_total,
                 gt_active,   gt_inactive,   gt_total
    for the time-based metrics (MAE, MAPE, etc.).
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

###############################################################################
# 2. TRANSITION DETECTION METRICS
###############################################################################

def compute_transition_detection_metrics(df_pred, df_gt, task_col="Activity"):
    """
    Because each CSV is assumed to be in CHRONOLOGICAL order, each row i 
    corresponds to a segment in time with an 'Activity'.

    A "transition" is thus going from row i's Activity to row i+1's Activity.

    We'll:
      - Extract the list of consecutive transitions from df_gt
      - Extract the list of consecutive transitions from df_pred
      - Use a set-based approach: 
          GT_transitions = set of (Act_i -> Act_(i+1))
          Pred_transitions = set of (PAct_i -> PAct_(i+1))
        Then:
          TP = number of transitions in the intersection
          FP = transitions in predicted but not in GT
          FN = transitions in GT but not in predicted
        Precision = TP / (TP+FP)
        Recall = TP / (TP+FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    # Build transitions from GT
    transitions_gt = []
    for i in range(len(df_gt)-1):
        curr_act = df_gt[task_col].iloc[i]
        next_act = df_gt[task_col].iloc[i+1]
        # If the next row is a truly new task, we count that as a boundary
        # (some data might contain repeated tasks in consecutive rows).
        transitions_gt.append((curr_act, next_act))
    
    # Build transitions from Pred
    transitions_pred = []
    for i in range(len(df_pred)-1):
        curr_act = df_pred[task_col].iloc[i]
        next_act = df_pred[task_col].iloc[i+1]
        transitions_pred.append((curr_act, next_act))
    
    set_gt   = set(transitions_gt)
    set_pred = set(transitions_pred)
    
    tp = len(set_gt.intersection(set_pred))
    fp = len(set_pred - set_gt)
    fn = len(set_gt - set_pred)
    
    precision = tp / (tp + fp) if (tp+fp)>0 else 0
    recall    = tp / (tp + fn) if (tp+fn)>0 else 0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0
    
    results = {
        "num_GT_transitions": len(set_gt),
        "num_Pred_transitions": len(set_pred),
        "TP_transitions": tp,
        "FP_transitions": fp,
        "FN_transitions": fn,
        "Transition_Precision": precision,
        "Transition_Recall": recall,
        "Transition_F1": f1
    }
    return results


###############################################################################
# 3. TIME-BASED CONFUSION MATRIX (APPROX) AT AGGREGATE LEVEL
###############################################################################

def approximate_time_based_confusion_matrix(df, task_col="Activity"):
    """
    We want a 'time-based confusion matrix' across tasks. With purely aggregated data
    (X seconds for Task A, etc.), we cannot know how many of A's ground-truth seconds
    were predicted as B, C, etc.

    => We'll produce a matrix with tasks in rows (GT) and tasks in columns (Pred),
       placing min(gt_total, pred_total) on the diagonal, 0 off-diagonal,
       as an approximate demonstration.
    """
    tasks_all = sorted(df[task_col].unique())
    
    cm = np.zeros((len(tasks_all), len(tasks_all)))
    
    for i, gt_task in enumerate(tasks_all):
        row = df.loc[df[task_col] == gt_task]
        if row.empty:
            continue
        gt_val   = float(row["gt_total"].iloc[0])
        pred_val = float(row["pred_total"].iloc[0])
        
        for j, pred_task in enumerate(tasks_all):
            if gt_task == pred_task:
                cm[i, j] = min(gt_val, pred_val)
            else:
                cm[i, j] = 0.0
    
    df_cm = pd.DataFrame(cm, index=tasks_all, columns=tasks_all)
    return df_cm

###############################################################################
# 4. TASK CLASSIFICATION METRICS (AGGREGATED)
###############################################################################

def compute_task_classification_metrics_aggregated(df, task_col="Activity"):
    """
    With aggregated times per task, we approximate:
       TruePositive (TP) = min(gt_total, pred_total) for that task
       FalseNegative (FN)= (gt_total - TP) if positive
       FalsePositive (FP)= (pred_total - TP) if positive
    => Then compute precision, recall, F1 per task + micro-average.
    """
    tasks = df[task_col].unique()
    
    results = []
    total_TP=0; total_FP=0; total_FN=0
    
    for t in tasks:
        row = df.loc[df[task_col]==t].iloc[0]
        gt_total   = row["gt_total"]
        pred_total = row["pred_total"]
        
        overlap = min(gt_total, pred_total)
        TP = overlap
        FN = max(0, gt_total - overlap)
        FP = max(0, pred_total - overlap)
        
        precision_t = TP / (TP+FP) if (TP+FP)>0 else 0
        recall_t    = TP / (TP+FN) if (TP+FN)>0 else 0
        f1_t        = (2*precision_t*recall_t/(precision_t+recall_t)
                       if (precision_t+recall_t)>0 else 0)
        
        results.append({
            "Activity":t,
            "TP_sec":TP, "FP_sec":FP, "FN_sec":FN,
            "Precision":precision_t,
            "Recall":recall_t,
            "F1-score":f1_t,
            "gt_total":gt_total,
            "pred_total":pred_total
        })
        
        total_TP += TP
        total_FP += FP
        total_FN += FN
    
    micro_precision = total_TP/(total_TP+total_FP) if (total_TP+total_FP)>0 else 0
    micro_recall    = total_TP/(total_TP+total_FN) if (total_TP+total_FN)>0 else 0
    micro_f1        = (2*micro_precision*micro_recall/(micro_precision+micro_recall)
                       if (micro_precision+micro_recall)>0 else 0)
    
    df_metrics = pd.DataFrame(results).sort_values("Activity").reset_index(drop=True)
    df_metrics.loc[len(df_metrics)] = {
        "Activity":"MICRO-AVERAGE",
        "TP_sec":total_TP, "FP_sec":total_FP, "FN_sec":total_FN,
        "Precision":micro_precision,
        "Recall":micro_recall,
        "F1-score":micro_f1,
        "gt_total":df_metrics["gt_total"].sum(),
        "pred_total":df_metrics["pred_total"].sum()
    }
    return df_metrics

###############################################################################
# 5. TIME DEVIATIONS OR ERRORS IN TIME SPENT PER TASK
###############################################################################

def compute_time_deviation_metrics(df, task_col="Activity"):
    """
    Compute MAE, MAPE for total time, active time, inactive time across tasks.
    Also compute correlation on total times across tasks.
    
    Returns:
      metric_summary -> dict with overall metrics
      df_out -> the df with extra columns for per-task errors 
    """
    df_out = df.copy()
    
    # total
    df_out["gt_total_eps"] = df_out["gt_total"].replace({0:1e-9})
    df_out["abs_error_total"] = (df_out["pred_total"] - df_out["gt_total"]).abs()
    df_out["pct_error_total"] = df_out["abs_error_total"]/df_out["gt_total_eps"]
    mae_total  = df_out["abs_error_total"].mean()
    mape_total = df_out["pct_error_total"].mean()*100
    
    # active
    df_out["gt_active_eps"] = df_out["gt_active"].replace({0:1e-9})
    df_out["abs_error_active"] = (df_out["pred_active"] - df_out["gt_active"]).abs()
    df_out["pct_error_active"] = df_out["abs_error_active"] / df_out["gt_active_eps"]
    mae_active  = df_out["abs_error_active"].mean()
    mape_active = df_out["pct_error_active"].mean()*100
    
    # inactive
    df_out["gt_inactive_eps"] = df_out["gt_inactive"].replace({0:1e-9})
    df_out["abs_error_inactive"] = (df_out["pred_inactive"] - df_out["gt_inactive"]).abs()
    df_out["pct_error_inactive"] = df_out["abs_error_inactive"] / df_out["gt_inactive_eps"]
    mae_inactive  = df_out["abs_error_inactive"].mean()
    mape_inactive = df_out["pct_error_inactive"].mean()*100
    
    # correlation
    if len(df_out)>1:
        corr_pearson, _  = pearsonr(df_out["pred_total"], df_out["gt_total"])
        corr_spearman, _ = spearmanr(df_out["pred_total"], df_out["gt_total"])
    else:
        corr_pearson, corr_spearman = (np.nan, np.nan)
    
    metric_summary = {
        "MAE_Total": mae_total,
        "MAPE_Total": mape_total,
        "MAE_Active": mae_active,
        "MAPE_Active": mape_active,
        "MAE_Inactive": mae_inactive,
        "MAPE_Inactive": mape_inactive,
        "Corr_Pearson_Total": corr_pearson,
        "Corr_Spearman_Total": corr_spearman
    }
    
    return metric_summary, df_out

###############################################################################
# 6. ACTIVITY (ACTIVE vs. INACTIVE) DETECTION METRICS
###############################################################################

def compute_activity_detection_metrics_aggregated(df):
    """
    We sum predicted active vs. ground-truth active, etc. 
    Then approximate TP by the overlap of pred_active & gt_active, etc.
    
    Also we can compute an approximate 'overall accuracy' = 
       (overlap_active + overlap_inactive)/(total_gt_active+total_gt_inactive)
    """
    total_gt_active   = df["gt_active"].sum()
    total_gt_inactive = df["gt_inactive"].sum()
    total_pred_active = df["pred_active"].sum()
    total_pred_inactive = df["pred_inactive"].sum()
    
    overlap_active   = min(total_gt_active, total_pred_active)
    overlap_inactive = min(total_gt_inactive, total_pred_inactive)
    
    # active as positive class
    TP_active = overlap_active
    FP_active = max(0, total_pred_active - overlap_active)
    FN_active = max(0, total_gt_active - overlap_active)
    
    prec_active = TP_active/(TP_active+FP_active) if (TP_active+FP_active)>0 else 0
    rec_active  = TP_active/(TP_active+FN_active) if (TP_active+FN_active)>0 else 0
    f1_active   = (2*prec_active*rec_active/(prec_active+rec_active)
                   if (prec_active+rec_active)>0 else 0)
    
    # inactive as positive class
    TP_inactive = overlap_inactive
    FP_inactive = max(0, total_pred_inactive - overlap_inactive)
    FN_inactive = max(0, total_gt_inactive - overlap_inactive)
    
    prec_inactive = TP_inactive/(TP_inactive+FP_inactive) if (TP_inactive+FP_inactive)>0 else 0
    rec_inactive  = TP_inactive/(TP_inactive+FN_inactive) if (TP_inactive+FN_inactive)>0 else 0
    f1_inactive   = (2*prec_inactive*rec_inactive/(prec_inactive+rec_inactive)
                     if (prec_inactive+rec_inactive)>0 else 0)
    
    total_gt_time = total_gt_active + total_gt_inactive
    total_overlap = overlap_active + overlap_inactive
    overall_accuracy = total_overlap/total_gt_time if total_gt_time>0 else 0
    
    results = {
        "Active_Precision": prec_active,
        "Active_Recall": rec_active,
        "Active_F1": f1_active,
        "Inactive_Precision": prec_inactive,
        "Inactive_Recall": rec_inactive,
        "Inactive_F1": f1_inactive,
        "Overall_Accuracy_ActiveInactive": overall_accuracy
    }
    return results

###############################################################################
# 7. MAIN SCRIPT: COMBINE EVERYTHING (PER‐TASK + OVERALL), INCLUDING TRANSITIONS
###############################################################################

def main():
    data_pairs = read_model_and_gt_files("consolidated")
    if not data_pairs:
        print("No valid pairs found in 'consolidated'.")
        return
    
    all_merged_for_time = []  # For time-based aggregated metrics
    transition_results_list = []  # For transition detection across each video
    
    for (df_pred, df_gt, prefix) in data_pairs:
        # 1) Compute transition detection for this single video
        #    We'll treat the raw rows themselves as a sequence of tasks.
        trans_res = compute_transition_detection_metrics(df_pred, df_gt, task_col="Activity")
        trans_res["video_prefix"] = prefix
        transition_results_list.append(trans_res)
        
        # 2) Merge tasks for time-based metrics
        df_merged = match_tasks(df_pred, df_gt)
        df_merged["video_prefix"] = prefix
        all_merged_for_time.append(df_merged)
    
    # Summarize transition detection across videos
    df_transition = pd.DataFrame(transition_results_list)
    print("\n================ TRANSITION DETECTION PER VIDEO ================\n")
    print(df_transition.to_string(index=False))
    
    # Optionally compute mean or micro-averages across videos
    # We'll do a simple mean of precision/recall/f1
    avg_precision = df_transition["Transition_Precision"].mean()
    avg_recall    = df_transition["Transition_Recall"].mean()
    avg_f1        = df_transition["Transition_F1"].mean()
    print("\n========== AVERAGE TRANSITION DETECTION ACROSS ALL VIDEOS ==========\n")
    print(f"Mean Precision: {avg_precision:.3f}")
    print(f"Mean Recall:    {avg_recall:.3f}")
    print(f"Mean F1:        {avg_f1:.3f}")
    
    # -----------------------------------------------------------------------
    # Combine all videos for time-based aggregated metrics
    # -----------------------------------------------------------------------
    df_all = pd.concat(all_merged_for_time, ignore_index=True)
    
    # Summation across videos => get single row per Activity
    df_agg_tasks = (
        df_all
        .groupby("Activity", as_index=False)
        .agg({
            "pred_active":"sum",
            "pred_inactive":"sum",
            "pred_total":"sum",
            "gt_active":"sum",
            "gt_inactive":"sum",
            "gt_total":"sum"
        })
    )
    
    ###########################################################################
    # (1) TASK CLASSIFICATION: TIME-BASED CONFUSION MATRIX (APPROX)
    ###########################################################################
    df_cm = approximate_time_based_confusion_matrix(df_agg_tasks, task_col="Activity")
    print("\n===== APPROX TIME-BASED CONFUSION MATRIX (AGGREGATE) =====\n")
    print(df_cm)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(df_cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Approx Time-Based Confusion Matrix (Aggregate)")
    plt.xlabel("Predicted Task")
    plt.ylabel("Ground Truth Task")
    plt.tight_layout()
    plt.show()
    
    ###########################################################################
    # (2) TASK CLASSIFICATION METRICS (AGGREGATED)
    ###########################################################################
    df_task_cls = compute_task_classification_metrics_aggregated(df_agg_tasks)
    print("\n===== TASK CLASSIFICATION (AGGREGATED) =====\n")
    print(df_task_cls.to_string(index=False))
    
    ###########################################################################
    # (3) TIME DEVIATIONS OR ERRORS IN TIME SPENT PER TASK
    ###########################################################################
    time_dev_summary, df_with_errors = compute_time_deviation_metrics(df_agg_tasks)
    print("\n===== TIME DEVIATIONS PER TASK =====\n")
    show_cols = [
        "Activity",
        "pred_active","gt_active","abs_error_active","pct_error_active",
        "pred_inactive","gt_inactive","abs_error_inactive","pct_error_inactive",
        "pred_total","gt_total","abs_error_total","pct_error_total"
    ]
    df_show = df_with_errors[show_cols].sort_values("Activity")
    print(df_show.to_string(index=False))
    
    ###########################################################################
    # (4) ACTIVITY DETECTION (ACTIVE vs. INACTIVE)
    ###########################################################################
    act_results = compute_activity_detection_metrics_aggregated(df_agg_tasks)
    print("\n===== ACTIVITY DETECTION METRICS (AGGREGATED) =====\n")
    for k,v in act_results.items():
        print(f"{k}: {v:.3f}")
    
    ###########################################################################
    # (5) OVERALL SUMMARIES
    ###########################################################################
    print("\n===== OVERALL TIME DEVIATION SUMMARY =====\n")
    for k,v in time_dev_summary.items():
        if isinstance(v,(float,int)):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
    
    # The final row of df_task_cls is the micro-average => overall P, R, F1
    micro_row = df_task_cls.loc[df_task_cls["Activity"]=="MICRO-AVERAGE"].squeeze()
    print("\n===== OVERALL TASK CLASSIFICATION (MICRO-AVERAGE) =====\n")
    print(micro_row.to_string())

    ###########################################################################
    # Example Plot: Compare predicted vs. GT total time per task
    ###########################################################################
    plt.figure(figsize=(8,5))
    df_sorted = df_with_errors.sort_values("gt_total", ascending=False)
    x = np.arange(len(df_sorted))
    plt.bar(x - 0.2, df_sorted["gt_total"], width=0.4, label="GT total")
    plt.bar(x + 0.2, df_sorted["pred_total"], width=0.4, label="Pred total")
    plt.xticks(x, df_sorted["Activity"], rotation=45, ha="right")
    plt.ylabel("Seconds")
    plt.title("Per-Task GT vs. Predicted Total Time (All Videos Combined)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    ###########################################################################
    # Example Plot: Boxplot of % error in total time across tasks
    ###########################################################################
    plt.figure(figsize=(5,4))
    sns.boxplot(y=df_with_errors["pct_error_total"]*100)
    plt.ylabel("Percentage Error (%)")
    plt.title("Distribution of % Error in Total Seconds (Per Task)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# In[27]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################

def read_model_and_gt_files(consolidated_folder="consolidated"):
    """
    Reads all pairs of CSV files in 'consolidated_folder' that match the pattern:
      <prefix>_parsed.csv  (model output)
      <prefix>_MANUALLYLABELED.csv  (ground truth)
    Each file has columns:
       Activity, Active_seconds, Inactive_seconds [optionally Total_seconds]
    Returns a list of tuples: [(df_pred, df_gt, prefix), ...]
    """
    parsed_files = sorted(glob.glob(os.path.join(consolidated_folder, "*_parsed.csv")))
    data_pairs = []
    
    for parsed_fp in parsed_files:
        prefix = os.path.basename(parsed_fp).replace("_parsed.csv", "")
        gt_fp = os.path.join(consolidated_folder, f"{prefix}_MANUALLYLABELED.csv")
        
        if not os.path.exists(gt_fp):
            print(f"Warning: Ground truth file not found for {parsed_fp}")
            continue
        
        df_pred = pd.read_csv(parsed_fp)
        df_gt   = pd.read_csv(gt_fp)
        
        # Check for minimal required columns
        for col in ["Activity", "Active_seconds", "Inactive_seconds"]:
            if col not in df_pred.columns:
                raise ValueError(f"Missing '{col}' in {parsed_fp}")
            if col not in df_gt.columns:
                raise ValueError(f"Missing '{col}' in {gt_fp}")
        
        # If "Total_seconds" not present, create it
        if "Total_seconds" not in df_pred.columns:
            df_pred["Total_seconds"] = df_pred["Active_seconds"] + df_pred["Inactive_seconds"]
        if "Total_seconds" not in df_gt.columns:
            df_gt["Total_seconds"] = df_gt["Active_seconds"] + df_gt["Inactive_seconds"]
        
        data_pairs.append((df_pred, df_gt, prefix))
    
    return data_pairs


def match_tasks(df_pred, df_gt, task_col="Activity"):
    """
    1) Aggregate each DataFrame by Activity (sum) => one row per Activity
       (in case there are duplicates).
    2) Merge them on 'Activity'
    Returns a DataFrame with columns:
       Activity, pred_active, pred_inactive, pred_total,
                 gt_active,   gt_inactive,   gt_total
    for the time-based metrics (MAE, MAPE, etc.).
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

###############################################################################
# NEW FUNCTION: GENERATE REQUESTED TABLE
###############################################################################

def create_ground_truth_only_chart(df_agg_tasks):
    """
    Create a bar chart showing only ground truth total time for each task.
    Excludes 'END' task.
    
    Input:
    - df_agg_tasks: DataFrame with aggregated task data
    """
    # Filter out the 'END' task
    df_filtered = df_agg_tasks[df_agg_tasks["Activity"] != "END"].copy()
    
    # Sort tasks by ground truth total time for better visualization
    df_sorted = df_filtered.sort_values("gt_total", ascending=False)
    
    # Set up the figure with appropriate size
    plt.figure(figsize=(14, 8))
    
    # Define positions
    x = np.arange(len(df_sorted))
    
    # Create the ground truth bars for total time
    plt.bar(x, df_sorted["gt_total"], color="#ccf4a5", label='Ground Truth')
    
    # Add actual time values on top of each bar
    for i, value in enumerate(df_sorted["gt_total"]):
        plt.text(i, value + 5, f"{value:.0f}", ha='center', va='bottom', 
                 color='black', fontsize=9)
    
    # Customize the chart
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    plt.title('Ground Truth Total Time by Task', fontsize=14)
    plt.xticks(x, df_sorted["Activity"], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to leave some space for the labels
    plt.ylim(0, df_sorted["gt_total"].max() * 1.1)  # Add 10% padding for labels
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("ground_truth_only_chart.png", dpi=300, bbox_inches='tight')
    print("\nGround Truth chart saved as 'ground_truth_only_chart.png'")
    
    # Display the chart
    plt.show()

###############################################################################
# NEW FUNCTION: CREATE STACKED BAR CHART
###############################################################################

def create_active_time_bar_chart(df_agg_tasks):
    """
    Create a bar chart comparing ground truth vs model-identified active time for each task.
    Adds MAPE labels at the top of the GT bar but centered between GT and Model bars.
    Excludes 'END' task and only shows Active category.
    
    Input:
    - df_agg_tasks: DataFrame with aggregated task data
    """
    # Filter out the 'END' task
    df_filtered = df_agg_tasks[df_agg_tasks["Activity"] != "END"].copy()
    
    # Sort tasks by ground truth active time for better visualization
    df_sorted = df_filtered.sort_values("gt_active", ascending=False)
    
    # Set up the figure with appropriate size
    plt.figure(figsize=(14, 8))
    
    # Define bar width and positions
    bar_width = 0.35
    x = np.arange(len(df_sorted))
    
    # Calculate MAPE for active time for each task
    mape_active = []
    for _, row in df_sorted.iterrows():
        if row["gt_active"] > 0:
            mape = abs(row["pred_active"] - row["gt_active"]) / row["gt_active"] * 100
            mape_active.append(f"{mape:.1f}%")
        else:
            mape_active.append("N/A")
    
    # Create the ground truth bars for active time only
    gt_bars = plt.bar(x - bar_width/2, df_sorted["gt_active"], bar_width, 
            label='Ground Truth', color="#ccf4a5")
    
    # Create the model prediction bars for active time only
    model_bars = plt.bar(x + bar_width/2, df_sorted["pred_active"], bar_width, 
            label='Model Identified', color="#aacff5")
    
    # Add MAPE labels positioned at the top of the highest bar
    for i, mape in enumerate(mape_active):
        # Determine which bar is higher for this task
        gt_height = df_sorted["gt_active"].iloc[i]
        model_height = df_sorted["pred_active"].iloc[i]
        # Position at the top of the highest bar
        y_pos = max(gt_height, model_height) + 5  # Adding small offset for visibility
        # Position x coordinate still centered between the bars
        x_pos = x[i]
        
        plt.text(x_pos, y_pos, 
                 f"MAPE: {mape}", ha='center', va='bottom', 
                 color='black', fontsize=9)
    
    # Customize the chart
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Active Time (seconds)', fontsize=12)
    plt.title('Ground Truth vs Model-Identified Active Time by Task', fontsize=14)
    plt.xticks(x, df_sorted["Activity"], rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to leave some space for the labels
    y_max = max(df_sorted["gt_active"].max(), df_sorted["pred_active"].max())
    plt.ylim(0, y_max * 1.15)  # Add 15% padding for labels
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("active_time_comparison_chart.png", dpi=300, bbox_inches='tight')
    print("\nChart saved as 'active_time_comparison_chart.png'")
    
    # Display the chart
    plt.show()

def create_per_task_aggregate_chart(df_agg_tasks):
    """
    Create a bar chart comparing total model time vs ground truth time for each task.
    Excludes 'END' task.
    
    Input:
    - df_agg_tasks: DataFrame with aggregated task data
    """
    # Filter out the 'END' task
    df_filtered = df_agg_tasks[df_agg_tasks["Activity"] != "END"].copy()
    
    # Sort tasks by ground truth total time for better visualization
    df_sorted = df_filtered.sort_values("gt_total", ascending=False)
    
    # Set up the figure with appropriate size
    plt.figure(figsize=(14, 8))
    
    # Define bar width and positions
    bar_width = 0.35
    x = np.arange(len(df_sorted))
    
    # Calculate MAPE for total time for each task
    mape_total = []
    for _, row in df_sorted.iterrows():
        if row["gt_total"] > 0:
            mape = abs(row["pred_total"] - row["gt_total"]) / row["gt_total"] * 100
            mape_total.append(f"{mape:.1f}%")
        else:
            mape_total.append("N/A")
    
    # Create the ground truth bars for total time
    gt_bars = plt.bar(x - bar_width/2, df_sorted["gt_total"], bar_width, 
            label='Ground Truth', color="#ccf4a5")
    
    # Create the model prediction bars for total time
    model_bars = plt.bar(x + bar_width/2, df_sorted["pred_total"], bar_width, 
            label='Model Identified', color="#aacff5")
    
    # Add MAPE labels positioned at the top of the highest bar
    for i, mape in enumerate(mape_total):
        # Determine which bar is higher for this task
        gt_height = df_sorted["gt_total"].iloc[i]
        model_height = df_sorted["pred_total"].iloc[i]
        # Position at the top of the highest bar
        y_pos = max(gt_height, model_height) + 5  # Adding small offset for visibility
        # Position x coordinate still centered between the bars
        x_pos = x[i]
        
        plt.text(x_pos, y_pos, 
                 f"MAPE: {mape}", ha='center', va='bottom', 
                 color='black', fontsize=9)
    
    # Customize the chart
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    plt.title('Ground Truth vs Model-Identified Total Time by Task', fontsize=14)
    plt.xticks(x, df_sorted["Activity"], rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to leave some space for the labels
    y_max = max(df_sorted["gt_total"].max(), df_sorted["pred_total"].max())
    plt.ylim(0, y_max * 1.15)  # Add 15% padding for labels
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("total_time_comparison_chart.png", dpi=300, bbox_inches='tight')
    print("\nTotal time chart saved as 'total_time_comparison_chart.png'")
    
    # Display the chart
    plt.show()
def create_ground_truth_table(df_agg_tasks):
    """
    Create a table showing ground truth total time for each task
    and its percentage of the overall total time.
    Excludes 'END' task.
    
    Input:
    - df_agg_tasks: DataFrame with aggregated task data
    
    Returns:
    - df_table: DataFrame with the table data
    """
    # Filter out the 'END' task
    df_filtered = df_agg_tasks[df_agg_tasks["Activity"] != "END"].copy()
    
    # Calculate total time across all tasks
    total_time = df_filtered["gt_total"].sum()
    
    # Calculate percentage for each task
    df_filtered["percentage"] = (df_filtered["gt_total"] / total_time * 100).round(2)
    
    # Sort tasks by ground truth total time for better visualization
    df_table = df_filtered.sort_values("gt_total", ascending=False)
    
    # Select and rename columns for the final table
    df_table = df_table[["Activity", "gt_total", "percentage"]]
    df_table = df_table.rename(columns={
        "Activity": "Task",
        "gt_total": "Total Time (seconds)",
        "percentage": "Percentage (%)"
    })
    
    # Display the table
    print("\n======== GROUND TRUTH TIME BY TASK ========\n")
    print(df_table.to_string(index=False))
    
    # Save the table to a CSV file
    output_file = "ground_truth_time_table.csv"
    df_table.to_csv(output_file, index=False)
    print(f"\nGround truth time table saved to {output_file}")
    
    return df_table
# Add this line to your main() function
def calculate_aggregated_active_mape(df_agg_tasks):
    """
    Calculate the MAPE for active time aggregated across all tasks,
    excluding the 'END' task.
    
    Input:
    - df_agg_tasks: DataFrame with aggregated task data
    
    Returns:
    - aggregated_mape: MAPE value for active time across all tasks
    """
    # Filter out the 'END' task
    df_filtered = df_agg_tasks[df_agg_tasks["Activity"] != "END"].copy()
    
    # Calculate sums across all tasks
    total_gt_active = df_filtered["gt_active"].sum()
    total_pred_active = df_filtered["pred_active"].sum()
    
    # Calculate the absolute error
    absolute_error = abs(total_pred_active - total_gt_active)
    
    # Calculate the MAPE
    if total_gt_active > 0:
        aggregated_mape = (absolute_error / total_gt_active) * 100
    else:
        print("Warning: Total ground truth active time is zero.")
        aggregated_mape = float('inf')
    
    print(f"\n======== AGGREGATED ACTIVE TIME MAPE ========")
    print(f"Total Ground Truth Active Time: {total_gt_active:.2f} seconds")
    print(f"Total Predicted Active Time: {total_pred_active:.2f} seconds")
    print(f"Absolute Error: {absolute_error:.2f} seconds")
    print(f"Aggregated Active Time MAPE: {aggregated_mape:.2f}%")
    
    return aggregated_mape
###############################################################################
# MAIN SCRIPT: MODIFIED TO OUTPUT THE REQUESTED CHART
###############################################################################

def main():
    data_pairs = read_model_and_gt_files("consolidated")
    if not data_pairs:
        print("No valid pairs found in 'consolidated'.")
        return
    
    all_merged_for_time = []  # For time-based aggregated metrics
    
    for (df_pred, df_gt, prefix) in data_pairs:
        # Merge tasks for time-based metrics
        df_merged = match_tasks(df_pred, df_gt)
        df_merged["video_prefix"] = prefix
        all_merged_for_time.append(df_merged)
    
    # Combine all videos for time-based aggregated metrics
    df_all = pd.concat(all_merged_for_time, ignore_index=True)
    
    # Summation across videos => get single row per Activity
    df_agg_tasks = (
        df_all
        .groupby("Activity", as_index=False)
        .agg({
            "pred_active":"sum",
            "pred_inactive":"sum",
            "pred_total":"sum",
            "gt_active":"sum",
            "gt_inactive":"sum",
            "gt_total":"sum"
        })
    )
    
    # Generate the MAPE table
    mape_table = generate_mape_table(df_agg_tasks)
    
    # Display the table
    print("\n======== MAPE COMPARISON TABLE ========\n")
    print(mape_table.to_string(index=False))
    
    # Save the table to a CSV file
    output_file = "mape_comparison_table.csv"
    mape_table.to_csv(output_file, index=False)
    print(f"MAPE table saved to {output_file}")
    
    # Create and display the active time bar chart
    create_active_time_bar_chart(df_agg_tasks)
    
    # Create and display the per-task aggregate (total) time chart
    create_per_task_aggregate_chart(df_agg_tasks)
    
    # Create and display the ground truth only chart
    create_ground_truth_only_chart(df_agg_tasks)
    df_table = create_ground_truth_table(df_agg_tasks)
    
    # Add this line to your main() function
    aggregated_active_mape = calculate_aggregated_active_mape(df_agg_tasks)
    print("MAPE of active time aggregated across all tasks: ", aggregated_active_mape)
if __name__ == "__main__":
    main()


# In[ ]:





# In[25]:




