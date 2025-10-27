import os
import shutil
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re
from math import pi

#########################################
# 1. CONFIGURATION
#########################################

# -- Paths --
base_dir = os.path.join(os.getenv('GROUP_HOME'), 'liemn/ComputerVision/k_fold_data')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Subfolder for saving each fold's training curve plots:
train_plots_dir = os.path.join(results_dir, 'training_plots')
os.makedirs(train_plots_dir, exist_ok=True)

# Subfolder for final aggregated (across folds) plots:
aggregate_plots_dir = os.path.join(results_dir, 'aggregate_plots')
os.makedirs(aggregate_plots_dir, exist_ok=True)

# Subfolder for aggregated fold-wise training curves 
aggregate_fold_curves_dir = os.path.join(aggregate_plots_dir, 'fold_curve_averages')
os.makedirs(aggregate_fold_curves_dir, exist_ok=True)

# -- Hyperparameters --
EPOCHS = 50
IMG_SIZE = 960  # Match inference resolution (960px per paper Section 2.4.2)
BATCH_SIZE = 32

# -- Model(s) --
MODELS = ['yolov8x.pt']  

# -- Class names --
NAMES = [
    'order_box', 'pop_up_box', 'side_bar', 'side_bar_active_tab', 'side_bar_menu',
    'task_bar_active_tab', 'task_bar_menu', 'task_title', 'text_box', 'login'
]

# -- Folds --
NUM_FOLDS = 5
FOLDS = range(1, NUM_FOLDS + 1)

#########################################
# 2. YAML CREATION FOR EACH FOLD
#########################################
def create_data_yaml(fold_dir):
    data = {
        'train': os.path.join(fold_dir, 'train'),
        'val': os.path.join(fold_dir, 'test'),
        'nc': len(NAMES),
        'names': NAMES
    }
    yaml_path = os.path.join(fold_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

#########################################
# 3. TRAIN AND EVALUATE FOR ONE FOLD
#########################################
def train_and_evaluate(fold_dir, fold_num, model_name):
    run_name = f'fold_{fold_num}_{model_name.split(".")[0]}'
    run_dir  = os.path.join('runs', 'detect', run_name)
    weights_dir = os.path.join(run_dir, 'weights')
    last_checkpoint_path = os.path.join(weights_dir, 'last.pt')
    
    # Create data.yaml
    yaml_path = create_data_yaml(fold_dir)

    # Decide if we resume
    resume_training = os.path.isfile(last_checkpoint_path)
    
    # Train
    model = YOLO(model_name)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=run_name,
        project='runs/detect',
        resume=resume_training
    )
    
    # Evaluate
    metrics = model.val()
    
    # Save final results (per-fold text)
    results_txt_path = os.path.join(results_dir, f'{run_name}_results.txt')
    with open(results_txt_path, 'w') as f:
        f.write(f"Fold {fold_num} - Model {model_name} Results:\n")
        f.write(f"mAP50: {metrics.box.map50:.3f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.3f}\n")
        
        for i, cls_name in enumerate(NAMES):
            p  = metrics.box.p[i]
            r  = metrics.box.r[i]
            f1 = metrics.box.f1[i]
            f.write(f"\n{cls_name}:\n")
            f.write(f"  Precision: {p:.3f}\n")
            f.write(f"  Recall: {r:.3f}\n")
            f.write(f"  F1-score: {f1:.3f}\n")
    
    # Copy and parse training logs to produce per-fold training curves
    csv_path = os.path.join(run_dir, 'results.csv')
    if os.path.isfile(csv_path):
        # Copy the CSV for safekeeping
        destination_csv = os.path.join(results_dir, f'{run_name}_training_results.csv')
        shutil.copyfile(csv_path, destination_csv)
        
        # Now create training curves for this fold
        plot_per_fold_training_curves(csv_path, run_name, train_plots_dir)
    else:
        print(f"Warning: No results.csv found at {csv_path}; skipping fold curves.")
    
    return metrics.box.map50, metrics.box.map

#########################################
# 4. PLOT PER-FOLD TRAINING CURVES
#########################################
def plot_per_fold_training_curves(csv_path, run_name, output_dir):
    """
    Reads YOLOv8 logs from results.csv and plots:
      - Train/Val box/cls/dfl losses (if available)
      - Precision, Recall, F1
      - mAP@0.5, mAP@0.5-0.95
    Saves a figure in `output_dir` named <run_name>_training_curves.png
    """
    df = pd.read_csv(csv_path)
    
    # Common columns in YOLOv8
    epoch_col = 'epoch'
    
    # Train loss keys
    tbox_col = 'train/box_loss'
    tcls_col = 'train/cls_loss'
    tdfl_col = 'train/dfl_loss'
    
    # Val loss keys
    vbox_col = 'val/box_loss'
    vcls_col = 'val/cls_loss'
    vdfl_col = 'val/dfl_loss'
    
    # Metric keys
    prec_col = 'metrics/precision(B)'
    rec_col  = 'metrics/recall(B)'
    map50_col= 'metrics/mAP50(B)'
    map95_col= 'metrics/mAP50-95(B)'
    
    # Compute F1 if possible
    if prec_col in df.columns and rec_col in df.columns:
        p_arr = df[prec_col].values
        r_arr = df[rec_col].values
        f1_arr= np.where((p_arr + r_arr) > 0, 2*p_arr*r_arr/(p_arr+r_arr), 0)
    else:
        f1_arr= None
    
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    fig.suptitle(f"Training Curves - {run_name}", fontsize=16)
    
    # (A) Losses Subplot
    ax = axes[0,0]
    epochs = df[epoch_col] if epoch_col in df.columns else range(len(df))
    if tbox_col in df.columns:
        ax.plot(epochs, df[tbox_col], label='Train Box', color='blue')
    if tcls_col in df.columns:
        ax.plot(epochs, df[tcls_col], label='Train Cls', color='cyan')
    if tdfl_col in df.columns:
        ax.plot(epochs, df[tdfl_col], label='Train DFL', color='magenta')
    
    if vbox_col in df.columns:
        ax.plot(epochs, df[vbox_col], label='Val Box', linestyle='--', color='blue')
    if vcls_col in df.columns:
        ax.plot(epochs, df[vcls_col], label='Val Cls', linestyle='--', color='cyan')
    if vdfl_col in df.columns:
        ax.plot(epochs, df[vdfl_col], label='Val DFL', linestyle='--', color='magenta')
    
    ax.set_title("Losses (Train vs. Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    
    # (B) Precision, Recall, F1
    ax = axes[0,1]
    if prec_col in df.columns:
        ax.plot(epochs, df[prec_col], label='Precision', color='green')
    if rec_col in df.columns:
        ax.plot(epochs, df[rec_col], label='Recall', color='red')
    if f1_arr is not None:
        ax.plot(epochs, f1_arr, label='F1', color='black')
    ax.set_title("Precision / Recall / F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True)
    
    # (C) mAP@0.5
    ax = axes[1,0]
    if map50_col in df.columns:
        ax.plot(epochs, df[map50_col], label='mAP@0.5', color='purple')
    ax.set_title("mAP@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend()
    ax.grid(True)
    
    # (D) mAP@0.5:0.95
    ax = axes[1,1]
    if map95_col in df.columns:
        ax.plot(epochs, df[map95_col], label='mAP@0.5:0.95', color='orange')
    ax.set_title("mAP@0.5:0.95")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{run_name}_training_curves.png")
    plt.savefig(out_path)
    plt.close(fig)

#########################################
# 5. MAIN EXECUTION: K-FOLD TRAINING
#########################################
overall_results = {model: {'mAP50': [], 'mAP50-95': []} for model in MODELS}

for i in FOLDS:
    fold_dir = os.path.join(base_dir, f'Fold_{i}')
    print(f"Processing Fold {i} ...")
    
    for model in MODELS:
        print(f"  Training & evaluating {model}")
        mAP50, mAP50_95 = train_and_evaluate(fold_dir, i, model)
        overall_results[model]['mAP50'].append(mAP50)
        overall_results[model]['mAP50-95'].append(mAP50_95)

# Summarize final cross-fold results
overall_txt_path = os.path.join(results_dir, 'overall_results.txt')
with open(overall_txt_path, 'w') as f:
    f.write("Overall Results:\n\n")
    for model in MODELS:
        folds_map50    = overall_results[model]['mAP50']
        folds_map50_95 = overall_results[model]['mAP50-95']
        avg_mAP50    = np.mean(folds_map50)
        avg_mAP50_95 = np.mean(folds_map50_95)
        std_mAP50    = np.std(folds_map50)
        std_mAP50_95 = np.std(folds_map50_95)
        f.write(f"{model}:\n")
        f.write(f"  Average mAP50:    {avg_mAP50:.3f} ± {std_mAP50:.3f}\n")
        f.write(f"  Average mAP50-95: {avg_mAP50_95:.3f} ± {std_mAP50_95:.3f}\n\n")

print("All folds complete. Now generating aggregated cross-fold plots...")

#########################################
# 6. PARSE FOLD TEXT RESULTS FOR PER-CLASS METRICS
#########################################
def parse_fold_text_results(file_path):
    """
    Example text format:
      Fold 1 - Model yolov8x.pt Results:
      mAP50: 0.848
      mAP50-95: 0.683

      order_box:
        Precision: 0.983
        Recall: 0.341
        F1-score: 0.506
      ...
    Returns dict with 'mAP50', 'mAP50-95' and 'class_metrics' {...}
    """
    data = {'mAP50':None, 'mAP50-95':None, 'class_metrics':{}}
    current_class = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('mAP50:'):
                data['mAP50'] = float(line.split(':')[-1].strip())
            elif line.startswith('mAP50-95:'):
                data['mAP50-95'] = float(line.split(':')[-1].strip())
            else:
                m = re.match(r'^([A-Za-z0-9_]+):$', line)
                if m:
                    current_class = m.group(1)
                    data['class_metrics'][current_class] = {'p':0,'r':0,'f1':0}
                elif current_class and line.startswith('Precision:'):
                    data['class_metrics'][current_class]['p'] = float(line.split(':')[-1])
                elif current_class and line.startswith('Recall:'):
                    data['class_metrics'][current_class]['r'] = float(line.split(':')[-1])
                elif current_class and 'F1-score:' in line:
                    data['class_metrics'][current_class]['f1'] = float(line.split(':')[-1])
    return data

fold_class_data = {}  # Will store final results from each fold
for i in FOLDS:
    run_name = f'fold_{i}_{MODELS[0].split(".")[0]}'
    txt_path = os.path.join(results_dir, f"{run_name}_results.txt")
    if os.path.isfile(txt_path):
        fold_class_data[i] = parse_fold_text_results(txt_path)
    else:
        fold_class_data[i] = None

# Build a DataFrame of per-class metrics across folds
rows = []
for fold_id, res_dict in fold_class_data.items():
    if res_dict is None: 
        continue
    for cls in NAMES:
        cmet = res_dict['class_metrics'].get(cls, None)
        if cmet:
            rows.append({
                'fold': fold_id,
                'class': cls,
                'precision': cmet['p'],
                'recall': cmet['r'],
                'f1': cmet['f1']
            })
df_metrics = pd.DataFrame(rows)

#########################################
# 7. CROSS-FOLD PLOTS OF PER-CLASS METRICS
#########################################
grouped = df_metrics.groupby('class')[['precision','recall','f1']]
mean_metrics = grouped.mean()
std_metrics  = grouped.std()

classes = mean_metrics.index
x_inds = np.arange(len(classes))
bar_width=0.2

# (A) Bar chart with error bars
plt.figure(figsize=(10,6))
plt.bar(x_inds - bar_width, mean_metrics['precision'], bar_width,
        yerr=std_metrics['precision'], capsize=4, label='Precision')
plt.bar(x_inds, mean_metrics['recall'], bar_width,
        yerr=std_metrics['recall'], capsize=4, label='Recall')
plt.bar(x_inds + bar_width, mean_metrics['f1'], bar_width,
        yerr=std_metrics['f1'], capsize=4, label='F1')

plt.xticks(x_inds, classes, rotation=45, ha='right')
plt.ylabel("Metric Value")
plt.title("Per-class metrics (mean ± std across folds)")
plt.legend()
plt.tight_layout()
per_class_bar = os.path.join(aggregate_plots_dir, "per_class_bar_error.png")
plt.savefig(per_class_bar)
plt.close()

# (B) Bubble chart (avg P vs. avg R, bubble size = F1)
mean_p = mean_metrics['precision'].values
mean_r = mean_metrics['recall'].values
mean_f = mean_metrics['f1'].values
bubble_size = mean_f * 600

plt.figure(figsize=(8,6))
plt.scatter(mean_p, mean_r, s=bubble_size, alpha=0.6)
for i, cls_name in enumerate(classes):
    plt.text(mean_p[i]+0.001, mean_r[i]+0.001, cls_name, fontsize=9)

plt.title("Precision vs. Recall (bubble size = F1)")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.grid(True)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
bubble_path = os.path.join(aggregate_plots_dir, "bubble_precision_recall.png")
plt.tight_layout()
plt.savefig(bubble_path)
plt.close()

# (C) Radar (spider) chart of average P, R, F1
def radar_plot(ax, values, labels, title):
    """
    ax: polar axis
    values: list or array of length N
    labels: class names
    title: subplot title
    """
    N = len(values)
    angles = [n/float(N)*2*pi for n in range(N)]
    angles += [angles[0]]
    values = list(values) + [values[0]]

    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([angle*180/pi for angle in angles[:-1]], labels)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.3)
    ax.set_title(title, y=1.08)

cls_list = list(classes)
p_vals = mean_metrics['precision'].values
r_vals = mean_metrics['recall'].values
f_vals = mean_metrics['f1'].values

fig = plt.figure(figsize=(18,6))
ax1 = fig.add_subplot(1,3,1, polar=True)
radar_plot(ax1, p_vals, cls_list, "Precision")

ax2 = fig.add_subplot(1,3,2, polar=True)
radar_plot(ax2, r_vals, cls_list, "Recall")

ax3 = fig.add_subplot(1,3,3, polar=True)
radar_plot(ax3, f_vals, cls_list, "F1")

plt.suptitle("Radar Charts of Mean Class Metrics (across folds)", fontsize=16)
radar_path = os.path.join(aggregate_plots_dir, "radar_charts.png")
plt.tight_layout()
plt.savefig(radar_path)
plt.close()

# (D) Simple bar chart of fold-level mAP (0.5 and 0.5:0.95)
folds_map50 = overall_results[MODELS[0]]['mAP50']
folds_map95 = overall_results[MODELS[0]]['mAP50-95']
f_idx = np.arange(len(folds_map50))

plt.figure(figsize=(6,4))
w=0.4
plt.bar(f_idx - w/2, folds_map50, w, label='mAP@0.5')
plt.bar(f_idx + w/2, folds_map95, w, label='mAP@0.5:0.95')
plt.xticks(f_idx, [str(i) for i in FOLDS])
plt.xlabel("Fold")
plt.ylabel("mAP")
plt.title("mAP across folds")
plt.legend()
plt.tight_layout()
fold_map_path = os.path.join(aggregate_plots_dir, "map_across_folds.png")
plt.savefig(fold_map_path)
plt.close()

#########################################
# 8. AGGREGATED FOLD-WISE TRAINING CURVES
#    (lighter alpha lines for each fold, plus darker average line)
#########################################

runs_detect_dir = 'runs/detect'
all_dfs = []
for i in FOLDS:
    run_name = f'fold_{i}_{MODELS[0].split(".")[0]}'
    csv_file = os.path.join(runs_detect_dir, run_name, 'results.csv')
    if os.path.isfile(csv_file):
        df_fold = pd.read_csv(csv_file)
        df_fold['fold'] = i
        all_dfs.append(df_fold)
    else:
        print(f"Warning: {csv_file} missing. Skipping fold {i} for aggregated curves.")

if not all_dfs:
    print("No CSV logs found for aggregated training curves. Exiting.")
else:
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Create F1 column if we have precision + recall
    if 'metrics/precision(B)' in df_all.columns and 'metrics/recall(B)' in df_all.columns:
        p = df_all['metrics/precision(B)']
        r = df_all['metrics/recall(B)']
        f1 = np.where((p + r) > 0, 2*p*r/(p+r), 0)
        df_all['f1'] = f1
    else:
        df_all['f1'] = np.nan
    
    # We'll define a helper to plot each metric with folds in lighter lines + average in darker
    def plot_metric_across_folds(df, metric_col, title, y_label, out_name):
        """
        Each fold in a lighter line, average (over epochs) in a darker line.
        Distinguish folds by alpha or color. 
        """
        if 'epoch' not in df.columns:
            print("No 'epoch' column found. Can't plot epoch-based metric.")
            return
        if metric_col not in df.columns:
            print(f"Column '{metric_col}' not found in df. Skipping {title}")
            return
        
        folds = sorted(df['fold'].unique())
        plt.figure(figsize=(8,6))
        base_color = 'blue'
        alpha_folds = 0.3
        
        # Plot each fold
        for fold_id in folds:
            df_f = df[df['fold'] == fold_id].sort_values('epoch')
            plt.plot(df_f['epoch'], df_f[metric_col], color=base_color, alpha=alpha_folds)
        
        # Plot average across folds
        # Group by epoch and compute mean. This only works well if folds have same # of epochs
        df_mean = df.groupby('epoch')[metric_col].mean().reset_index()
        plt.plot(df_mean['epoch'], df_mean[metric_col], color=base_color, alpha=1.0, linewidth=2.5, label='Average')
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        
        out_path = os.path.join(aggregate_fold_curves_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    
    # Let's define multiple metrics to plot separately for train vs. val
    metric_specs = [
        # (column_name, title, y_label, out_filename)
        ('train/box_loss',    'Train Box Loss (All Folds)', 'Loss', 'aggregate_train_box_loss.png'),
        ('val/box_loss',      'Val Box Loss (All Folds)',   'Loss', 'aggregate_val_box_loss.png'),
        ('train/cls_loss',    'Train Cls Loss (All Folds)', 'Loss', 'aggregate_train_cls_loss.png'),
        ('val/cls_loss',      'Val Cls Loss (All Folds)',   'Loss', 'aggregate_val_cls_loss.png'),
        ('train/dfl_loss',    'Train DFL Loss (All Folds)', 'Loss', 'aggregate_train_dfl_loss.png'),
        ('val/dfl_loss',      'Val DFL Loss (All Folds)',   'Loss', 'aggregate_val_dfl_loss.png'),
        ('metrics/precision(B)', 'Precision (All Folds)', 'Metric', 'aggregate_precision.png'),
        ('metrics/recall(B)',    'Recall (All Folds)',    'Metric', 'aggregate_recall.png'),
        ('f1',                  'F1 Score (All Folds)',   'Metric', 'aggregate_f1.png'),
        ('metrics/mAP50(B)',    'mAP@0.5 (All Folds)',    'mAP',    'aggregate_map50.png'),
        ('metrics/mAP50-95(B)', 'mAP@0.5:0.95 (All Folds)','mAP',    'aggregate_map50_95.png'),
    ]
    
    for col, ttl, yl, fn in metric_specs:
        plot_metric_across_folds(df_all, col, ttl, yl, fn)

print("\nAll done! Key outputs:\n"
      f" - Per-fold training curves in: {train_plots_dir}\n"
      f" - Aggregated cross-fold metrics in: {aggregate_plots_dir}\n"
      f" - Aggregated fold-wise training curves in: {aggregate_fold_curves_dir}\n")
