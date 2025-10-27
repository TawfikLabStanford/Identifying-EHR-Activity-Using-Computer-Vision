#!/usr/bin/env python3
"""
Unified pipeline script for end-to-end EHR activity detection.

This script orchestrates the complete workflow from video input to
consolidated activity detection results.

Usage:
    python run_pipeline.py --config config.yaml --input video.mp4 --output ./results
    python run_pipeline.py --input video.mp4  # Uses default config

Steps:
    1. Load configuration
    2. Extract frames from video (1 fps)
    3. Run YOLO detection on frames
    4. Perform OCR and task classification
    5. Calculate frame-to-frame MSE for activity detection
    6. Consolidate results into time periods
    7. (Optional) Run validation if ground truth provided
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Import functions from main.py
from main import (
    load_config,
    extract_frames,
    runModel,
    load_epic_dictionary,
    txtToList,
    parseLabelFile,
    detectActivity,
    consolidate,
    compute_mode_crop_box,
    apply_mode_crop,
    getSize,
    IMAGES_SUBDIR,
    YOLO_OUTPUT_SUBDIR,
    LABELS_SUBDIR
)

import csv
import numpy as np
import pandas as pd
from datetime import timedelta
import re


def run_pipeline(config_path, input_video, output_dir, ground_truth=None, verbose=True):
    """
    Execute complete EHR activity detection pipeline.

    Parameters:
        config_path (str): Path to config.yaml
        input_video (str): Path to input video file
        output_dir (str): Output directory for results
        ground_truth (str): Optional path to ground truth annotations CSV
        verbose (bool): Print detailed progress information

    Returns:
        dict: Results summary with paths to output files
    """
    if verbose:
        print("="*80)
        print("EHR ACTIVITY DETECTION PIPELINE")
        print("="*80)
        print(f"Input video: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"Config file: {config_path}")
        print("="*80)

    # 1. Load configuration
    if verbose:
        print("\n[STEP 1/7] Loading configuration...")

    config = load_config(config_path)

    # Validate input video exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Epic dictionary
    if verbose:
        print("\n[STEP 2/7] Loading Epic task dictionary...")

    epic_dictionary_path = config['paths']['epic_dictionary']
    epic_dictionary = load_epic_dictionary(epic_dictionary_path)

    if not epic_dictionary:
        print(f"[WARNING] No Epic dictionary loaded from {epic_dictionary_path}")

    # 3. Extract frames
    if verbose:
        print("\n[STEP 3/7] Extracting frames from video (1 fps)...")

    video_stem = Path(input_video).stem
    frame_dir = os.path.join(output_dir, IMAGES_SUBDIR)

    extract_frames(input_video, frame_dir)

    frame_count = len([f for f in os.listdir(frame_dir) if f.lower().endswith('.png')])
    if verbose:
        print(f"  Extracted {frame_count} frames to {frame_dir}")

    # Get original video dimensions
    orig_w, orig_h = getSize(input_video)
    if verbose:
        print(f"  Original video dimensions: {orig_w}x{orig_h}")

    # 4. Crop black borders
    if verbose:
        print("\n[STEP 4/7] Detecting and removing black borders...")

    mode_box = compute_mode_crop_box(frame_dir, black_threshold=30)
    if mode_box:
        if verbose:
            print(f"  Mode crop box: {mode_box}")
        apply_mode_crop(frame_dir, mode_box)
        cropped_w, cropped_h = mode_box[2], mode_box[3]
        if verbose:
            print(f"  Cropped dimensions: {cropped_w}x{cropped_h}")
    else:
        if verbose:
            print("  No crop detected, using original dimensions")
        cropped_w, cropped_h = orig_w, orig_h

    # 5. Run YOLO detection
    if verbose:
        print("\n[STEP 5/7] Running YOLO detection on frames...")

    yolo_output_dir = os.path.join(output_dir, YOLO_OUTPUT_SUBDIR)
    model_path = config['paths']['model_weights']

    runModel(
        path_to_images=frame_dir,
        yolo_output_path=yolo_output_dir,
        model_weights=model_path,
        conf_threshold=config['yolo']['confidence_threshold'],
        imgsz=config['yolo']['image_size']
    )

    if verbose:
        print(f"  YOLO detection complete")

    # 6. Create CSV with detections and perform OCR
    if verbose:
        print("\n[STEP 6/7] Performing OCR and task classification...")

    out_csv_path = os.path.join(output_dir, f"{video_stem}_detections.csv")
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(".png")])
    label_folder = os.path.join(yolo_output_dir, "results", LABELS_SUBDIR)

    with open(out_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_stamp", "page_title", "sidebar", "order_box", "text_box", "pop_up_box"])

        for frame_file in frame_files:
            base_name = os.path.splitext(frame_file)[0]
            label_file_path = os.path.join(label_folder, base_name + ".txt")

            if os.path.exists(label_file_path):
                lines = txtToList(label_file_path)
                row_data = parseLabelFile(lines, frame_file, frame_dir, epic_dictionary)
            else:
                # No YOLO detection => fill with NaN
                match = re.search(r'(\d+)', base_name)
                frameNum = match.group(1) if match else "0"
                sec = int(frameNum) - 1
                timestamp = str(timedelta(seconds=sec))
                row_data = [timestamp, np.nan, np.nan, np.nan, np.nan, np.nan]

            writer.writerow(row_data)

    if verbose:
        print(f"  Wrote detections CSV: {out_csv_path}")

    # 7. Calculate MSE for activity detection
    if verbose:
        print("\n[STEP 7/7] Calculating activity (MSE) and consolidating results...")

    activity_vals = detectActivity(frame_dir, cropped_h, cropped_w)

    # Insert activity values into CSV
    df = pd.read_csv(out_csv_path)
    min_len = min(len(df), len(activity_vals))
    df["%change from previous frame"] = activity_vals[:min_len]
    df.to_csv(out_csv_path, index=False)

    # 8. Consolidate tasks & active/inactive states
    consolidate(
        out_csv_path,
        mse_threshold=config['activity_detection']['mse_threshold'],
        task_change_threshold=config['task_classification']['task_change_threshold'],
        threshold_of_inactivity=config['activity_detection']['threshold_of_inactivity']
    )

    consolidated_path = os.path.join("consolidated", f"{video_stem}_detections_parsed.csv")

    # 9. Optional: Run validation if ground truth provided
    validation_results = None
    if ground_truth:
        if verbose:
            print("\n[VALIDATION] Running validation against ground truth...")

        try:
            from validate_results import calculate_metrics
            validation_results = calculate_metrics(consolidated_path, ground_truth)

            if verbose:
                print(f"  Validation MAPE: {validation_results.get('mape', 'N/A'):.2f}%")
                print(f"  Validation complete")
        except Exception as e:
            print(f"[WARNING] Validation failed: {e}")

    # Summary
    results = {
        'input_video': input_video,
        'output_directory': output_dir,
        'detections_csv': out_csv_path,
        'consolidated_csv': consolidated_path,
        'frame_count': frame_count,
        'validation_results': validation_results
    }

    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Detections CSV: {out_csv_path}")
        print(f"Consolidated CSV: {consolidated_path}")
        print(f"Total frames processed: {frame_count}")
        print("="*80)

    return results


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="EHR Activity Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input video file (required)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )

    parser.add_argument(
        '--ground-truth',
        type=str,
        default=None,
        help='Optional path to ground truth CSV for validation'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed progress output'
    )

    return parser.parse_args()


def main():
    """
    Main entry point for pipeline script.
    """
    args = parse_arguments()

    try:
        results = run_pipeline(
            config_path=args.config,
            input_video=args.input,
            output_dir=args.output,
            ground_truth=args.ground_truth,
            verbose=not args.quiet
        )

        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
