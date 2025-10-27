#!/usr/bin/env python3

import os
import sys
import csv
import re
import shutil
import random
import difflib
import yaml
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

import pybboxes as pbx
import pandas as pd

from datetime import timedelta
from textblob import TextBlob
from typing import List
from collections import Counter

from ultralytics import YOLO

################################################################################
# Configuration Loading
################################################################################

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.

    Parameters:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"[WARNING] Config file {config_path} not found. Using defaults.")
        return {
            'paths': {
                'model_weights': './models/best.pt',
                'epic_dictionary': './data/Epic_Dictionary.txt',
                'input_videos': './data/input_videos',
                'output_dir': './output'
            },
            'yolo': {
                'confidence_threshold': 0.1,
                'image_size': 960
            },
            'task_classification': {
                'task_change_threshold': 2,
                'fuzzy_match_cutoff': 0.7
            },
            'activity_detection': {
                'mse_threshold': 0.2,
                'threshold_of_inactivity': 5
            }
        }

# Load configuration
config = load_config()

################################################################################
# Global Paths and Constants
################################################################################

model_path = config['paths']['model_weights']
source_path = config['paths']['input_videos']
epic_dictionary_path = config['paths']['epic_dictionary']

IMAGES_SUBDIR = "images"
YOLO_OUTPUT_SUBDIR = "yolo_output"
LABELS_SUBDIR = "labels"

NAMES = [
    "order_box",
    "pop_up_box",
    "side_bar",
    "side_bar_active_tab",
    "side_bar_menu",
    "task_bar_active_tab",
    "task_bar_menu",
    "task_title",
    "text_box",
    "login"
]

# States
active = True
inactive = False

################################################################################
# Epic Task Dictionary (for final classification)
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


################################################################################
# 1) Load epic_dictionary.txt
################################################################################
def load_epic_dictionary(path: str) -> List[str]:
    """
    Load Epic task names from dictionary file.

    Parameters:
        path (str): Path to Epic dictionary text file

    Returns:
        list: List of Epic task names
    """
    if not os.path.exists(path):
        print(f"[WARNING] epic_dictionary.txt not found at: {path}. Returning empty list.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(lines)} entries from epic_dictionary.")
    return lines


################################################################################
# 2) Fuzzy match recognized text to epic_dictionary
################################################################################
def dictionary_fuzzy_match(text: str, dictionary_list: List[str], cutoff=0.7) -> str:
    """
    Match text to Epic dictionary using fuzzy matching.

    Parameters:
        text (str): Text to match
        dictionary_list (list): List of valid Epic task names
        cutoff (float): Similarity threshold (default: 0.7)

    Returns:
        str: Best matching task name or original text if no match
    """
    if not text:
        return text
    matches = difflib.get_close_matches(text, dictionary_list, n=1, cutoff=cutoff)
    return matches[0] if matches else text


################################################################################
# 3) Extract frames at 1 fps (using ffmpeg)
################################################################################
def extract_frames(video_path: str, out_dir: str):
    """
    Extract frames from video at 1 fps.

    Parameters:
        video_path (str): Path to input video file
        out_dir (str): Output directory for extracted frames
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = f'ffmpeg -i "{video_path}" -vf "fps=1" "{os.path.join(out_dir, "%06d.png")}" -loglevel warning'
    os.system(cmd)


################################################################################
# 4) YOLO Inference (model.predict)
################################################################################
def runModel(
    path_to_images: str,
    yolo_output_path: str,
    model_weights: str,
    conf_threshold: float = 0.1,
    imgsz: int = 960
):
    """
    Run YOLOv8 inference on extracted frames.

    Parameters:
        path_to_images (str): Directory containing input frames
        yolo_output_path (str): Output directory for YOLO results
        model_weights (str): Path to YOLOv8 model weights
        conf_threshold (float): Confidence threshold (default: 0.1, per paper Section 2.4.2)
        imgsz (int): Image size for inference (default: 960px, per paper Section 2.4.2)
    """
    print(f"[INFO] Running YOLOv8 on {path_to_images}, conf={conf_threshold}, imgsz={imgsz}")
    os.makedirs(yolo_output_path, exist_ok=True)
    results_path = os.path.join(yolo_output_path, "results")
    labels_path = os.path.join(results_path, "labels")
    os.makedirs(labels_path, exist_ok=True)

    model = YOLO(model_weights)
    results = model.predict(
        source=path_to_images,
        conf=conf_threshold,
        imgsz=imgsz,
        project=yolo_output_path,
        name="results",
        save_txt=True,
        save=False,
        exist_ok=True,
        stream=True,
        verbose=False,
        device="cuda:0",
    )

    # Save only label text files using the original image filename
    for r in results:
        original_file = os.path.basename(r.path)
        base_name, _ = os.path.splitext(original_file)
        label_filename = os.path.join(labels_path, base_name + ".txt")
        r.save_txt(txt_file=label_filename)

    print(f"[INFO] YOLO inference done.")


################################################################################
# 5) Utility to read YOLO .txt lines
################################################################################
def txtToList(label_txt: str) -> List[str]:
    """
    Read YOLO label file into list of lines.

    Parameters:
        label_txt (str): Path to YOLO label file

    Returns:
        list: Lines from label file
    """
    with open(label_txt, "r") as f:
        return f.read().splitlines()


################################################################################
# 6) OCR & dictionary matching
################################################################################
def getText(
    item: str,
    labels: List[str],
    label_file: str,
    frames_location: str,
    epic_dictionary_list: List[str],
):
    """
    Extract and recognize text from bounding boxes via OCR.

    Parameters:
        item (str): YOLO class name to search for
        labels (list): YOLO detection lines
        label_file (str): Label filename
        frames_location (str): Directory containing frames
        epic_dictionary_list (list): Epic task dictionary

    Returns:
        str: Recognized and matched text
    """
    for elem in labels:
        tokens = elem.split()
        if not tokens:
            continue
        class_index = tokens[0]
        if class_index == str(NAMES.index(item)):
            bbox_data = tokens[1:]
            bbox_data = tuple(map(float, bbox_data))

            # Find the corresponding image
            img_name = os.path.splitext(label_file)[0] + ".png"
            img_path = os.path.join(frames_location, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_w, img_h = img.shape[1], img.shape[0]
            voc_bbox = pbx.convert_bbox(bbox_data, from_type="yolo", to_type="voc", image_size=(img_w, img_h))
            x_min, y_min, x_max, y_max = map(int, voc_bbox)

            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
            raw_text = pytesseract.image_to_string(thresh_img, lang="eng", config='-c page_separator=""')

            cleaned_text = re.sub(r"[^ \nA-Za-z0-9/]+", "", raw_text)
            cleaned_text = re.sub(r"[0-9]+", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            corrected_text = TextBlob(cleaned_text).correct()
            text_str = str(corrected_text)

            matched_text = dictionary_fuzzy_match(text_str, epic_dictionary_list, cutoff=0.7)
            return matched_text

    return ""


################################################################################
# 7) Checking presence of a class in YOLO lines
################################################################################
def checkPresent(item: str, labels: List[str]) -> int:
    """
    Check if YOLO class is present in detections.

    Parameters:
        item (str): YOLO class name
        labels (list): YOLO detection lines

    Returns:
        int: 1 if present, 0 otherwise
    """
    for line in labels:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == str(NAMES.index(item)):
            return 1
    return 0


################################################################################
# 8) parseLabelFile -> single CSV row
################################################################################
def parseLabelFile(labels, labelFile, frames_location, epic_dictionary_list):
    """
    Parse YOLO detections into CSV row format.

    Parameters:
        labels (list): YOLO detection lines
        labelFile (str): Label filename
        frames_location (str): Directory containing frames
        epic_dictionary_list (list): Epic task dictionary

    Returns:
        list: CSV row [timestamp, pageTitle, sideBar, orderBox, textBox, popUpBox]
    """
    parsedInfo = []

    # Extract numeric portion from labelFile
    base_name = os.path.splitext(labelFile)[0]
    match = re.search(r'(\d+)', base_name)
    frameNum = match.group(1) if match else "0"

    sec = int(frameNum) - 1
    timestamp = timedelta(seconds=sec)
    parsedInfo.append(str(timestamp))

    # pageTitle logic
    pageTitle = ""
    if "task_title" in NAMES:
        pageTitle = getText("task_title", labels, labelFile, frames_location, epic_dictionary_list)

    # Fallback if no pageTitle detected
    if pageTitle == "" and "page_title" in NAMES:
        if checkPresent("login", labels):
            pageTitle = "Login"
        else:
            if checkPresent("task_bar_active_tab", labels):
                pageTitle = getText("task_bar_active_tab", labels, labelFile, frames_location, epic_dictionary_list)
            if checkPresent("side_bar_active_tab", labels):
                side_act = getText("side_bar_active_tab", labels, labelFile, frames_location, epic_dictionary_list)
                if side_act:
                    pageTitle = side_act

    # Rename note -> Note Entry/Review
    p_lower = pageTitle.lower()
    if "note" in p_lower:
        if checkPresent("text_box", labels) or "edit" in p_lower:
            pageTitle = "Note Entry"
        else:
            pageTitle = "Note Review"
    # Rename order -> Order Entry/Review
    if "order" in p_lower:
        if checkPresent("order_box", labels):
            pageTitle = "Order Entry"
        else:
            pageTitle = "Order Review"

    parsedInfo.append(pageTitle)

    # sideBar
    sideBar = ""
    if "side_bar_active_tab" in NAMES:
        sideBar = getText("side_bar_active_tab", labels, labelFile, frames_location, epic_dictionary_list)
    parsedInfo.append(sideBar)

    # orderBox presence
    orderBox = checkPresent("order_box", labels)
    parsedInfo.append(orderBox)

    # textBox presence
    textBox = checkPresent("text_box", labels)
    parsedInfo.append(textBox)

    # popUpBox presence
    popUpBox = checkPresent("pop_up_box", labels)
    parsedInfo.append(popUpBox)

    return parsedInfo


################################################################################
# 9) morphological_and_bilateral
################################################################################
def morphological_and_bilateral(image, kernel_size, d, sColor, sSpace):
    """
    Apply morphological opening and bilateral filtering.

    Parameters:
        image: Input image
        kernel_size (int): Morphological kernel size
        d (int): Bilateral filter diameter
        sColor (int): Bilateral filter sigma color
        sSpace (int): Bilateral filter sigma space

    Returns:
        Filtered image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    filtered = cv2.bilateralFilter(opened, d, sColor, sSpace)
    return filtered


################################################################################
# 10) detectActivity - Next frame MSE logic
################################################################################
MAX_BILAT_D = 15
MAX_BILAT_SCOLOR = 150
MAX_BILAT_SSPACE = 100

def detectActivity(
    images_path,
    height,
    width,
    kernel_size=5,
    scale_factor_d=2e-05,
    scale_factor_sColor=4e-06,
    scale_factor_sSpace=4e-05,
):
    """
    Detect activity using mean squared error between consecutive frames.

    Parameters:
        images_path (str): Directory containing frames
        height (int): Frame height
        width (int): Frame width
        kernel_size (int): Morphological kernel size
        scale_factor_d (float): Bilateral filter d scaling
        scale_factor_sColor (float): Bilateral filter sigma color scaling
        scale_factor_sSpace (float): Bilateral filter sigma space scaling

    Returns:
        list: MSE values for each frame
    """
    frame_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(".png")])
    frame_count = len(frame_files)
    if frame_count == 0:
        print(f"[WARNING] No frames in {images_path}")
        return []

    images_filtered = []
    video_size = height * width

    d = max(1, min(int(video_size * scale_factor_d), MAX_BILAT_D))
    sColor = max(1, min(int(video_size * scale_factor_sColor), MAX_BILAT_SCOLOR))
    sSpace = max(1, min(int(video_size * scale_factor_sSpace), MAX_BILAT_SSPACE))

    for i, fname in enumerate(frame_files):
        fpath = os.path.join(images_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            images_filtered.append(None)
        else:
            filtered = morphological_and_bilateral(img, kernel_size, d, sColor, sSpace)
            images_filtered.append(filtered)

    activity_list = [0]*frame_count
    for i in range(frame_count - 1):
        if images_filtered[i] is None or images_filtered[i+1] is None:
            activity_list[i] = 0
        else:
            mse_val = np.mean((images_filtered[i+1] - images_filtered[i])**2)
            activity_list[i] = mse_val

    # Last frame => 0
    activity_list[frame_count - 1] = 0
    return activity_list


################################################################################
# 11) Crop logic (mode crop, overwrite frames)
################################################################################
def compute_crop_box(frame, black_threshold=30):
    """
    Compute crop box to remove black borders from frame.

    Parameters:
        frame: Input frame
        black_threshold (int): Threshold for black pixel detection

    Returns:
        tuple: (x, y, width, height) crop box
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
    h, w = bin_img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    # Flood fill from each corner
    cv2.floodFill(bin_img, mask, (0, 0), 255)
    cv2.floodFill(bin_img, mask, (w-1, 0), 255)
    cv2.floodFill(bin_img, mask, (0, h-1), 255)
    cv2.floodFill(bin_img, mask, (w-1, h-1), 255)
    floodfill_inverted = cv2.bitwise_not(bin_img)
    if cv2.countNonZero(floodfill_inverted) < 1:
        return (0, 0, w, h)
    contours, _ = cv2.findContours(floodfill_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(largest_contour)
    return (x, y, ww, hh)


def compute_mode_crop_box(frame_dir, black_threshold=30):
    """
    Compute most common crop box across all frames.

    Parameters:
        frame_dir (str): Directory containing frames
        black_threshold (int): Threshold for black pixel detection

    Returns:
        tuple: Mode crop box (x, y, width, height)
    """
    crop_boxes = []
    for f in sorted(os.listdir(frame_dir)):
        if f.lower().endswith(".png"):
            fp = os.path.join(frame_dir, f)
            frame = cv2.imread(fp)
            if frame is None:
                continue
            crop_box = compute_crop_box(frame, black_threshold)
            crop_boxes.append(crop_box)
    if not crop_boxes:
        return None
    counter = Counter(crop_boxes)
    mode_box, _ = counter.most_common(1)[0]
    return mode_box


def apply_mode_crop(frame_dir, mode_box):
    """
    Apply crop box to all frames in directory.

    Parameters:
        frame_dir (str): Directory containing frames
        mode_box (tuple): Crop box (x, y, width, height)
    """
    x, y, ww, hh = mode_box
    for f in sorted(os.listdir(frame_dir)):
        if f.lower().endswith(".png"):
            fp = os.path.join(frame_dir, f)
            frame = cv2.imread(fp)
            if frame is None:
                continue
            cropped = frame[y:y+hh, x:x+ww]
            cv2.imwrite(fp, cropped)


################################################################################
# 12) getSize(video) via OpenCV
################################################################################
def getSize(video_path):
    """
    Get video dimensions.

    Parameters:
        video_path (str): Path to video file

    Returns:
        tuple: (width, height)
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


################################################################################
# 13) Consolidate function
################################################################################
def get_task_dictionary_match(string: str):
    """
    Match task name to task dictionary categories.

    Parameters:
        string (str): Task name to match

    Returns:
        str: Best matching category
    """
    best_key = "Other"
    best_score = 0
    for k, synonyms in task_dictionary.items():
        for val in synonyms:
            score = difflib.SequenceMatcher(None, string.lower(), val.lower()).ratio()
            if score > best_score:
                best_score = score
                best_key = k
    return best_key if best_score > 0.8 else "Other"


def get_task(index, data_frame):
    """
    Get task category for given frame index.

    Parameters:
        index (int): Frame index
        data_frame: DataFrame with detections

    Returns:
        str: Task category
    """
    page_title = str(data_frame["page_title"][index])
    side_bar   = str(data_frame["sidebar"][index])
    matched_pt = get_task_dictionary_match(page_title)
    matched_sb = get_task_dictionary_match(side_bar)
    if matched_pt == "Other" and matched_sb != "Other":
        return matched_sb
    return matched_pt


def next_frames_titles_same(i, data, task_change_threshold):
    """
    Check if next N frames have same task title.

    Parameters:
        i (int): Current index
        data: DataFrame
        task_change_threshold (int): Number of frames to check

    Returns:
        bool: True if all titles are same
    """
    subset = set(data["page_title"][i : i + task_change_threshold])
    return len(subset) == 1


def enough_rows(i, data, threshold):
    """
    Check if enough rows remain for threshold check.

    Parameters:
        i (int): Current index
        data: DataFrame
        threshold (int): Number of rows needed

    Returns:
        bool: True if enough rows remain
    """
    return i + threshold <= len(data)


def get_state_from_mse(mse_val, threshold=0.2):
    """
    Determine activity state from MSE value.

    Parameters:
        mse_val (float): MSE value
        threshold (float): MSE threshold

    Returns:
        bool: True if active, False if inactive
    """
    return mse_val > threshold


def consolidate(input_csv, mse_threshold=0.2, task_change_threshold=2, threshold_of_inactivity=5):
    """
    Consolidate frame-level detections into activity periods.

    This function implements the consolidation methodology described in paper Section 2.4.3.

    Parameters:
        input_csv (str): Path to CSV with frame-level detections
        mse_threshold (float): MSE threshold for activity detection (default: 0.2, from Bayesian optimization)
        task_change_threshold (int): Number of frames to confirm task change (default: 2, per paper Section 2.4.3)
        threshold_of_inactivity (int): Consecutive seconds for inactivity (default: 5, per paper Section 2.4.3)

    Returns:
        None: Writes consolidated CSV to output directory
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"[WARNING] No data in {input_csv}. Cannot consolidate.")
        return

    # Convert time_stamp to integer seconds
    df["time_stamp"] = pd.to_timedelta(df["time_stamp"]).dt.total_seconds().astype(int)

    prev_title = ""
    prev_state = inactive
    consecutive = 0
    active_time = 0
    inactive_time = 0
    output_rows = []

    try:
        start_time = df["time_stamp"].iloc[0]
    except:
        start_time = 0

    for i in range(len(df)):
        curr_time  = df["time_stamp"].iloc[i]
        curr_title = get_task(i, df)
        curr_mse   = df["%change from previous frame"].iloc[i]
        curr_state = get_state_from_mse(curr_mse, threshold=mse_threshold)

        task_change = False
        # Check if the task changed
        if curr_title != prev_title:
            if enough_rows(i, df, task_change_threshold):
                if next_frames_titles_same(i, df, task_change_threshold):
                    # Finalize previous block
                    if prev_state == active:
                        active_time += consecutive
                    else:
                        inactive_time += consecutive
                    output_rows.append([start_time, prev_title, active_time, inactive_time])

                    task_change = True
                    active_time, inactive_time = 0, 0
                    start_time = curr_time
                    consecutive = 1
                    # Force new block to be active once the title changes
                    curr_state = True
                    curr_title = get_task(i, df)
                else:
                    curr_title = prev_title

        # Check if the state (active/inactive) changed
        if not task_change:
            if curr_state != prev_state:
                if enough_rows(i, df, threshold_of_inactivity):
                    next_states = df["%change from previous frame"].iloc[i : i + threshold_of_inactivity]
                    all_inactive = all(ns <= mse_threshold for ns in next_states)
                    all_active   = all(ns > mse_threshold  for ns in next_states)
                    if all_inactive or all_active:
                        # Finalize previous
                        if prev_state == active:
                            active_time += consecutive
                        else:
                            inactive_time += consecutive
                        # Start a new block
                        consecutive = 1
                    else:
                        # Revert to old state
                        curr_state = prev_state
                        consecutive += 1
                else:
                    curr_state = prev_state
                    consecutive += 1
            else:
                consecutive += 1

        prev_state = curr_state
        prev_title = curr_title

    # Finalize the last block
    if prev_state == active:
        active_time += consecutive
    else:
        inactive_time += consecutive

    output_rows.append([start_time, prev_title, active_time, inactive_time])

    out_df = pd.DataFrame(output_rows, columns=["time_start", "title", "time_active", "time_inactive"])

    # Post-processing: First active "Other" block is "Login"
    found_first_active_other = False
    for idx in out_df.index:
        title_val = out_df.at[idx, "title"]
        act_time  = out_df.at[idx, "time_active"]

        if title_val == "Other" and act_time > 0 and not found_first_active_other:
            out_df.at[idx, "title"] = "Login"
            found_first_active_other = True

    # Create consolidated folder
    base_dir = os.getcwd()
    consolidated_dir = os.path.join(base_dir, "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv   = os.path.join(consolidated_dir, f"{base_name}_parsed.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Consolidation done => {out_csv}")


################################################################################
# Main Pipeline
################################################################################
def main():
    """
    Main pipeline for EHR activity detection.

    Processes videos to detect EHR tasks and activity periods:
    1. Extract frames at 1 fps
    2. Crop black borders
    3. Run YOLO detection
    4. Perform OCR and task classification
    5. Calculate frame-to-frame MSE for activity detection
    6. Consolidate into activity periods
    """
    print("[INFO] Starting main pipeline...")

    # Load epic dictionary
    epic_dictionary = load_epic_dictionary(epic_dictionary_path)
    if not os.path.exists(source_path):
        print(f"[ERROR] Source path {source_path} does not exist.")
        return

    # Gather .mp4 files in source path
    videos = [f for f in os.listdir(source_path) if f.lower().endswith(".mp4")]
    if not videos:
        print("[WARNING] No .mp4 videos found in source path.")
        return

    # Process each video
    for vid in videos:
        print(f"\n[INFO] Processing video => {vid}")
        video_stem = os.path.splitext(vid)[0]
        video_path = os.path.join(source_path, vid)

        # Get original dimensions
        orig_w, orig_h = getSize(video_path)
        print(f"[INFO] {vid} => dimension (w={orig_w}, h={orig_h})")

        # Create output folder for this video
        video_out_dir = os.path.join(source_path, video_stem)
        os.makedirs(video_out_dir, exist_ok=True)

        # Extract frames
        frame_dir = os.path.join(video_out_dir, IMAGES_SUBDIR)
        extract_frames(video_path, frame_dir)

        # Determine mode crop and apply
        mode_box = compute_mode_crop_box(frame_dir, black_threshold=30)
        if mode_box:
            print(f"[INFO] Mode crop determined: {mode_box}")
            apply_mode_crop(frame_dir, mode_box)
            cropped_w, cropped_h = mode_box[2], mode_box[3]
        else:
            print("[WARNING] No mode crop found; proceeding without cropping.")
            cropped_w, cropped_h = orig_w, orig_h

        # YOLO detection on cropped frames
        yolo_output_dir = os.path.join(video_out_dir, YOLO_OUTPUT_SUBDIR)
        runModel(
            path_to_images=frame_dir,
            yolo_output_path=yolo_output_dir,
            model_weights=model_path,
            conf_threshold=config['yolo']['confidence_threshold'],
            imgsz=config['yolo']['image_size']
        )

        # Create CSV with detections
        out_csv_path = os.path.join(video_out_dir, f"{video_stem}.csv")
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

        print(f"[INFO] Wrote detections-based CSV => {out_csv_path}")

        # Compute next-frame MSE on cropped frames
        print("[INFO] Calculating next-frame MSE ...")
        activity_vals = detectActivity(frame_dir, cropped_h, cropped_w)

        # Insert activity values into CSV
        df = pd.read_csv(out_csv_path)
        min_len = min(len(df), len(activity_vals))
        df["%change from previous frame"] = activity_vals[:min_len]
        df.to_csv(out_csv_path, index=False)

        # Consolidate tasks & active/inactive states
        print("[INFO] Consolidating tasks & active states ...")
        consolidate(
            out_csv_path,
            mse_threshold=config['activity_detection']['mse_threshold'],
            task_change_threshold=config['task_classification']['task_change_threshold'],
            threshold_of_inactivity=config['activity_detection']['threshold_of_inactivity']
        )

    print("\n[INFO] Completed pipeline for all videos. End of main().")


if __name__ == "__main__":
    main()
