# Identifying Electronic Health Record Tasks and Activity Using Computer Vision

Code release for the paper: **"Identifying Electronic Health Record Tasks and Activity Using Computer Vision"**

**Authors:** Liem M. Nguyen, Amrita Sinha, Adam Dziorny, Daniel Tawfik

## Overview

This repository provides a computer vision-based model that can:
1. **Classify EHR tasks** being performed from screen recordings (9 high-level task categories)
2. **Detect task changes** - transitions between different task categories
3. **Quantify active-use time** - distinguish active periods from inactivity

The model uses YOLOv8 for UI element detection, Tesseract OCR for text recognition, and a frame comparison algorithm to measure activity.

## Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Core inference pipeline 
| `train_yolov8_kfold.py` | Train YOLOv8 with 5-fold CV
| `validate_results.py` | Calculate validation metrics 
| `grid_search.py` | Grid search parameter optimization
| `asymmetric_analysis.py` | Asymmetric threshold analysis
| `bayesian_optimization.py` | Bayesian hyperparameter search
| `run_pipeline.py` | Unified workflow


## Data

While we cannot provide any training or validation data due to intellectual property restrictions (EHR provider policies) and HIPAA regulations, we **can provide our trained model weights** (`best.pt`) which enable researchers to apply our methods to their own screen recordings.

**Available:**
- Trained YOLOv8x model weights (`best.pt`)
- Epic_Dictionary.txt containing commonly seen task title in the PICU (109 task names, no PHI)
- main.py:`task_dictionary` mapping Epic_Dictionary.txt titles to high level task categories (Chart_Review, In_Basket, Login, Navigation, Note_Entry, Note_Review, Order_Entry, Results_Review, Other) 
- Complete source code and analysis scripts

**Unvailable:**
- Training videos (Epic Sandbox - EHR provider IP restrictions)
- Validation videos (Real-world PICU recordings - HIPAA/PHI)
- Ground truth annotations (contains PHI)

## Citation
Nguyen LM, Sinha A, Dziorny A, Tawfik D. Identifying Electronic Health Record Tasks and Activity Using Computer Vision. Appl Clin Inform. 2025;16(5):1350-1358. doi:10.1055/a-2698-0841

## License
This code is released under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).
**Academic and non-commercial use only.** For commercial licensing inquiries, please contact the authors.

## AI Usage
Claude Code was used to support consolidation of scripts, formatting, and in-line documentation for this repository. 

## Contact

Corresponding Author: Liem M. Nguyen

Email: liemn@stanford.edu

Affiliation: Stanford University School of Medicine

## Acknowledgements

This study was conducted at Lucile Packard Children's Hospital Stanford and approved with waiver of consent by Stanford University's Institutional Review Board.
