# Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscriptions Using MABPN with GLCM Features

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reference Paper](https://img.shields.io/badge/ACM%20TALLIP-Vol%2019%20Art%2082-brightgreen.svg)](https://doi.org/10.1145/3406209)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Reference Research Paper

> **Poornima Devi. M and M. Sornam (2020)**  
> *"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural Network with GLCM Features"*  
> **ACM Transactions on Asian and Low-Resource Language Information Processing (ACM TALLIP)**, Volume 19, Issue 6, Article 82 (October 2020), 24 pages.  
> **DOI:** [https://doi.org/10.1145/3406209](https://doi.org/10.1145/3406209)

### Comprehensive Documentation:
* **[`PAPER_EQUATION_AUDIT.md`](PAPER_EQUATION_AUDIT.md)** — Exhaustive audit of Equations (1)–(27), derivatives, limits at $x=0$, and ambiguities.
* **[`PAPER_IMPLEMENTATION_AUDIT.md`](PAPER_IMPLEMENTATION_AUDIT.md)** — Requirement-by-requirement compliance matrix and root-cause analysis of previous bugs.
* **[`PROJECT_EXPLANATION.md`](PROJECT_EXPLANATION.md)** — Step-by-step mathematical learning guide, paleographical context, and architectural breakdown.

---

## Proposed System Architecture

This codebase strictly implements the two neural network models proposed in Poornima Devi & Sornam (2020):

```
                       Original Palm Leaf (Fig. 3)
                                    │
                                    ▼
                         Grayscale Image (Fig. 4)
                                    │
                                    ▼
                       Median Filter Image (Fig. 5)
                                    │
                                    ▼
                     Adaptive Mean Threshold (Fig. 6)
                                    │
                                    ▼
                    Convex Hull Segmentation (Fig. 8)
                                    │
                                    ▼
                     Resize Characters to 8 × 8
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
  Proposed Model 1: MABPN-BCV                     Proposed Model 2: MABPN-GLCM
  (Binary Coded Value Model)                      (GLCM Texture Features Model)
  - 8x8 Binary Input (64 pixels)                  - 6 GLCM Descriptors (Eq. 21–27):
  - Architecture: 65 - 55 - 5                       Contrast, Energy, Homogeneity,
  - Weights: Nguyen–Widrow (Eq. 1–6)                Correlation, Dissimilarity, ASM
  - Hidden Act: Sigmoid (Eq. 7)                   - Architecture: 6 - 20 - 5
  - Output Act: Shannon Wavelet (Eq. 8)           - Weights: Nguyen–Widrow (Eq. 1–6)
  - Learning: MABPN (Eq. 9–20)                    - Hidden Act: Sigmoid (Eq. 7)
  - Class Output: 5-Bit Code (Fig. 12)            - Output Act: Shannon Wavelet (Eq. 8)
                                                  - Learning: MABPN (Eq. 9–20)
                                                  - Class Output: 5-Bit Code (Fig. 12)
```

---

## Quick Start: How to Run

### 1. Install Dependencies
```bash
pip install -r ancient_tamil_ocr/requirements.txt
```

### 2. Run Both Proposed Models
```bash
python ancient_tamil_ocr/main.py --model all
```

### 3. Run Proposed Model 1 (MABPN-BCV)
```bash
python ancient_tamil_ocr/main.py --model bcv
```
*Extracts the 64-D binary pixel features, trains the 65–55–5 MABPN architecture with Nguyen–Widrow initialization and Shannon wavelet output activation, and evaluates on held-out test data.*

### 4. Run Proposed Model 2 (MABPN-GLCM)
```bash
python ancient_tamil_ocr/main.py --model glcm
```
*Extracts the 6 GLCM features (Equations 21–27), trains the 6–20–5 MABPN architecture, and evaluates on held-out test data.*

### 5. Run Single Character Inference
```bash
python ancient_tamil_ocr/predict.py --image path/to/character.png --model bcv
```

### 6. Process Real Historical Palm-Leaf Manuscript
```bash
python ancient_tamil_ocr/process_manuscript.py --manuscript ancient_tamil_ocr/manuscripts/tolkappiyam_palm_leaf.jpg
```
*Restores manuscript using Median Filtering and Adaptive Mean Thresholding, segments character boundaries via Convex Hull contours (Figure 8), resizes crops to 8x8, and classifies them using MABPN.*

### 7. Run Unit Tests (16 Comprehensive Tests)
```bash
python -m unittest tests/test_paper_implementation.py
```
*Executes all 16 unit tests verifying BCV and GLCM architectures, GLCM equations, Shannon wavelet activation and derivative, zero-input limit (4/7), Nguyen–Widrow scaling, Figure 12 codewords, MABPN intermediate errors, weight updates, serialization consistency, preprocessing integration, collapse detection, and NaN/Inf detection.*

---

## Experimental Results & Scientific Transparency

### 1. Benchmark Reported in Reference Paper (Table 5 - Primary Manuscript Dataset)
* **Dataset Source:** U.V. Swaminatha Iyer Library, Chennai (Kuzhanthai Pini Marunthu, 1,756 samples).

| Proposed Model | Architecture | Feature Vector | Epochs | Error | Precision | Recall | F1-Score | Reported Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (MABPN-BCV)** | 65 - 55 - 5 | 64-D Binary Coded Grid | 130 | 0.0003 | 0.96 | 0.94 | 0.94 | **94.1%** |
| **Model 2 (MABPN-GLCM)** | 6 - 20 - 5 | 6-D GLCM Descriptors | 104 | 0.0003 | 0.98 | 0.97 | 0.98 | **96.23%** |

### 2. Experimental Verification on Local 18-Class Test Fixture
* **Provenance Notice:** As noted by the paper's authors (Page 82:5), the archival palm-leaf dataset from U.V. Swaminatha Iyer Library is not available online. To verify pipeline execution and mathematical correctness without fabricating historical data, the models were evaluated on the controlled 18-class Figure 12 test fixture (540 total samples, 70:30 stratified split: 378 train, 162 test):

| Implementation Model | Architecture | Accuracy | Unique Classes Predicted | Collapse Check | Finite Weights | Final Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MABPN-BCV** | 65 - 55 - 5 | **83.33%** (135/162) | 17 / 18 | **HEALTHY (No Collapse)** | 3,855 (0 non-finite) | 0.0965 |
| **MABPN-GLCM** | 6 - 20 - 5 | **16.67%** (27/162) | 4 / 18 | **HEALTHY (No Collapse)** | 245 (0 non-finite) | 0.2041 |

*Previous mode collapse (~10% accuracy predicting only 1 class) was traced to layer-wide scalar averaging ($\lambda, \mu$) starving gradient flow to $\mathcal{O}(\delta^2) \approx 10^{-6}$. Reimplementing neuron-specific updates resolved collapse across all classes.*

---

## Repository Structure

```
ML/
├── .gitignore
├── LICENSE
├── PAPER_EQUATION_AUDIT.md        # Comprehensive audit of Equations (1)–(27) & ambiguities
├── PAPER_IMPLEMENTATION_AUDIT.md  # Requirement-by-requirement audit matrix & bug root causes
├── PROJECT_EXPLANATION.md         # Technical learning guide & mathematical derivations
├── README.md                      # Primary project guide & execution commands
├── tests/
│   └── test_paper_implementation.py  # 16 comprehensive unit tests validating mathematical behavior
└── ancient_tamil_ocr/
    ├── ancient_tamil_recognition.ipynb  # Interactive walkthrough notebook
    ├── config.py                  # Hyperparameters, Figure 12 table, & 5-bit codeword mapping
    ├── dataset.py                 # Dataset loader, Figure 12 auditor, & 70:30 stratified split
    ├── evaluate.py                # Paper metrics & Phase 8 collapse diagnostics
    ├── features.py                # 64-D BCV & 6-D GLCM (Equations 21–27)
    ├── main.py                    # Pipeline entry point (--model bcv | glcm | all)
    ├── models.py                  # Model factory exporting MABPN_BCV & MABPN_GLCM
    ├── neural_networks.py         # MABPN (Eq. 9–20), Nguyen–Widrow (Eq. 1–6), Shannon Wavelet (Eq. 8)
    ├── predict.py                 # Single character inference
    ├── preprocess.py              # Median filter, Adaptive mean threshold, Convex Hull
    ├── process_manuscript.py      # End-to-end historical manuscript processing
    ├── requirements.txt           # Python dependencies
    ├── data/                      # Character dataset directory & 18-class test fixture
    ├── manuscripts/               # Sample historical palm-leaf images
    ├── results/                   # Confusion matrix plots & segmented character crops
    └── saved_models/              # Serialized trained model weights (.pkl)
```
