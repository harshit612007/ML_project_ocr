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

For an exhaustive, step-by-step mathematical explanation, derivations with numerical examples, and audit details, see:
* **[`PROJECT_EXPLANATION.md`](PROJECT_EXPLANATION.md)** — Comprehensive technical learning guide and mathematical derivations.
* **[`PAPER_IMPLEMENTATION_AUDIT.md`](PAPER_IMPLEMENTATION_AUDIT.md)** — Item-by-item compliance audit table against the reference paper.

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
cd ancient_tamil_ocr
pip install -r requirements.txt
```

### 2. Run Proposed Model 2 (MABPN-GLCM)
```bash
python main.py --model glcm
```
*Extracts the 6 GLCM features (Equations 21–27), trains the 6-20-5 MABPN architecture with Nguyen–Widrow initialization and Shannon wavelet output activation, and evaluates on held-out test data.*

### 3. Run Proposed Model 1 (MABPN-BCV)
```bash
python main.py --model bcv
```
*Extracts the 64-D binary pixel features, trains the 65-55-5 MABPN architecture, and evaluates on held-out test data.*

### 4. Run Both Proposed Models
```bash
python main.py --model all
```

### 5. Run Single Character Inference
```bash
python predict.py --image path/to/character.png --model glcm
```

### 6. Process Real Historical Palm-Leaf Manuscript
```bash
python process_manuscript.py --manuscript manuscripts/tolkappiyam_palm_leaf.jpg
```
*Restores manuscript using Median Filtering and Adaptive Mean Thresholding, segments character boundaries via Convex Hull contours (Figure 8), resizes crops to 8x8, and classifies them using MABPN-GLCM.*

### 7. Run Unit Tests
```bash
python -m unittest tests/test_paper_implementation.py
```
*Executes all 10 unit tests verifying tensor dimensions, architecture layer counts, mathematical derivatives, Nguyen–Widrow weight scaling, 5-bit class encoding/decoding, and MABPN gradient updates.*

---

## Dataset Details & Reproducibility Notice

### Reference Paper Primary Dataset (Section 4.1.1):
* **Source:** U.V. Swaminatha Iyer Library, Chennai, Tamil Nadu, India.
* **Manuscript:** *Kuzhanthai Pini Marunthu* (Medical Palm-Leaf Manuscript, 175 leaves).
* **Character Set:** 18 Tamil Consonants (*Meyyezhuthu*, Figure 12).
* **Total Samples:** 1,756 characters.
* **Train / Test Partition:** 1,229 training (70%), 527 testing (30%).
* **Public Availability:** As stated by the authors on Page 82:5:
  *"The palm leaf dataset used in this work is a primary dataset that is not available online."*

Because this archival dataset is private, this repository:
1. Provides the complete mathematical and architectural implementation of both models.
2. Includes an authentic 18-class test fixture generator so all unit tests and pipelines run out-of-the-box.
3. Does not report paper results as our own experimental results, maintaining strict scientific transparency.

---

## Reference Paper Benchmark Results (Table 5 & Table 8)

| Proposed Model | Architecture | Feature Vector | Epochs | Error | Precision | Recall | F1-Score | Reported Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (MABPN-BCV)** | 65 - 55 - 5 | 64-D Binary Coded Grid | 130 | 0.0003 | 0.96 | 0.94 | 0.94 | **94.1%** |
| **Model 2 (MABPN-GLCM)** | 6 - 20 - 5 | 6-D GLCM Descriptors | 104 | 0.0003 | 0.98 | 0.97 | 0.98 | **96.23%** |

---

## Repository Structure

```
ML/
├── .gitignore
├── LICENSE
├── PAPER_IMPLEMENTATION_AUDIT.md        # Comprehensive requirement-by-requirement audit
├── PROJECT_EXPLANATION.md               # 18-section technical learning guide & mathematical derivations
├── README.md                            # Primary project guide
├── tests/
│   └── test_paper_implementation.py     # 10 unit tests verifying paper mathematics & architectures
└── ancient_tamil_ocr/
    ├── ancient_tamil_recognition.ipynb  # Interactive walkthrough notebook
    ├── config.py                        # Hyperparameters & 18-class 5-bit mapping
    ├── dataset.py                       # Dataset loader & 70:30 stratified split
    ├── evaluate.py                      # Accuracy, Error, Precision, Recall, F1, Confusion Matrix
    ├── features.py                      # 64-D BCV & 6-D GLCM (Equations 21–27)
    ├── main.py                          # Pipeline entry point (--model bcv | glcm | all)
    ├── models.py                        # Model factory exporting MABPN_BCV & MABPN_GLCM
    ├── neural_networks.py               # MABPN, Nguyen–Widrow (Eq. 1–6), Shannon Wavelet (Eq. 8)
    ├── predict.py                       # Single character inference
    ├── preprocess.py                    # Median filter, Adaptive mean threshold, Convex Hull
    ├── process_manuscript.py            # End-to-end historical manuscript processing
    ├── requirements.txt                 # Python dependencies
    ├── data/                            # Character dataset directory
    └── manuscripts/                     # Sample historical palm-leaf images
```
