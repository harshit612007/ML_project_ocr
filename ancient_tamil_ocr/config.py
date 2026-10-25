"""
Configuration settings for Ancient Tamil Palm-Leaf Character Recognition Pipeline.
Reference Paper:
  Poornima Devi. M and M. Sornam (2020), "Classification of Ancient Handwritten Tamil
  Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural
  Network with GLCM Features", ACM Transactions on Asian and Low-Resource Language
  Information Processing, Vol. 19, Iss. 6, Article 82.
  DOI: https://doi.org/10.1145/3406209
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "saved_models"
MANUSCRIPTS_DIR = PROJECT_ROOT / "manuscripts"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42

# Preprocessing Specifications (Section 3 & Figures 4-6)
# Standardized character dimension in the paper is uniformly 8x8 (Page 82:6)
CHAR_SIZE = (8, 8)
IMG_SIZE = (8, 8)

# Preprocessing filters
DENOISE_MEDIAN_KSIZE = 3      # Median filter for noise reduction (Figure 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 15  # Adaptive mean thresholding (Figure 6)
ADAPTIVE_THRESH_C = 4

# Dataset Split (Section 4.1 & Table 4, 8)
# 70% Training (1,229 samples), 30% Testing (527 samples) for 1,756 characters
TRAIN_TEST_SPLIT_RATIO = 0.30

# Paper Proposed Architectures (Section 3.1.2 & 3.1.3)
# Model 1: MABPN-BCV
BCV_INPUT_DIM = 64     # 8x8 binary pixels
BCV_INPUT_NEURONS = 65 # 64 inputs + 1 bias neuron
BCV_HIDDEN_NEURONS = 55
BCV_OUTPUT_NEURONS = 5 # 5-bit class encoding

# Model 2: MABPN-GLCM
GLCM_INPUT_DIM = 6     # Exactly 6 GLCM features (Equations 21-27)
GLCM_INPUT_NEURONS = 6 # 6 features (+ bias)
GLCM_HIDDEN_NEURONS = 20
GLCM_OUTPUT_NEURONS = 5 # 5-bit class encoding

# Target convergence error threshold from Figure 9 flowchart
CONVERGENCE_ERROR_TARGET = 0.0001
REPORTED_MINIMIZED_ERROR = 0.0003

# GLCM Settings (Section 3.1.3)
GLCM_DISTANCES = [1, 2] # Distance 1 and 2
GLCM_ANGLES = [0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345] # 0, 45, 90, 135 degrees

# The 18 Meyyezhuthu (Body-Letters / Consonants) Character Classes (Figure 12)
# Exactly matching Figure 12: 18 classes with their 5-bit binary output codes
TAMIL_CLASSES_18 = [
    "01_ka",   # Class 1:  க் / க  -> 00001
    "02_nga",  # Class 2:  ங் / ங  -> 00010
    "03_cha",  # Class 3:  ச் / ச  -> 00011
    "04_gna",  # Class 4:  ஞ் / ஞ  -> 00100
    "05_ta",   # Class 5:  ட் / ட  -> 00101
    "06_nna",  # Class 6:  ண் / ண  -> 00110
    "07_tha",  # Class 7:  த் / த  -> 00111
    "08_na",   # Class 8:  ந் / ந  -> 01000
    "09_pa",   # Class 9:  ப் / ப  -> 01001
    "10_ma",   # Class 10: ம் / ம  -> 01010
    "11_ya",   # Class 11: ய் / ய  -> 01011
    "12_ra",   # Class 12: ர் / ர  -> 01100
    "13_la",   # Class 13: ல் / ல  -> 01101
    "14_va",   # Class 14: வ் / வ  -> 01110
    "15_zha",  # Class 15: ழ் / ழ  -> 01111
    "16_lla",  # Class 16: ள் / ள  -> 10000
    "17_rra",  # Class 17: ற் / ற  -> 10001
    "18_nna2", # Class 18: ன் / ன  -> 10010
]

# Exact 5-bit target codeword for each class index (1 to 18) as shown in Figure 12
CLASS_TO_5BIT = {
    1:  [0, 0, 0, 0, 1],
    2:  [0, 0, 0, 1, 0],
    3:  [0, 0, 0, 1, 1],
    4:  [0, 0, 1, 0, 0],
    5:  [0, 0, 1, 0, 1],
    6:  [0, 0, 1, 1, 0],
    7:  [0, 0, 1, 1, 1],
    8:  [0, 1, 0, 0, 0],
    9:  [0, 1, 0, 0, 1],
    10: [0, 1, 0, 1, 0],
    11: [0, 1, 0, 1, 1],
    12: [0, 1, 1, 0, 0],
    13: [0, 1, 1, 0, 1],
    14: [0, 1, 1, 1, 0],
    15: [0, 1, 1, 1, 1],
    16: [1, 0, 0, 0, 0],
    17: [1, 0, 0, 0, 1],
    18: [1, 0, 0, 1, 0],
}

# Unicode Glyph Mapping for the 18 Tamil Consonants
TAMIL_GLYPH_MAP_18 = {
    "01_ka": "க",
    "02_nga": "ங",
    "03_cha": "ச",
    "04_gna": "ஞ",
    "05_ta": "ட",
    "06_nna": "ண",
    "07_tha": "த",
    "08_na": "ந",
    "09_pa": "ப",
    "10_ma": "ம",
    "11_ya": "ய",
    "12_ra": "ர",
    "13_la": "ல",
    "14_va": "வ",
    "15_zha": "ழ",
    "16_lla": "ள",
    "17_rra": "ற",
    "18_nna2": "ன",
}

# Aliases for backwards compatibility
BENCHMARK_CLASSES = TAMIL_CLASSES_18
TAMIL_GLYPH_MAP = TAMIL_GLYPH_MAP_18
