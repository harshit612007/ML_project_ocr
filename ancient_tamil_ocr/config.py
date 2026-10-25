"""
Configuration settings for Ancient Tamil Palm-Leaf Character Recognition Pipeline.
Reference Paper:
  Poornima Devi. M and M. Sornam (2020), "Classification of Ancient Handwritten Tamil
  Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural
  Network with GLCM Features", ACM Transactions on Asian and Low-Resource Language
  Information Processing, Vol. 19, Iss. 6, Article 82.
  DOI: https://doi.org/10.1145/3406209
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

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
DENOISE_MEDIAN_KSIZE = 3         # Median filter for noise reduction (Figure 5)
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
GLCM_ANGLES = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0] # 0, 45, 90, 135 degrees

# =============================================================================
# EXPLICIT 18 MEYYEZHUTHU CLASS DEFINITIONS & FIGURE 12 FIVE-BIT MAPPING
# =============================================================================
# Figure 12 (Page 82:15): Characters with corresponding class labels and outputs
# The paper strictly classifies the 18 Tamil Consonants (Meyyezhuthu).
# Unknown labels (e.g., vowels '01_a') must be strictly rejected.

FIGURE_12_TABLE = {
    1:  {"name": "ka",   "glyph": "க", "code": [0, 0, 0, 0, 1]},
    2:  {"name": "nga",  "glyph": "ங", "code": [0, 0, 0, 1, 0]},
    3:  {"name": "cha",  "glyph": "ச", "code": [0, 0, 0, 1, 1]},
    4:  {"name": "gna",  "glyph": "ஞ", "code": [0, 0, 1, 0, 0]},
    5:  {"name": "ta",   "glyph": "ட", "code": [0, 0, 1, 0, 1]},
    6:  {"name": "nna",  "glyph": "ண", "code": [0, 0, 1, 1, 0]},
    7:  {"name": "tha",  "glyph": "த", "code": [0, 0, 1, 1, 1]},
    8:  {"name": "na",   "glyph": "ந", "code": [0, 1, 0, 0, 0]},
    9:  {"name": "pa",   "glyph": "ப", "code": [0, 1, 0, 0, 1]},
    10: {"name": "ma",   "glyph": "ம", "code": [0, 1, 0, 1, 0]},
    11: {"name": "ya",   "glyph": "ய", "code": [0, 1, 0, 1, 1]},
    12: {"name": "ra",   "glyph": "ர", "code": [0, 1, 1, 0, 0]},
    13: {"name": "la",   "glyph": "ல", "code": [0, 1, 1, 0, 1]},
    14: {"name": "va",   "glyph": "வ", "code": [0, 1, 1, 1, 0]},
    15: {"name": "zha",  "glyph": "ழ", "code": [0, 1, 1, 1, 1]},
    16: {"name": "lla",  "glyph": "ள", "code": [1, 0, 0, 0, 0]},
    17: {"name": "rra",  "glyph": "ற", "code": [1, 0, 0, 0, 1]},
    18: {"name": "nna2", "glyph": "ன", "code": [1, 0, 0, 1, 0]},
}

# Ordered canonical class names
TAMIL_CLASSES_18 = [f"{cls_num:02d}_{info['name']}" for cls_num, info in FIGURE_12_TABLE.items()]

# Exact 5-bit target codeword for each class index (1 to 18)
CLASS_TO_5BIT: Dict[int, List[int]] = {
    cls_num: info["code"] for cls_num, info in FIGURE_12_TABLE.items()
}

# Mapping from canonical name to 5-bit vector
NAME_TO_5BIT: Dict[str, np.ndarray] = {
    info["name"]: np.array(info["code"], dtype=np.float64) for cls_num, info in FIGURE_12_TABLE.items()
}

# Glyph mapping
TAMIL_GLYPH_MAP_18: Dict[str, str] = {
    f"{cls_num:02d}_{info['name']}": info["glyph"] for cls_num, info in FIGURE_12_TABLE.items()
}
for cls_num, info in FIGURE_12_TABLE.items():
    TAMIL_GLYPH_MAP_18[info["name"]] = info["glyph"]


def normalize_label(label: Any) -> str:
    """
    Normalizes a folder name, label, or class index into its canonical class name (e.g., '01_ka', 1, or 'ka' -> 'ka').
    Rejects labels that are not part of the 18 Meyyezhuthu classes (e.g., vowels '01_a').
    """
    if isinstance(label, (int, np.integer)):
        if 1 <= int(label) <= 18:
            return FIGURE_12_TABLE[int(label)]["name"]
        raise ValueError(f"Class index {label} out of range [1, 18].")

    cleaned = str(label).strip().lower()
    if cleaned.isdigit():
        idx = int(cleaned)
        if 1 <= idx <= 18:
            return FIGURE_12_TABLE[idx]["name"]

    # Remove leading number prefixes like '01_', '1_', '04_'
    cleaned = re.sub(r"^\d+_", "", cleaned)

    # Check if cleaned name matches any of the 18 classes
    for cls_num, info in FIGURE_12_TABLE.items():
        if cleaned == info["name"] or label == info["glyph"]:
            return info["name"]
        # Allow aliases like 'nja' for 'gna' or 'naa' for 'na'
        if cleaned in ("nja", "nya") and info["name"] == "gna":
            return "gna"
        if cleaned == "nna_alveolar" and info["name"] == "nna2":
            return "nna2"

    raise ValueError(
        f"Label '{label}' is not one of the 18 reference paper classes in Figure 12.\n"
        f"Allowed classes: {[info['name'] for info in FIGURE_12_TABLE.values()]}"
    )


def get_5bit_code(label: Any) -> np.ndarray:
    """
    Returns the exact 5-bit codeword for a character label or class index (1 to 18).
    Raises ValueError if label is not in Figure 12.
    """
    canonical_name = normalize_label(label)
    return NAME_TO_5BIT[canonical_name].copy()


def decode_5bit_to_canonical(output_vector: np.ndarray) -> str:
    """
    Decodes continuous 5-bit network output into the nearest canonical class name
    using minimum Euclidean distance to the 18 valid codewords in Figure 12.
    """
    vec = np.asarray(output_vector, dtype=np.float64).flatten()
    best_name = "ka"
    min_dist = float("inf")

    for cls_num, info in FIGURE_12_TABLE.items():
        codeword = np.array(info["code"], dtype=np.float64)
        dist = np.sum((vec - codeword) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_name = info["name"]

    return best_name


def decode_5bit_to_index(output_vector: np.ndarray) -> int:
    """
    Decodes continuous 5-bit network output into the nearest class index (1 to 18)
    using minimum Euclidean distance to the 18 valid codewords in Figure 12.
    """
    vec = np.asarray(output_vector, dtype=np.float64).flatten()
    best_idx = 1
    min_dist = float("inf")

    for cls_num, info in FIGURE_12_TABLE.items():
        codeword = np.array(info["code"], dtype=np.float64)
        dist = np.sum((vec - codeword) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_idx = cls_num

    return best_idx
