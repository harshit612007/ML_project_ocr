"""
Configuration settings for Ancient Tamil Palm-Leaf Character Recognition Pipeline.
Domain: Ancient Tamil Palm-Leaf Manuscripts (தமிழ் பனையோலை சுவடிகள் - Panaiyolai Suvadigal).
Methodology: Classical Machine Learning & Shallow Neural Networks (Strictly NO Deep Learning).
References:
  - Sundaram et al., "Offline Handwritten Tamil Character Recognition on Palm Leaf Manuscripts"
  - Nirmala & Palanisamy, "Tamil palm leaf character recognition using Gabor features and SVM"
  - Narang, Jindal & Ahuja (2020), Soft Computing 24(22) (SIFT + Gabor + PCA Pipeline)
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "saved_models"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for strict reproducibility
RANDOM_SEED = 42

# Image Processing Specifications
IMG_SIZE = (64, 64)  # Width, Height (Standardized 64x64 for manuscript characters)
DENOISE_METHOD = "gaussian"  # 'gaussian', 'median', or 'bilateral'
DENOISE_GAUSSIAN_KERNEL = (3, 3)
DENOISE_GAUSSIAN_SIGMA = 1.0
DENOISE_MEDIAN_KSIZE = 3

# Contrast Enhancement (CLAHE)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Binarization
BINARIZE_METHOD = "sauvola"  # 'sauvola' (best for degraded palm leaf) or 'otsu'
SAUVOLA_WINDOW_SIZE = 15
SAUVOLA_K = 0.2

# Morphological Operations
APPLY_MORPH_CLEANUP = True
MORPH_KERNEL_SIZE = (2, 2)

# Dataset Split
TRAIN_TEST_SPLIT_RATIO = 0.20  # 80% train, 20% test stratified

# Feature Extraction Settings
# 1. Gabor Filter Bank (Captures multi-directional rounded palm-leaf strokes)
GABOR_FREQUENCIES = [0.08, 0.16, 0.25, 0.35]  # Multi-frequency spatial scales
GABOR_NUM_ORIENTATIONS = 8  # 8 orientations: 0 to 7pi/8
GABOR_SIGMA_X = 2.0
GABOR_SIGMA_Y = 2.0

# 2. Dense SIFT
SIFT_GRID_STEP = 8  # Grid spacing in pixels
SIFT_KEYPOINT_SIZE = 12  # Scale of each keypoint
SIFT_BOVW_CLUSTERS = 64  # Vocabulary size for Bag-of-Visual-Words
SIFT_POOLING_METHOD = "bovw"  # 'bovw' with spatial quadrant pooling

# 3. HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# 4. Statistical Features
ZONING_GRID = (4, 4)  # 4x4 zoning
PROJECTION_BINS = 16

# Dimensionality Reduction
PCA_VARIANCE_RATIO = 0.95  # Retain 95% of cumulative explained variance

# Classification & Cross-Validation
CV_FOLDS = 10  # 10-fold cross validation

# 10 Classical Ancient Tamil Palm-Leaf Character Classes
BENCHMARK_CLASSES = [
    "01_a",    # அ (Short Vowel A)
    "02_aa",   # ஆ (Long Vowel Aa)
    "03_i",    # இ (Short Vowel I)
    "04_ka",   # க (Consonant Ka)
    "05_cha",  # ச (Consonant Cha)
    "06_ta",   # த (Consonant Ta/Tha)
    "07_pa",   # ப (Consonant Pa)
    "08_ma",   # ம (Consonant Ma)
    "09_ra",   # ர (Consonant Ra)
    "10_la",   # ல (Consonant La)
]

TAMIL_GLYPH_MAP = {
    "01_a": "அ",
    "02_aa": "ஆ",
    "03_i": "இ",
    "04_ka": "க",
    "05_cha": "ச",
    "06_ta": "த",
    "07_pa": "ப",
    "08_ma": "ம",
    "09_ra": "ர",
    "10_la": "ல",
}

# Aliases for backwards compatibility across modules
GLYPH_MAP = TAMIL_GLYPH_MAP
DEVANAGARI_GLYPH_MAP = TAMIL_GLYPH_MAP
