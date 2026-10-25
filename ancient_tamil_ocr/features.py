"""
Feature Extraction module for Ancient Tamil Palm-Leaf Character Recognition.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Implements ONLY the two feature representations proposed in the reference paper:
1. Model 1 (MABPN-BCV): Binary Coded Value (BCV) representation
   - 8x8 binary character image -> 64 binary values (Section 3.1.2)
2. Model 2 (MABPN-GLCM): Gray-Level Co-occurrence Matrix (GLCM) Features
   - Exactly six second-order GLCM statistical features (Section 3.1.3, Equations 21-27):
     1. Contrast (Equation 21)
     2. Energy (Equation 22)
     3. Homogeneity (Equation 23)
     4. Correlation (Equation 24)
     5. Dissimilarity (Equation 26)
     6. Angular Second Moment (ASM) (Equation 27)
"""

from typing import Tuple, List, Union
import numpy as np
import cv2
from skimage.feature import graycomatrix

import config
import preprocess


# =============================================================================
# 1. MODEL 1: BINARY CODED VALUE (BCV) FEATURE REPRESENTATION
# =============================================================================

def extract_binary_coded_features(
    img: np.ndarray,
    grid_size: Tuple[int, int] = config.CHAR_SIZE
) -> np.ndarray:
    """
    Proposed Model 1 (MABPN-BCV) Feature Extraction (Section 3.1.2, Page 82:8).
    "For this proposed model, 8 x 8 binary images were considered as input that
     contains 64-pixel values."
    
    Transforms the character image into an 8x8 binary representation and flattens
    it into a 64-dimensional binary feature vector x in {0.0, 1.0}^64.
    
    Parameters:
      img: Input character image (grayscale or RGB).
      grid_size: Standardized (8, 8) grid dimensions.
      
    Returns:
      1D NumPy array of length 64 with binary pixel values.
    """
    gray = preprocess.to_grayscale(img)
    if gray.shape != grid_size:
        resized = preprocess.resize_character(gray, grid_size)
    else:
        resized = gray
    
    # Binary conversion (foreground ink = 1.0, background = 0.0)
    binary = (resized > 127).astype(np.float64)
    return binary.flatten()


# =============================================================================
# 2. MODEL 2: GLCM STATISTICAL FEATURE EXTRACTION (6 FEATURES)
# =============================================================================

def extract_glcm_features(
    img: np.ndarray,
    distances: List[int] = config.GLCM_DISTANCES,
    angles: List[float] = config.GLCM_ANGLES,
    n_levels: int = 16
) -> np.ndarray:
    """
    Proposed Model 2 (MABPN-GLCM) Feature Extraction (Section 3.1.3, Equations 21-27).
    "segmented images were resized into 8 x 8 dimensions, then six GLCM features
     were extracted from the segmented character images. Those six features are
     contrast, energy, homogeneity, correlation, dissimilarity, and Angular
     Second Moment (ASM)."
     
    Dimension of feature space is exactly 6 (Page 82:11).
    
    Equations:
      1. Contrast (Eq 21):      C   = sum_{l,k} G_{l,k} * (l - k)^2
      2. Energy (Eq 22):        E   = sum_{l,k} G_{l,k}
      3. Homogeneity (Eq 23):   H   = sum_{l,k} G_{l,k} / (1 + (l - k)^2)
      4. Correlation (Eq 24):   Cor = sum_{l,k} G_{l,k} * ((l - mu_l)*(k - mu_k)) / sqrt(sigma_l^2 * sigma_k^2)
      5. Dissimilarity (Eq 26): D   = sum_{l,k} G_{l,k} * |l - k|
      6. ASM (Eq 27):           ASM = sum_{l,k} (G_{l,k})^2
      
    Parameters:
      img: Input character image (resized to 8x8).
      distances: Spatial offsets [1, 2] (Page 82:11).
      angles: Orientations [0, 45, 90, 135 degrees].
      n_levels: Number of quantized gray levels (default: 16 for 8x8 patches).
      
    Returns:
      1D NumPy array of shape (6,) containing [Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM].
    """
    gray = preprocess.to_grayscale(img)
    if gray.shape != config.CHAR_SIZE:
        resized = preprocess.resize_character(gray, config.CHAR_SIZE)
    else:
        resized = gray

    # Quantize 8x8 image into n_levels for stable co-occurrence matrix
    quantized = (resized.astype(np.float64) / 256.0 * n_levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, n_levels - 1)

    # Compute GLCM over distances and angles
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=True,
        normed=True
    )  # shape: (levels, levels, len(distances), len(angles))

    contrast_vals = []
    energy_vals = []
    homogeneity_vals = []
    correlation_vals = []
    dissimilarity_vals = []
    asm_vals = []

    # Grid indices l and k
    l_idx, k_idx = np.indices((n_levels, n_levels))

    for d_idx in range(len(distances)):
        for a_idx in range(len(angles)):
            P = glcm[:, :, d_idx, a_idx]
            total_sum = np.sum(P)
            if total_sum == 0:
                continue

            # Marginals for correlation
            mu_l = np.sum(l_idx * P)
            mu_k = np.sum(k_idx * P)
            var_l = np.sum(((l_idx - mu_l) ** 2) * P)
            var_k = np.sum(((k_idx - mu_k) ** 2) * P)
            std_l = np.sqrt(var_l)
            std_k = np.sqrt(var_k)

            # 1. Contrast (Equation 21)
            contrast = np.sum(P * ((l_idx - k_idx) ** 2))
            contrast_vals.append(contrast)

            # 2. Energy (Equation 22)
            energy = np.sum(P)
            energy_vals.append(energy)

            # 3. Homogeneity (Equation 23)
            homogeneity = np.sum(P / (1.0 + (l_idx - k_idx) ** 2))
            homogeneity_vals.append(homogeneity)

            # 4. Correlation (Equation 24)
            if std_l * std_k > 1e-12:
                corr = np.sum(P * ((l_idx - mu_l) * (k_idx - mu_k))) / (std_l * std_k)
            else:
                corr = 0.0
            correlation_vals.append(corr)

            # 5. Dissimilarity (Equation 26)
            dissimilarity = np.sum(P * np.abs(l_idx - k_idx))
            dissimilarity_vals.append(dissimilarity)

            # 6. Angular Second Moment (ASM) (Equation 27)
            asm = np.sum(P ** 2)
            asm_vals.append(asm)

    # Average each of the 6 features across all directional GLCMs
    feat_vector = np.array([
        float(np.mean(contrast_vals)) if contrast_vals else 0.0,
        float(np.mean(energy_vals)) if energy_vals else 0.0,
        float(np.mean(homogeneity_vals)) if homogeneity_vals else 0.0,
        float(np.mean(correlation_vals)) if correlation_vals else 0.0,
        float(np.mean(dissimilarity_vals)) if dissimilarity_vals else 0.0,
        float(np.mean(asm_vals)) if asm_vals else 0.0,
    ], dtype=np.float64)

    return feat_vector


def extract_features_batch(
    images: List[np.ndarray],
    model_type: str = "glcm"
) -> np.ndarray:
    """
    Extracts features for a batch of character images according to the paper model.
    model_type:
      - 'bcv':  Model 1 (MABPN-BCV), returns (N, 64)
      - 'glcm': Model 2 (MABPN-GLCM), returns (N, 6)
    """
    mt = model_type.lower()
    if mt in ("bcv", "mabpn-bcv", "bcv_model"):
        feats = [extract_binary_coded_features(img) for img in images]
    elif mt in ("glcm", "mabpn-glcm", "glcm_model"):
        feats = [extract_glcm_features(img) for img in images]
    else:
        raise ValueError(f"Unknown paper model type: '{model_type}'. Expected 'bcv' or 'glcm'.")
    return np.array(feats, dtype=np.float64)
