"""
Preprocessing and Character Segmentation module for Ancient Tamil Palm-Leaf Manuscripts.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Implements the exact preprocessing pipeline described in Section 3 and Figures 4-8:
1. Grayscale Conversion (Figure 4)
2. Median Filtering for Denoising (Figure 5)
3. Adaptive Mean Thresholding for Binarization (Figure 6)
4. Contour-Based Convex Hull Bounding-Box Segmentation (Figure 8)
5. Character Resizing to Uniform 8x8 Dimension (Page 82:6)
6. Binary Conversion for the BCV representation (Page 82:8)
"""

from pathlib import Path
from typing import Union, Tuple, List, Dict
import numpy as np
import cv2

import config


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Step 1: Grayscale conversion (Section 3 & Figure 4).
    Converts color or multi-channel image to single-channel 8-bit grayscale.
    """
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.squeeze().astype(np.uint8)


def median_filter(img: np.ndarray, ksize: int = config.DENOISE_MEDIAN_KSIZE) -> np.ndarray:
    """
    Step 2: Median filtering (Section 3 & Figure 5).
    Removes impulse noise and fiber grain while preserving character stroke edges.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def adaptive_mean_threshold(
    img: np.ndarray,
    block_size: int = config.ADAPTIVE_THRESH_BLOCK_SIZE,
    c_val: int = config.ADAPTIVE_THRESH_C
) -> np.ndarray:
    """
    Step 3: Adaptive mean thresholding (Section 3 & Figure 6).
    Applies adaptive thresholding using local neighborhood mean to segment ink
    strokes from non-uniformly illuminated and aged palm-leaf patina.
    Returns: Binary image with foreground ink as 255 and background as 0.
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_val
    )


def resize_character(img: np.ndarray, target_size: Tuple[int, int] = config.CHAR_SIZE) -> np.ndarray:
    """
    Step 5: Character resizing (Section 3, Page 82:6).
    "The dimension of the resized character was uniformly set as 8 x 8."
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def binarize_to_bcv(char_8x8: np.ndarray) -> np.ndarray:
    """
    Binary Conversion for Model 1 (MABPN-BCV) (Section 3.1.2, Page 82:8).
    Converts 8x8 character image into binary values {0, 1},
    producing a 64-element binary vector.
    """
    if char_8x8.ndim == 3:
        char_8x8 = to_grayscale(char_8x8)
    if char_8x8.shape != (8, 8):
        char_8x8 = resize_character(char_8x8, (8, 8))
    # Threshold at median or 127
    binary = (char_8x8 > 127).astype(np.float64)
    return binary.flatten()


def segment_characters_convex_hull(
    binary_img: np.ndarray,
    min_dim: int = 10,
    max_dim: int = 150
) -> List[Tuple[int, int, int, int]]:
    """
    Step 4: Contour-Based Convex Hull Bounding Box Segmentation (Section 3 & Figure 8).
    
    Paper methodology:
      1. Detect character regions and find convex hull points within detected region.
      2. Trace contours to detect curve geometry and connected stroke boundaries.
      3. Draw bounding box enclosing the convex hull and segment characters.
    
    Returns:
      List of bounding boxes (x, y, w, h) sorted in reading order along the manuscript.
    """
    if binary_img.ndim == 3:
        binary_mask = to_grayscale(binary_img)
    else:
        binary_mask = binary_img.copy()

    # Find external contours of ink strokes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Compute Convex Hull around contour
        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(hull)

        # Filter out extreme noise artifacts
        if min_dim <= w <= max_dim and min_dim <= h <= max_dim:
            bounding_boxes.append((x, y, w, h))

    # Sort left-to-right along the manuscript strip
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 60, b[0]))
    return bounding_boxes


def preprocess_palm_leaf_page(
    image_input: Union[str, Path, np.ndarray],
    ksize: int = config.DENOISE_MEDIAN_KSIZE
) -> Dict[str, np.ndarray]:
    """
    Executes the full palm-leaf page preprocessing pipeline (Section 3, Figures 4-6):
      Original -> Grayscale -> Median Filter -> Adaptive Mean Threshold
    """
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_input}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.asarray(image_input, dtype=np.uint8).copy()

    gray = to_grayscale(img_rgb)
    filtered = median_filter(gray, ksize=ksize)
    thresholded = adaptive_mean_threshold(filtered)

    return {
        "original": img_rgb,
        "grayscale": gray,
        "median_filter": filtered,
        "adaptive_threshold": thresholded
    }


def preprocess_character_sample(
    char_img: np.ndarray,
    target_size: Tuple[int, int] = config.CHAR_SIZE
) -> np.ndarray:
    """
    Preprocesses an isolated character image according to the paper:
    Grayscale -> Median Filter -> Resize to 8x8.
    """
    gray = to_grayscale(char_img)
    filtered = median_filter(gray, ksize=3)
    resized = resize_character(filtered, target_size)
    return resized
