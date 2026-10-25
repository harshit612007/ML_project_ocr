"""
Palm-Leaf Manuscript Processing and Recognition Pipeline.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Implements the end-to-end manuscript processing pipeline from Section 3:
1. Grayscale Conversion (Figure 4)
2. Median Filtering (Figure 5)
3. Adaptive Mean Thresholding (Figure 6)
4. Contour-based Convex Hull Bounding-Box Segmentation (Figure 8)
5. Character Resizing to uniform 8x8 dimensions (Page 82:6)
6. GLCM Feature Extraction (Equations 21-27) & MABPN Recognition
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import joblib

import config
import preprocess
import features
import models


def process_manuscript_strip(
    manuscript_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_crops: int = 15
):
    """
    Processes an authentic historical palm-leaf manuscript strip using the paper's pipeline.
    """
    if manuscript_path is None:
        leaf_p = config.MANUSCRIPTS_DIR / "tolkappiyam_palm_leaf.jpg"
    else:
        leaf_p = Path(manuscript_path)
        if not leaf_p.is_absolute() and not leaf_p.exists():
            leaf_p = config.PROJECT_ROOT / manuscript_path

    if output_dir is None:
        out_p = config.RESULTS_DIR / "manuscript_crops"
    else:
        out_p = Path(output_dir)
        if not out_p.is_absolute():
            out_p = config.PROJECT_ROOT / output_dir
    if not leaf_p.exists():
        raise FileNotFoundError(f"Manuscript image not found: {leaf_p}")

    out_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print(" ANCIENT TAMIL PALM-LEAF MANUSCRIPT PROCESSING (Section 3 & Figures 4-8)")
    print(f" Source: {leaf_p.name}")
    print("=" * 76)

    # 1. Grayscale, Median Filter, Adaptive Mean Thresholding
    stages = preprocess.preprocess_palm_leaf_page(leaf_p)
    thresh = stages["adaptive_threshold"]
    print("  [1/4] Applied Grayscale, Median Filter, and Adaptive Mean Thresholding.")

    # 2. Contour-based Convex Hull Bounding-Box Segmentation (Figure 8)
    boxes = preprocess.segment_characters_convex_hull(thresh)
    print(f"  [2/4] Discovered {len(boxes)} character bounding boxes using Convex Hull contours.")

    # Load trained MABPN-GLCM model if available
    glcm_model_p = config.MODELS_DIR / "mabpn_glcm_model.pkl"
    classes_p = config.MODELS_DIR / "class_names.pkl"
    model = None
    if glcm_model_p.exists():
        model = joblib.load(glcm_model_p)
        class_names = joblib.load(classes_p) if classes_p.exists() else config.TAMIL_CLASSES_18

    # 3. Crop, Resize to 8x8, Extract GLCM, and Predict
    print("  [3/4] Resizing segmented character crops to 8x8 and extracting GLCM features...")
    gray_img = stages["grayscale"]
    crops_processed = 0

    for i, (x, y, w, h) in enumerate(boxes[:max_crops]):
        crop = gray_img[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        # Resize to uniform 8x8 dimension
        char_8x8 = preprocess.resize_character(crop, config.CHAR_SIZE)

        # Save extracted 8x8 character crop
        crop_file = out_p / f"char_{i+1:02d}_x{x}_y{y}.png"
        cv2.imwrite(str(crop_file), char_8x8)

        # Extract 6 GLCM features (Equations 21-27)
        feat = features.extract_glcm_features(char_8x8).reshape(1, -1)

        pred_text = ""
        if model is not None:
            pred_class = model.predict(feat)[0]
            glyph = config.TAMIL_GLYPH_MAP_18.get(pred_class, "")
            pred_text = f" -> Predicted: {pred_class} ({glyph})"

        print(f"     Crop {i+1:02d}: Bounding Box ({x}, {y}, {w}, {h}) | 8x8 crop saved{pred_text}")
        crops_processed += 1

    print(f"\n  [4/4] Successfully segmented and processed {crops_processed} character crops.")
    print(f"  Extracted 8x8 character crops saved in: {out_p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process palm-leaf manuscript using the reference paper pipeline")
    parser.add_argument("--manuscript", type=str, default="manuscripts/tolkappiyam_palm_leaf.jpg", help="Path to manuscript image")
    parser.add_argument("--output", type=str, default="results/manuscript_crops", help="Directory to save segmented crops")
    args = parser.parse_args()

    process_manuscript_strip(manuscript_path=args.manuscript, output_dir=args.output)
