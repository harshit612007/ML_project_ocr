"""
Authentic Palm-Leaf Manuscript Processing & Character Extraction Module.
Processes real historical Palmyra palm-leaf manuscripts (e.g. Tolkappiyam, Cirupancamulam),
restores them with Sauvola binarization, segments individual character crops,
and runs classical ML inference on authentic historical palm-leaf writing.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import config
import preprocess
import predict


def process_historical_manuscript(
    manuscript_path: str = "manuscripts/tolkappiyam_palm_leaf.jpg",
    output_dir: str = "results/manuscript_analysis",
    max_characters_to_extract: int = 12
):
    """
    End-to-End Pipeline on a Genuine Historical Palm-Leaf Manuscript:
      1. Loads real high-resolution palm leaf strip (e.g. Tolkappiyam)
      2. Performs Sauvola adaptive binarization to isolate incised soot ink
      3. Uses classical morphological connected components to locate character bounding boxes
      4. Extracts individual 64x64 standardized character crops
      5. Runs trained classical ML recognition on each extracted historical character
      6. Generates publication-ready visual analysis figure
    """
    leaf_path = Path(manuscript_path)
    if not leaf_path.exists():
        raise FileNotFoundError(f"Manuscript image not found: {leaf_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    extracted_chars_dir = out_path / "extracted_characters"
    extracted_chars_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" AUTHENTIC TAMIL PALM-LEAF MANUSCRIPT PROCESSING PIPELINE")
    print(f" Source Manuscript: {leaf_path.name}")
    print("=" * 70)

    # 1. Load real manuscript image
    img_bgr = cv2.imread(str(leaf_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read manuscript image: {leaf_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"  [1/5] Loaded manuscript strip: {w} x {h} pixels (Aspect ratio: {w/h:.1f}:1)")

    # 2. Preprocess real palm leaf using CLAHE and Sauvola binarization
    print("  [2/5] Applying Sauvola adaptive binarization to historical leaf...")
    gray = preprocess.to_grayscale(img_rgb)
    denoised = cv2.GaussianBlur(gray, (3, 3), 1.0)
    clahe_enh = preprocess.enhance_contrast_clahe(denoised, clip_limit=2.5)
    binarized = preprocess.binarize_palm_leaf(clahe_enh, method="sauvola")
    cleaned = preprocess.morphological_cleanup(binarized)

    # 3. Locate character candidates using connected components / contour analysis
    print("  [3/5] Segmenting character contours along the horizontal palm leaf...")
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours matching character proportions on a palm leaf
    min_h = int(h * 0.08)  # ~14 pixels
    max_h = int(h * 0.45)  # ~80 pixels
    min_w = int(h * 0.08)  # ~14 pixels
    max_w = int(h * 0.50)  # ~88 pixels

    candidate_boxes = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if min_h <= h_c <= max_h and min_w <= w_c <= max_w:
            candidate_boxes.append((x_c, y_c, w_c, h_c))

    # Sort left-to-right along the manuscript strip
    candidate_boxes = sorted(candidate_boxes, key=lambda b: (b[0] // 100, b[0]))
    print(f"  [3/5] Discovered {len(candidate_boxes)} character candidates across the leaf.")

    if len(candidate_boxes) == 0:
        # Fallback: sliding window across the leaf if contours were fragmented
        step = int(h * 0.8)
        win_size = int(h * 0.8)
        for x_start in range(20, w - win_size, step):
            candidate_boxes.append((x_start, int(h * 0.1), win_size, win_size))

    # Select top candidates to extract
    selected_boxes = candidate_boxes[:max_characters_to_extract]
    print(f"  [4/5] Extracting and classifying top {len(selected_boxes)} characters...")

    # Load trained inference pipeline
    model, feat_pipe, pca, class_names = predict.load_or_train_default_pipeline()

    # Create annotated manuscript overlay
    annotated_manuscript = img_rgb.copy()
    extracted_results = []

    for idx, (bx, by, bw, bh) in enumerate(selected_boxes, start=1):
        # Add slight padding around character
        pad = 4
        y1 = max(0, by - pad)
        y2 = min(h, by + bh + pad)
        x1 = max(0, bx - pad)
        x2 = min(w, bx + bw + pad)

        char_crop_rgb = img_rgb[y1:y2, x1:x2]
        char_crop_bin = cleaned[y1:y2, x1:x2]

        # Standardize crop to 64x64
        char_64_bin = cv2.resize(char_crop_bin, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
        char_64_rgb = cv2.resize(char_crop_rgb, config.IMG_SIZE, interpolation=cv2.INTER_AREA)

        # Save individual character file
        char_file_name = f"char_{idx:02d}_x{bx}.png"
        char_file_path = extracted_chars_dir / char_file_name
        cv2.imwrite(str(char_file_path), cv2.cvtColor(char_64_rgb, cv2.COLOR_RGB2BGR))

        # Predict using trained classical pipeline
        feat_vec = feat_pipe.transform(np.array([char_64_bin]))
        feat_pca = pca.transform(feat_vec)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat_pca)[0]
        else:
            probs = np.zeros(len(class_names))
            probs[model.predict(feat_pca)[0]] = 1.0

        top_idx = int(np.argmax(probs))
        pred_class = class_names[top_idx]
        pred_glyph = config.TAMIL_GLYPH_MAP.get(pred_class, "")
        conf = float(probs[top_idx])

        extracted_results.append({
            "Index": idx,
            "Box": (bx, by, bw, bh),
            "Crop_RGB": char_64_rgb,
            "Crop_Binary": char_64_bin,
            "Predicted_Class": pred_class,
            "Predicted_Glyph": pred_glyph,
            "Confidence": conf,
            "File": str(char_file_path)
        })

        # Draw green bounding box and label on manuscript
        cv2.rectangle(annotated_manuscript, (x1, y1), (x2, y2), (46, 204, 113), 2)
        cv2.putText(
            annotated_manuscript,
            f"#{idx}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (231, 76, 60),
            1,
            cv2.LINE_AA
        )

        print(f"    Char #{idx:02d} [x={bx:4d}]: Predicted -> {pred_class:<8} ({pred_glyph}) | Conf: {conf*100:5.1f}%")

    # 4. Generate Publication-Quality Manuscript Analysis Figure
    print("  [5/5] Generating publication-quality visual analysis card...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 2.2])

    # Row 1: Original Palm-Leaf Strip
    ax_raw = fig.add_subplot(gs[0])
    ax_raw.imshow(img_rgb)
    ax_raw.set_title(
        f"Authentic Historical Palm-Leaf Manuscript: {leaf_path.name} (Original RGB Color Scan)",
        fontsize=12,
        fontweight="bold",
        pad=6
    )
    ax_raw.axis("off")

    # Row 2: Sauvola Binarization with Detected Bounding Boxes
    ax_bin = fig.add_subplot(gs[1])
    ax_bin.imshow(cleaned, cmap="gray")
    for bx, by, bw, bh in selected_boxes:
        rect = plt.Rectangle((bx, by), bw, bh, edgecolor="#e74c3c", facecolor="none", linewidth=1.5)
        ax_bin.add_patch(rect)
    ax_bin.set_title(
        "Restored Inscription via Sauvola Adaptive Binarization with Character Segmentation Bounding Boxes",
        fontsize=12,
        fontweight="bold",
        pad=6
    )
    ax_bin.axis("off")

    # Row 3: Grid of Extracted Characters with Model Predictions
    sub_gs = gs[2].subgridspec(2, 6, hspace=0.35, wspace=0.25)
    for i, res in enumerate(extracted_results[:12]):
        row_i = i // 6
        col_i = i % 6
        ax_c = fig.add_subplot(sub_gs[row_i, col_i])
        ax_c.imshow(res["Crop_RGB"])
        glyph_str = f"({res['Predicted_Glyph']})" if res["Predicted_Glyph"] else ""
        ax_c.set_title(
            f"#{res['Index']}: {res['Predicted_Class']}\n{glyph_str} | {res['Confidence']*100:.1f}%",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50"
        )
        ax_c.axis("off")

    fig.suptitle(
        "Real Palmyra Palm-Leaf (*Panaiyolai*) Optical Character Recognition\n"
        "Classical Machine Learning Pipeline (Sauvola + Gabor + SIFT + PCA + Classifier)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    summary_card_path = out_path / "real_manuscript_recognition_card.png"
    plt.savefig(summary_card_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Visual recognition analysis saved to: {summary_card_path.resolve()}")
    print(f"[SUCCESS] Segmented characters saved to:        {extracted_chars_dir.resolve()}")
    print("=" * 70 + "\n")
    return summary_card_path, extracted_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process real historical Tamil palm-leaf manuscript.")
    parser.add_argument(
        "--manuscript",
        type=str,
        default="manuscripts/tolkappiyam_palm_leaf.jpg",
        help="Path to palm-leaf manuscript scan."
    )
    args = parser.parse_args()
    process_historical_manuscript(manuscript_path=args.manuscript)
