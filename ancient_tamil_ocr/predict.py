"""
Inference script for single Ancient Tamil Palm-Leaf character recognition.
Usage:
  python predict.py --image path/to/character.png [--model saved_models/best_model.pkl]
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import preprocess
import features
import dataset
import models


def load_or_train_default_pipeline():
    """
    Loads saved inference artifacts (pipeline, PCA, classifier, class_names).
    If missing, trains a quick reference model on the dataset and saves artifacts.
    """
    model_file = config.MODELS_DIR / "best_model.pkl"
    pipeline_file = config.MODELS_DIR / "feature_pipeline.pkl"
    pca_file = config.MODELS_DIR / "pca_model.pkl"
    classes_file = config.MODELS_DIR / "class_names.pkl"

    if (
        model_file.exists() and
        pipeline_file.exists() and
        pca_file.exists() and
        classes_file.exists()
    ):
        clf = joblib.load(model_file)
        feat_pipe = joblib.load(pipeline_file)
        pca = joblib.load(pca_file)
        class_names = joblib.load(classes_file)
        return clf, feat_pipe, pca, class_names

    print("[INFO] No trained model found in 'saved_models/'. Training default pipeline...")
    X_tr_raw, _, y_tr, _, class_names, _ = dataset.load_character_dataset(auto_generate_if_missing=True)
    X_tr_prep = preprocess.preprocess_dataset(X_tr_raw)

    feat_pipe = features.FeaturePipeline(feature_type="glcm", random_state=config.RANDOM_SEED)
    X_tr_feat = feat_pipe.fit_transform(X_tr_prep)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=config.PCA_VARIANCE_RATIO, random_state=config.RANDOM_SEED)
    X_tr_pca = pca.fit_transform(X_tr_feat)

    # Use primary research neural network model MABPN (Paper 2: 2020)
    clf = models.get_classifiers(random_state=config.RANDOM_SEED)["MABPN (Paper 2: 2020)"]
    clf.fit(X_tr_pca, y_tr)

    # Save artifacts
    joblib.dump(clf, model_file)
    joblib.dump(feat_pipe, pipeline_file)
    joblib.dump(pca, pca_file)
    joblib.dump(class_names, classes_file)
    print(f"[INFO] Pipeline trained and saved to: {config.MODELS_DIR.resolve()}")

    return clf, feat_pipe, pca, class_names


def predict_single_character(
    image_path: str,
    model=None,
    feat_pipe=None,
    pca=None,
    class_names=None,
    save_card: bool = True,
    output_card_path: Optional[str] = None
):
    """
    Runs full inference pipeline on a single character image:
      1. Preprocesses image (Sauvola, CLAHE, morphological cleanup)
      2. Extracts Gabor + SIFT features
      3. Projects through fitted PCA
      4. Predicts class with confidence probabilities
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input character image not found: {img_path}")

    if model is None or feat_pipe is None or pca is None or class_names is None:
        model, feat_pipe, pca, class_names = load_or_train_default_pipeline()

    # 1. Load original image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Failed to decode image from: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocess
    preprocessed_img = preprocess.preprocess_image(img_rgb)

    # 3. Extract features & Standardize
    feat_vec = feat_pipe.transform(np.array([preprocessed_img]))

    # 4. Apply PCA
    feat_pca = pca.transform(feat_vec)

    # 5. Predict probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feat_pca)[0]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(feat_pca)[0]
        # Softmax normalization
        exp_dec = np.exp(decision - np.max(decision))
        probabilities = exp_dec / np.sum(exp_dec)
    else:
        pred_idx = model.predict(feat_pca)[0]
        probabilities = np.zeros(len(class_names))
        probabilities[pred_idx] = 1.0

    # Sort candidates
    top_indices = np.argsort(probabilities)[::-1]
    top_preds = [
        (
            class_names[idx],
            config.TAMIL_GLYPH_MAP.get(class_names[idx], ""),
            float(probabilities[idx])
        )
        for idx in top_indices[:3]
    ]

    best_class, best_glyph, best_conf = top_preds[0]

    # Print Report
    print("\n" + "="*55)
    print(" ANCIENT TAMIL PALM-LEAF CHARACTER RECOGNITION")
    print("="*55)
    print(f" Input Image:       {img_path.resolve()}")
    print(f" PREDICTED CLASS:   {best_class} ({best_glyph})")
    print(f" CONFIDENCE:        {best_conf * 100:.2f}%")
    print("-" * 55)
    print(" Top-3 Candidate Predictions:")
    for rank, (cls_name, glyph, conf) in enumerate(top_preds, start=1):
        glyph_str = f"({glyph})" if glyph else ""
        print(f"   [{rank}] {cls_name:<10} {glyph_str:<4} : {conf * 100:6.2f}%")
    print("="*55 + "\n")

    # 6. Save visual prediction card
    if save_card:
        if output_card_path is None:
            output_card_path = config.RESULTS_DIR / "prediction_output.png"
        output_card_path = Path(output_card_path)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

        # Subplot 1: Raw Palm-Leaf Image
        axes[0].imshow(cv2.resize(img_rgb, (128, 128)))
        axes[0].set_title("Input Palm-Leaf Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Subplot 2: Restored Preprocessed Binary
        axes[1].imshow(preprocessed_img, cmap="gray")
        axes[1].set_title("Restored (Sauvola + CLAHE)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Subplot 3: Confidence Distribution
        cand_labels = [f"{cls} ({glyph})" for cls, glyph, _ in top_preds[::-1]]
        cand_probs = [conf * 100 for _, _, conf in top_preds[::-1]]
        colors = ["#95a5a6", "#3498db", "#2ecc71"]

        bars = axes[2].barh(cand_labels, cand_probs, color=colors, height=0.55, edgecolor="#2c3e50")
        for bar in bars:
            w = bar.get_width()
            axes[2].text(w + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}%", va="center", fontweight="bold", fontsize=10)

        axes[2].set_xlim(0, 115)
        axes[2].set_xlabel("Confidence (%)", fontsize=10, fontweight="semibold")
        axes[2].set_title(f"Prediction: {best_class} ({best_glyph})", fontsize=11, fontweight="bold", color="#27ae60")
        axes[2].grid(axis="x", linestyle="--", alpha=0.5)

        fig.suptitle(f"Ancient Devanagari Character Classification Result", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_card_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Prediction visual card saved to: {output_card_path.resolve()}")

    return best_class, best_conf, top_preds


def main():
    parser = argparse.ArgumentParser(description="Predict Ancient Devanagari character class from image.")
    parser.add_argument("--image", type=str, required=False, help="Path to input character image file")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model file (.pkl)")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction card PNG")
    args = parser.parse_args()

    if args.image:
        predict_single_character(args.image, output_card_path=args.output)
    else:
        # If no image specified, pick an available test sample from data/
        data_samples = list(config.DATA_DIR.glob("*/*.png"))
        if len(data_samples) > 0:
            test_img = str(data_samples[0])
            print(f"[INFO] No --image provided. Running inference on test sample: {test_img}")
            predict_single_character(test_img, output_card_path=args.output)
        else:
            print("[INFO] No dataset found. Generating benchmark dataset first...")
            dataset.generate_benchmark_palm_leaf_dataset(num_samples_per_class=40)
            data_samples = list(config.DATA_DIR.glob("*/*.png"))
            predict_single_character(str(data_samples[0]), output_card_path=args.output)


if __name__ == "__main__":
    main()
