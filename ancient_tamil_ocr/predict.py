"""
Single-character inference script for Ancient Tamil Palm-Leaf Character Recognition.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Usage:
  python predict.py --image path/to/character.png [--model glcm|bcv]
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import joblib
import numpy as np

import config
import preprocess
import features
import neural_networks


def predict_character(
    image_path: str,
    model_type: str = "glcm"
) -> str:
    """
    Predicts the Tamil character class for a single character crop using MABPN.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {img_p}")

    model_file = config.MODELS_DIR / f"mabpn_{model_type.lower()}_model.pkl"
    classes_file = config.MODELS_DIR / "class_names.pkl"

    if not model_file.exists():
        print(f"[INFO] Model file '{model_file.name}' not found in saved_models/. Running quick training...")
        import main
        main.run_pipeline(model_choice=model_type, max_iter_bcv=10, max_iter_glcm=10)

    model = joblib.load(model_file)
    class_names = joblib.load(classes_file) if classes_file.exists() else config.TAMIL_CLASSES_18

    # Load and preprocess character image to 8x8
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image at: {img_p}")
    resized = preprocess.preprocess_character_sample(img, target_size=config.CHAR_SIZE)

    # Feature extraction based on paper model
    if model_type.lower() == "bcv":
        feat = features.extract_binary_coded_features(resized).reshape(1, -1)
    else:
        feat = features.extract_glcm_features(resized).reshape(1, -1)

    predicted_label = model.predict(feat)[0]
    output_bits = model.predict_bits(feat)[0]

    glyph = config.TAMIL_GLYPH_MAP_18.get(predicted_label, "")
    print(f"\n=======================================================")
    print(f" Prediction Result for: {img_p.name}")
    print(f" Model Used:           MABPN-{model_type.upper()}")
    print(f" Predicted Class:      {predicted_label} ({glyph})")
    print(f" 5-Bit Output Vector:  {np.round(output_bits, 3).tolist()}")
    print(f"=======================================================\n")

    return predicted_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single ancient Tamil character using MABPN")
    parser.add_argument("--image", type=str, required=True, help="Path to character image file")
    parser.add_argument("--model", type=str, default="glcm", choices=["glcm", "bcv"], help="Model type to use")
    args = parser.parse_args()

    predict_character(args.image, model_type=args.model)
