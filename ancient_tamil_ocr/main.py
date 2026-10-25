"""
Main Orchestration Pipeline for Ancient Tamil Palm-Leaf Character Recognition.
Reference Paper:
  Poornima Devi. M and M. Sornam (2020), "Classification of Ancient Handwritten Tamil
  Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural
  Network with GLCM Features", ACM Transactions on Asian and Low-Resource Language
  Information Processing, Vol. 19, Iss. 6, Article 82.
  DOI: https://doi.org/10.1145/3406209

Implements ONLY the two proposed models from the paper:
  1. Model 1: MABPN-BCV (65-55-5 architecture with 64 binary coded inputs)
  2. Model 2: MABPN-GLCM (6-20-5 architecture with 6 GLCM features)

Usage:
  python main.py --model glcm   # Trains and evaluates Model 2 (MABPN-GLCM)
  python main.py --model bcv    # Trains and evaluates Model 1 (MABPN-BCV)
  python main.py --model all    # Evaluates both proposed models (default)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np

import config
import dataset
import preprocess
import features
import models
import evaluate


def print_banner():
    banner = r"""
========================================================================================
 ANCIENT TAMIL PALM-LEAF CHARACTER RECOGNITION USING MABPN
 Reference Paper: Poornima Devi. M & M. Sornam (2020), ACM TALLIP, Vol. 19, Art. 82
 Architecture: Modified Adaptive Backpropagation Neural Network (Nguyen-Widrow + Shannon)
 Models:
   1. MABPN-BCV  (65-55-5 Architecture, 64-D Binary Coded Input)
   2. MABPN-GLCM (6-20-5 Architecture, 6-D Second-Order GLCM Texture Features)
========================================================================================
"""
    print(banner)


def run_pipeline(
    model_choice: str = "all",
    learning_rate: float = 0.05,
    max_iter_bcv: int = 130,
    max_iter_glcm: int = 104,
    random_state: int = config.RANDOM_SEED
):
    print_banner()
    t_start = time.time()

    # 1. Dataset Loading (70:30 Stratified Split)
    print("\n[STAGE 1/4] Loading and verifying dataset...")
    X_train_raw, X_test_raw, y_train, y_test, class_names, status = dataset.load_character_dataset(
        data_dir=config.DATA_DIR,
        test_size=config.TRAIN_TEST_SPLIT_RATIO,
        random_state=random_state,
        allow_test_fixture=True
    )
    print(f"  -> Number of character classes: {len(class_names)}")
    print(f"  -> Training samples (70%):     {len(X_train_raw)}")
    print(f"  -> Testing samples (30%):      {len(X_test_raw)}")

    models_to_run = []
    if model_choice.lower() in ("bcv", "all"):
        models_to_run.append("bcv")
    if model_choice.lower() in ("glcm", "all"):
        models_to_run.append("glcm")

    for m_type in models_to_run:
        if m_type == "bcv":
            print("\n" + "=" * 80)
            print(" [STAGE 2/4 - MODEL 1] PREPARING MABPN-BCV (BINARY CODED VALUE)")
            print("=" * 80)
            print("  -> Preprocessing: Character Resizing to uniform 8x8 dimension")
            print("  -> Feature Extraction: 64 binary pixel values in {0.0, 1.0}^64")
            X_tr_feat = features.extract_features_batch(X_train_raw, model_type="bcv")
            X_te_feat = features.extract_features_batch(X_test_raw, model_type="bcv")
            print(f"  -> Feature vector dimension: {X_tr_feat.shape[1]} (Expected: 64)")

            print("\n[STAGE 3/4 - MODEL 1] Training MABPN-BCV...")
            print("  -> Architecture: 65 Input neurons (64 + 1 bias), 55 Hidden neurons, 5 Output neurons")
            print("  -> Weight Init: Nguyen-Widrow (Equations 1-6)")
            print("  -> Hidden Activation: Sigmoid (Equation 7)")
            print("  -> Output Activation: Shannon Wavelet f(x) = (sin(1.75x)-sin(0.75x))/(1.75x) (Eq 8)")
            print("  -> Learning Rule: Linear and Non-linear Adaptive Error Updates (Equations 9-20)")

            clf = models.MABPN_BCV(
                learning_rate=learning_rate,
                max_iter=max_iter_bcv,
                tol=config.CONVERGENCE_ERROR_TARGET,
                random_state=random_state
            )
            clf.fit(X_tr_feat, y_train)

            print("\n[STAGE 4/4 - MODEL 1] Evaluating MABPN-BCV on Held-Out Test Set...")
            y_pred = clf.predict(X_te_feat)
            metrics = evaluate.compute_paper_metrics(y_test, y_pred, class_names)
            final_err = clf.loss_history_[-1] if clf.loss_history_ else 0.0

            summary = evaluate.format_evaluation_summary(
                model_name="MABPN-BCV",
                metrics=metrics,
                epochs_trained=len(clf.loss_history_),
                final_error=final_err
            )
            print(summary)

            # Save artifacts
            evaluate.plot_paper_confusion_matrix(
                metrics["confusion_matrix"],
                class_names,
                model_name="MABPN-BCV"
            )
            joblib.dump(clf, config.MODELS_DIR / "mabpn_bcv_model.pkl")

        elif m_type == "glcm":
            print("\n" + "=" * 80)
            print(" [STAGE 2/4 - MODEL 2] PREPARING MABPN-GLCM (GRAY-LEVEL CO-OCCURRENCE MATRIX)")
            print("=" * 80)
            print("  -> Preprocessing: Character Resizing to uniform 8x8 dimension")
            print("  -> Feature Extraction: Exactly 6 second-order GLCM statistics (Equations 21-27):")
            print("     1. Contrast       2. Energy           3. Homogeneity")
            print("     4. Correlation    5. Dissimilarity    6. Angular Second Moment (ASM)")
            X_tr_feat = features.extract_features_batch(X_train_raw, model_type="glcm")
            X_te_feat = features.extract_features_batch(X_test_raw, model_type="glcm")
            print(f"  -> Feature vector dimension: {X_tr_feat.shape[1]} (Expected: 6)")

            print("\n[STAGE 3/4 - MODEL 2] Training MABPN-GLCM...")
            print("  -> Architecture: 6 Input neurons (+ 1 bias), 20 Hidden neurons, 5 Output neurons")
            print("  -> Weight Init: Nguyen-Widrow (Equations 1-6)")
            print("  -> Hidden Activation: Sigmoid (Equation 7)")
            print("  -> Output Activation: Shannon Wavelet f(x) = (sin(1.75x)-sin(0.75x))/(1.75x) (Eq 8)")
            print("  -> Learning Rule: Linear and Non-linear Adaptive Error Updates (Equations 9-20)")

            clf = models.MABPN_GLCM(
                learning_rate=learning_rate,
                max_iter=max_iter_glcm,
                tol=config.CONVERGENCE_ERROR_TARGET,
                random_state=random_state
            )
            clf.fit(X_tr_feat, y_train)

            print("\n[STAGE 4/4 - MODEL 2] Evaluating MABPN-GLCM on Held-Out Test Set...")
            y_pred = clf.predict(X_te_feat)
            metrics = evaluate.compute_paper_metrics(y_test, y_pred, class_names)
            final_err = clf.loss_history_[-1] if clf.loss_history_ else 0.0

            summary = evaluate.format_evaluation_summary(
                model_name="MABPN-GLCM",
                metrics=metrics,
                epochs_trained=len(clf.loss_history_),
                final_error=final_err
            )
            print(summary)

            # Save artifacts
            evaluate.plot_paper_confusion_matrix(
                metrics["confusion_matrix"],
                class_names,
                model_name="MABPN-GLCM"
            )
            joblib.dump(clf, config.MODELS_DIR / "mabpn_glcm_model.pkl")

    # Save class names
    joblib.dump(class_names, config.MODELS_DIR / "class_names.pkl")

    elapsed = time.time() - t_start
    print(f"\n[COMPLETE] Pipeline execution completed in {elapsed:.2f} seconds.")
    print(f"All model weights saved to: {config.MODELS_DIR.resolve()}")
    print(f"All confusion matrices saved to: {config.RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ancient Tamil Palm-Leaf Character Recognition (Poornima Devi & Sornam, 2020)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["bcv", "glcm", "all"],
        help="Proposed model to run: 'bcv' (Model 1), 'glcm' (Model 2), or 'all' (both)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate alpha (default: 0.05)"
    )
    parser.add_argument(
        "--epochs-bcv",
        type=int,
        default=130,
        help="Max training epochs for MABPN-BCV (default: 130 matching Table 5)"
    )
    parser.add_argument(
        "--epochs-glcm",
        type=int,
        default=104,
        help="Max training epochs for MABPN-GLCM (default: 104 matching Table 5)"
    )
    args = parser.parse_args()

    run_pipeline(
        model_choice=args.model,
        learning_rate=args.lr,
        max_iter_bcv=args.epochs_bcv,
        max_iter_glcm=args.epochs_glcm
    )
