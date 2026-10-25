"""
Master Orchestration Pipeline for Ancient Tamil Palm-Leaf Character Recognition.
Domain: Ancient Tamil Palm-Leaf Manuscripts (தமிழ் பனையோலை சுவடிகள் - Panaiyolai Suvadigal).
Classical Machine Learning Pipeline (SIFT + Gabor + HOG + PCA + Classical MLP / SVM)
Strictly Classical ML & Shallow Neural Networks (Zero Deep Learning).

Pipeline Stages:
  1. Dataset Acquisition / Synthetic Palm-Leaf Benchmark Validation
  2. Preprocessing (Sauvola Binarization, CLAHE, Morphological Cleanup)
  3. Feature Extraction (Gabor Filter Bank + Dense SIFT BoVW)
  4. PCA Dimensionality Reduction (~95% Variance Retained)
  5. Stratified 10-Fold Cross-Validation & Classifier Benchmarking
  6. Held-out Test Set Evaluation & Error Analysis
  7. Component Ablation Study (With/Without Preprocessing, With/Without PCA)
  8. Publication Figures & Report Generation
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

import config
import dataset
import preprocess
import features
import models
import evaluate


def print_banner():
    banner = r"""
=============================================================================
   ANCIENT TAMIL (PALM-LEAF MANUSCRIPT) CHARACTER RECOGNITION
   Classical Machine Learning Pipeline (SIFT + Gabor + PCA + MLP / SVM)
   Domain: Panaiyolai Suvadigal (பனையோலை சுவடிகள்)
   Strictly Classical ML & Shallow Neural Networks (NO Deep Learning)
=============================================================================
"""
    print(banner)


def run_pipeline(
    feature_set: str = "gabor_sift",
    selected_classifier: str = "SVM-Poly",
    run_all_benchmarks: bool = True,
    run_ablation: bool = True,
    tune_params: bool = False,
    cv_folds: int = config.CV_FOLDS,
    use_pca: bool = True,
    pca_ratio: float = config.PCA_VARIANCE_RATIO
):
    print_banner()

    t_start = time.time()

    # -------------------------------------------------------------
    # STAGE 1: DATASET ACQUISITION & VALIDATION
    # -------------------------------------------------------------
    print("\n[STAGE 1/7] Loading and verifying Ancient Tamil palm-leaf dataset...")
    X_train_raw, X_test_raw, y_train, y_test, class_names, _ = dataset.load_character_dataset(
        data_dir=config.DATA_DIR,
        test_size=config.TRAIN_TEST_SPLIT_RATIO,
        random_state=config.RANDOM_SEED,
        auto_generate_if_missing=True
    )
    print(f"  -> Total Character Classes: {len(class_names)}")
    print(f"  -> Training Samples (80%): {len(X_train_raw)}")
    print(f"  -> Test Samples (20%):     {len(X_test_raw)}")
    print(f"  -> Tamil Glyph Map:        {config.TAMIL_GLYPH_MAP}")

    # -------------------------------------------------------------
    # STAGE 2: PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n[STAGE 2/7] Executing Palm-Leaf Restoration & Preprocessing...")
    print("  -> Operations: Grayscale -> Resize (64x64) -> Gaussian Denoise -> CLAHE -> Sauvola Binarize -> Morph Cleanup")
    X_train_prep = preprocess.preprocess_dataset(X_train_raw)
    X_test_prep = preprocess.preprocess_dataset(X_test_raw)

    # Save visual comparison figure
    sample_indices = [np.where(y_train == c)[0][0] for c in range(min(5, len(class_names)))]
    sample_raw = [X_train_raw[idx] for idx in sample_indices]
    sample_labels = [class_names[c] for c in range(len(sample_indices))]
    preprocess.visualize_preprocessing_steps(sample_raw, sample_labels)

    # -------------------------------------------------------------
    # STAGE 3: FEATURE EXTRACTION
    # -------------------------------------------------------------
    print(f"\n[STAGE 3/7] Extracting Feature Set: '{feature_set.upper()}'...")
    feat_pipeline = features.FeaturePipeline(
        feature_type=feature_set,
        random_state=config.RANDOM_SEED
    )
    X_train_feat = feat_pipeline.fit_transform(X_train_prep)
    X_test_feat = feat_pipeline.transform(X_test_prep)

    raw_feat_dim = X_train_feat.shape[1]
    print(f"  -> Raw Feature Vector Dimension: {raw_feat_dim} attributes per character.")

    # -------------------------------------------------------------
    # STAGE 4: DIMENSIONALITY REDUCTION (PCA)
    # -------------------------------------------------------------
    if use_pca:
        print(f"\n[STAGE 4/7] Applying Principal Component Analysis (Target Variance: {pca_ratio*100:.0f}%)...")
        pca_model = PCA(n_components=pca_ratio, random_state=config.RANDOM_SEED)
        X_train_final = pca_model.fit_transform(X_train_feat)
        X_test_final = pca_model.transform(X_test_feat)

        pca_feat_dim = X_train_final.shape[1]
        var_retained = float(np.sum(pca_model.explained_variance_ratio_)) * 100.0

        print(f"  -> Feature Length BEFORE PCA: {raw_feat_dim}")
        print(f"  -> Feature Length AFTER PCA:  {pca_feat_dim} (Reduced by {((raw_feat_dim - pca_feat_dim) / raw_feat_dim)*100:.1f}%)")
        print(f"  -> Cumulative Variance Preserved: {var_retained:.2f}%")

        # Plot explained variance curve
        evaluate.plot_pca_variance(pca_model, threshold=pca_ratio)
    else:
        print("\n[STAGE 4/7] PCA skipped as requested. Using full feature vectors.")
        pca_model = None
        X_train_final = X_train_feat
        X_test_final = X_test_feat
        pca_feat_dim = raw_feat_dim

    # -------------------------------------------------------------
    # STAGE 5: STRATIFIED 10-FOLD CROSS-VALIDATION
    # -------------------------------------------------------------
    print(f"\n[STAGE 5/7] Running Stratified {cv_folds}-Fold Cross-Validation on Training Data...")
    all_classifiers = models.get_classifiers(random_state=config.RANDOM_SEED)

    if selected_classifier not in all_classifiers and selected_classifier != "all":
        print(f"  [WARNING] Classifier '{selected_classifier}' not found. Defaulting to 'SVM-Poly'.")
        selected_classifier = "SVM-Poly"

    primary_clf = all_classifiers["SVM-Poly"] if selected_classifier == "all" else all_classifiers[selected_classifier]

    if tune_params:
        print(f"  -> Tuning hyperparameters via GridSearchCV for '{selected_classifier}'...")
        primary_clf, best_params = models.tune_hyperparameters(
            selected_classifier,
            primary_clf,
            X_train_final,
            y_train
        )
        print(f"  -> Best Hyperparameters: {best_params}")

    cv_results = models.run_tenfold_cross_validation(
        primary_clf,
        X_train_final,
        y_train,
        cv_folds=cv_folds,
        random_state=config.RANDOM_SEED
    )

    print(f"  -> Model: {selected_classifier if selected_classifier != 'all' else 'SVM-Poly'}")
    print(f"  -> 10-Fold CV Mean Accuracy: {cv_results['mean_accuracy']*100:.2f}% (+/- {cv_results['std_accuracy']*100:.2f}%)")
    print(f"  -> 10-Fold CV Mean Macro F1: {cv_results['mean_f1']*100:.2f}% (+/- {cv_results['std_f1']*100:.2f}%)")

    # -------------------------------------------------------------
    # STAGE 6: HELD-OUT TEST SET EVALUATION
    # -------------------------------------------------------------
    print("\n[STAGE 6/7] Evaluating Best Model on 20% Held-Out Test Set...")
    primary_clf.fit(X_train_final, y_train)
    test_preds = primary_clf.predict(X_test_final)

    metrics = evaluate.evaluate_predictions(y_test, test_preds, class_names)
    print(f"  -> Final Test Accuracy:     {metrics['accuracy']*100:.2f}%")
    print(f"  -> Final Test Precision:    {metrics['precision_macro']*100:.2f}%")
    print(f"  -> Final Test Recall:       {metrics['recall_macro']*100:.2f}%")
    print(f"  -> Final Test Macro F1:     {metrics['f1_macro']*100:.2f}%")

    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save Confusion Matrix
    evaluate.plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        title=f"Confusion Matrix ({selected_classifier} + {feature_set.upper()})"
    )

    # Save Per-Class Accuracy
    df_per_class = evaluate.plot_per_class_accuracy(metrics["per_class_accuracy"], class_names)

    # Error Analysis: Most-Confused Pairs
    df_confused = evaluate.identify_most_confused_pairs(metrics["confusion_matrix"], class_names, top_n=5)
    print("\nTop Most-Confused Character Pairs:")
    if len(df_confused) > 0:
        for _, row in df_confused.iterrows():
            print(f"  * True: {row['True_Class']} ({row['True_Glyph']}) confused with {row['Predicted_Class']} ({row['Predicted_Glyph']}) -> {row['Misclassified_Count']} errors ({row['Paleographic_Reason']})")
    else:
        print("  * None! Zero classification errors on test set.")

    # Save trained models
    joblib.dump(primary_clf, config.MODELS_DIR / "best_model.pkl")
    joblib.dump(feat_pipeline, config.MODELS_DIR / "feature_pipeline.pkl")
    if pca_model:
        joblib.dump(pca_model, config.MODELS_DIR / "pca_model.pkl")
    joblib.dump(class_names, config.MODELS_DIR / "class_names.pkl")

    # -------------------------------------------------------------
    # BENCHMARK GRID: FEATURE SETS x CLASSIFIERS
    # -------------------------------------------------------------
    if run_all_benchmarks:
        print("\n[STAGE 7A] Benchmarking Feature Set x Classifier Matrix...")
        test_clfs = models.get_classifiers(random_state=config.RANDOM_SEED)

        print("  Extracting base feature sets (Gabor, SIFT, HOG, Statistical)...")
        # Extract base representations once
        pipe_gabor = features.FeaturePipeline(feature_type="gabor", random_state=config.RANDOM_SEED)
        X_tr_gabor = pipe_gabor.fit_transform(X_train_prep)
        X_te_gabor = pipe_gabor.transform(X_test_prep)

        pipe_sift = features.FeaturePipeline(feature_type="sift", random_state=config.RANDOM_SEED)
        X_tr_sift = pipe_sift.fit_transform(X_train_prep)
        X_te_sift = pipe_sift.transform(X_test_prep)

        pipe_hog = features.FeaturePipeline(feature_type="hog", random_state=config.RANDOM_SEED)
        X_tr_hog = pipe_hog.fit_transform(X_train_prep)
        X_te_hog = pipe_hog.transform(X_test_prep)

        pipe_stat = features.FeaturePipeline(feature_type="statistical", random_state=config.RANDOM_SEED)
        X_tr_stat = pipe_stat.fit_transform(X_train_prep)
        X_te_stat = pipe_stat.transform(X_test_prep)

        feature_dict = {
            "GABOR": (X_tr_gabor, X_te_gabor),
            "SIFT": (X_tr_sift, X_te_sift),
            "GABOR_SIFT": (np.hstack([X_tr_gabor, X_tr_sift]), np.hstack([X_te_gabor, X_te_sift])),
            "HOG": (X_tr_hog, X_te_hog),
            "STATISTICAL": (X_tr_stat, X_te_stat),
            "ALL": (
                np.hstack([X_tr_gabor, X_tr_sift, X_tr_hog, X_tr_stat]),
                np.hstack([X_te_gabor, X_te_sift, X_te_hog, X_te_stat])
            ),
        }

        benchmark_records = []

        for f_name, (X_tr_f, X_te_f) in feature_dict.items():
            print(f"  Evaluating feature extractor: '{f_name}' (Raw Dim: {X_tr_f.shape[1]})...")

            # Apply PCA
            pca_bench = PCA(n_components=config.PCA_VARIANCE_RATIO, random_state=config.RANDOM_SEED)
            X_tr_f_pca = pca_bench.fit_transform(X_tr_f)
            X_te_f_pca = pca_bench.transform(X_te_f)

            for c_name, clf_obj in test_clfs.items():
                from sklearn.base import clone
                m = clone(clf_obj)
                t_fit = time.time()
                m.fit(X_tr_f_pca, y_train)
                fit_duration = time.time() - t_fit
                preds = m.predict(X_te_f_pca)
                acc = accuracy_score(y_test, preds)
                f1_m = f1_score(y_test, preds, average="macro", zero_division=0)

                benchmark_records.append({
                    "Feature_Set": f_name,
                    "Raw_Dim": X_tr_f.shape[1],
                    "PCA_Dim": X_tr_f_pca.shape[1],
                    "Classifier": c_name,
                    "Accuracy_Pct": round(acc * 100.0, 2),
                    "Macro_F1_Pct": round(f1_m * 100.0, 2),
                    "Train_Time_Sec": round(fit_duration, 3),
                })

        df_bench = pd.DataFrame(benchmark_records)
        df_bench.to_csv(config.RESULTS_DIR / "results_table.csv", index=False)
        print(f"  -> Comprehensive results table saved to: {(config.RESULTS_DIR / 'results_table.csv').resolve()}")

        # Generate Heatmap and Bar Comparison Figures
        evaluate.plot_results_benchmark(df_bench)

    # -------------------------------------------------------------
    # COMPONENT ABLATION STUDY
    # -------------------------------------------------------------
    if run_ablation:
        print("\n[STAGE 7B] Running Component Ablation Study...")
        df_ablation = evaluate.run_ablation_study(
            X_train_raw, X_test_raw, y_train, y_test, class_names,
            X_tr_full=X_train_feat, X_te_full=X_test_feat,
            random_state=config.RANDOM_SEED
        )

    total_time = time.time() - t_start
    print("\n" + "="*70)
    print(f" PIPELINE COMPLETE in {total_time:.1f} seconds!")
    print(f" All results, CSV tables, and slide-ready figures saved in: {config.RESULTS_DIR.resolve()}")
    print("="*70 + "\n")

    return {
        "metrics": metrics,
        "cv_results": cv_results,
        "class_names": class_names,
        "raw_feat_dim": raw_feat_dim,
        "pca_feat_dim": pca_feat_dim if use_pca else raw_feat_dim,
    }


def main():
    parser = argparse.ArgumentParser(description="Ancient Tamil Palm-Leaf Character Recognition Pipeline")
    parser.add_argument(
        "--paper",
        type=int,
        choices=[1, 2],
        default=None,
        help="Quick configuration for reference papers: 1 = Ramya & Parvathavarthini (FFBNN + Binary), 2 = Poornima Devi & Sornam (MABPN + GLCM)"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="gabor_sift",
        choices=["gabor", "sift", "gabor_sift", "hog", "statistical", "glcm", "binary_coded", "all"],
        help="Feature extraction technique"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="SVM-Poly",
        choices=[
            "SVM-Poly", "SVM-RBF", "SVM-Linear", "k-NN", "Random Forest",
            "MLP (Shallow NN)", "MLP", "MLP-2Layer",
            "FFBNN", "FFBNN (Paper 1)", "MABPN", "MABPN (Paper 2)", "all"
        ],
        help="Classifier model (Default: SVM-Poly; supports FFBNN, MABPN, MLP, Random Forest, k-NN)"
    )
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA dimensionality reduction")
    parser.add_argument("--no-ablation", action="store_true", help="Skip ablation study")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip full feature x classifier benchmark grid")
    parser.add_argument("--tune", action="store_true", help="Enable GridSearchCV hyperparameter tuning")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of cross-validation folds")
    args = parser.parse_args()

    # Handle --paper shortcuts
    if args.paper == 1:
        feature_set = "binary_coded"
        chosen_clf = "FFBNN (Paper 1)"
        print("\n>>> CONFIGURED FOR PAPER 1: Ramya & Parvathavarthini (2014) - FFBNN + Binary Coded Features <<<\n")
    elif args.paper == 2:
        feature_set = "glcm"
        chosen_clf = "MABPN (Paper 2)"
        print("\n>>> CONFIGURED FOR PAPER 2: Poornima Devi & Sornam (2020) - MABPN + GLCM Texture Features <<<\n")
    else:
        feature_set = args.feature_set
        if args.classifier == "MLP":
            chosen_clf = "MLP (Shallow NN)"
        elif args.classifier == "FFBNN":
            chosen_clf = "FFBNN (Paper 1)"
        elif args.classifier == "MABPN":
            chosen_clf = "MABPN (Paper 2)"
        else:
            chosen_clf = args.classifier

    run_pipeline(
        feature_set=feature_set,
        selected_classifier=chosen_clf,
        run_all_benchmarks=not args.no_benchmark,
        run_ablation=not args.no_ablation,
        tune_params=args.tune,
        cv_folds=args.cv_folds,
        use_pca=not args.no_pca
    )


if __name__ == "__main__":
    main()
