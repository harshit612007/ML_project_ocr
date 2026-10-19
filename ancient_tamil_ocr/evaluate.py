"""
Evaluation module for Ancient Tamil Palm-Leaf Character Recognition.
Domain: Ancient Tamil Palm-Leaf Manuscripts (Panaiyolai Suvadigal).
Provides comprehensive metrics, publication-ready figures, error analysis,
and ablation studies.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure font for Tamil glyph rendering without warnings
try:
    import matplotlib.font_manager as fm
    font_names = set(f.name for f in fm.fontManager.ttflist)
    for cand in ["Nirmala UI", "Latha", "Vijaya", "DejaVu Sans"]:
        if cand in font_names:
            matplotlib.rcParams["font.sans-serif"] = [cand] + matplotlib.rcParams["font.sans-serif"]
            matplotlib.rcParams["font.family"] = "sans-serif"
            break
except Exception:
    pass

matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA

import config
import preprocess
import features
import models


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Calculate per-class accuracy
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / np.maximum(1, np.sum(cm, axis=1))

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class_accuracy": per_class_acc,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix - Ancient Tamil Palm-Leaf Character Recognition"
):
    """
    Plots a publication-quality confusion matrix heatmap showing raw counts
    and normalized percentage values.
    """
    if output_path is None:
        output_path = config.RESULTS_DIR / "confusion_matrix.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute row-normalized percentages for color intensity
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float32) / np.maximum(1, np.sum(cm, axis=1, keepdims=True))

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # Format annotations with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)" if cm[i, j] > 0 else "0"

    # Display glyphs if available
    display_labels = [
        f"{c}\n({config.TAMIL_GLYPH_MAP.get(c, '')})" if config.TAMIL_GLYPH_MAP.get(c) else c
        for c in class_names
    ]

    ax = sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Normalized Recall"},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="semibold", labelpad=10)
    plt.ylabel("True Class (Ground Truth)", fontsize=12, fontweight="semibold", labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {output_path.resolve()}")


def plot_per_class_accuracy(
    per_class_acc: np.ndarray,
    class_names: List[str],
    output_path_png: Optional[Path] = None,
    output_path_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generates horizontal bar chart and CSV table for per-class recognition accuracy.
    """
    if output_path_png is None:
        output_path_png = config.RESULTS_DIR / "per_class_accuracy.png"
    if output_path_csv is None:
        output_path_csv = config.RESULTS_DIR / "per_class_accuracy.csv"

    output_path_png = Path(output_path_png)
    output_path_csv = Path(output_path_csv)

    display_names = [
        f"{c} [{config.TAMIL_GLYPH_MAP.get(c, '')}]" for c in class_names
    ]

    df = pd.DataFrame({
        "Class": class_names,
        "Tamil_Glyph": [config.TAMIL_GLYPH_MAP.get(c, "") for c in class_names],
        "Accuracy_Pct": per_class_acc * 100.0
    }).sort_values(by="Accuracy_Pct", ascending=True)

    df.to_csv(output_path_csv, index=False)
    print(f"[INFO] Per-class accuracy table saved to: {output_path_csv.resolve()}")

    plt.figure(figsize=(9, 6))
    bars = plt.barh(df["Class"], df["Accuracy_Pct"], color="#2980b9", edgecolor="#1a5276", height=0.65)

    # Highlight top and challenging classes with gradient
    for bar in bars:
        width = bar.get_width()
        color = "#27ae60" if width >= 90.0 else ("#f39c12" if width >= 80.0 else "#e74c3c")
        bar.set_color(color)
        plt.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50"
        )

    plt.xlim(0, 110)
    plt.xlabel("Recognition Accuracy (%)", fontsize=12, fontweight="semibold")
    plt.ylabel("Character Class", fontsize=12, fontweight="semibold")
    plt.title("Per-Class Accuracy on Test Set (Ancient Palm-Leaf Characters)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Per-class accuracy plot saved to: {output_path_png.resolve()}")

    return df


def identify_most_confused_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_n: int = 5,
    output_path_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Identifies the most-confused character pairs (Ground Truth vs Prediction)
    and provides paleographic/structural error explanation.
    """
    if output_path_csv is None:
        output_path_csv = config.RESULTS_DIR / "confused_pairs.csv"
    output_path_csv = Path(output_path_csv)

    confused_list = []
    n_classes = len(class_names)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                true_cls = class_names[i]
                pred_cls = class_names[j]
                true_glyph = config.TAMIL_GLYPH_MAP.get(true_cls, "")
                pred_glyph = config.TAMIL_GLYPH_MAP.get(pred_cls, "")
                count = int(cm[i, j])
                total_true = int(np.sum(cm[i, :]))
                error_rate = float(count / max(1, total_true))

                # Paleographic explanation based on Tamil character morphology on palm leaf
                reason = "Similar curved stroke geometry and palm-leaf fiber degradation"
                if "07_pa" in true_cls and "08_ma" in pred_cls:
                    reason = "Open U-shape basin vs closed loop ambiguity under eroded stylus ink"
                elif "08_ma" in true_cls and "07_pa" in pred_cls:
                    reason = "Faded internal loop causing 'ம' to resemble open 'ப'"
                elif "01_a" in true_cls and "02_aa" in pred_cls:
                    reason = "Shared initial loop; rightward sweeping tail erosion or fiber confusion"
                elif "02_aa" in true_cls and "01_a" in pred_cls:
                    reason = "Downward sweeping tail of 'ஆ' faded along leaf fiber grain"
                elif "04_ka" in true_cls and "05_cha" in pred_cls:
                    reason = "Central stem with loop vs curved bottom stroke ambiguity"
                elif "06_ta" in true_cls and "04_ka" in pred_cls:
                    reason = "Top horizontal bar and descending curved loop similarity"
                elif "09_ra" in true_cls and "07_pa" in pred_cls:
                    reason = "Narrow vertical channel contour resembling open 'ப'"
                elif "10_la" in true_cls and "03_i" in pred_cls:
                    reason = "Double curved loop spiral ambiguity under leaf vein noise"

                confused_list.append({
                    "True_Class": true_cls,
                    "True_Glyph": true_glyph,
                    "Predicted_Class": pred_cls,
                    "Predicted_Glyph": pred_glyph,
                    "Misclassified_Count": count,
                    "Error_Rate_Pct": round(error_rate * 100.0, 2),
                    "Paleographic_Reason": reason,
                })

    df = pd.DataFrame(confused_list)
    if len(df) > 0:
        df = df.sort_values(by="Misclassified_Count", ascending=False).head(top_n)
    else:
        df = pd.DataFrame(columns=[
            "True_Class", "True_Glyph", "Predicted_Class", "Predicted_Glyph",
            "Misclassified_Count", "Error_Rate_Pct", "Paleographic_Reason"
        ])

    df.to_csv(output_path_csv, index=False)
    print(f"[INFO] Most-confused character pairs saved to: {output_path_csv.resolve()}")
    return df


def plot_pca_variance(
    pca_model: PCA,
    output_path: Optional[Path] = None,
    threshold: float = config.PCA_VARIANCE_RATIO
):
    """
    Plots cumulative explained variance vs number of principal components.
    Annotates the 95% variance threshold and the retained component count.
    """
    if output_path is None:
        output_path = config.RESULTS_DIR / "pca_variance_explained.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    n_components = len(cum_variance)

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        np.arange(1, n_components + 1),
        cum_variance * 100.0,
        color="#2980b9",
        linewidth=2.2,
        label="Cumulative Explained Variance"
    )

    # 95% threshold horizontal line
    plt.axhline(
        y=threshold * 100.0,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.8,
        label=f"{threshold*100:.0f}% Target Variance Threshold"
    )

    # Mark the cutoff point
    plt.axvline(
        x=n_components,
        color="#27ae60",
        linestyle=":",
        linewidth=1.8,
        label=f"Retained Dimensions: {n_components}"
    )
    plt.plot(n_components, cum_variance[-1] * 100.0, marker="o", markersize=8, color="#27ae60")

    plt.title(
        f"PCA Dimensionality Reduction: Cumulative Explained Variance\n"
        f"(Retaining {threshold*100:.0f}% Variance with {n_components} Principal Components)",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    plt.xlabel("Number of Principal Components", fontsize=11, fontweight="semibold")
    plt.ylabel("Cumulative Variance Explained (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] PCA cumulative variance plot saved to: {output_path.resolve()}")


def plot_results_benchmark(
    results_df: pd.DataFrame,
    output_path_heatmap: Optional[Path] = None,
    output_path_bar: Optional[Path] = None
):
    """
    Generates comparison heatmap and bar charts for Feature Set x Classifier grid.
    """
    if output_path_heatmap is None:
        output_path_heatmap = config.RESULTS_DIR / "classifier_feature_heatmap.png"
    if output_path_bar is None:
        output_path_bar = config.RESULTS_DIR / "classifier_comparison.png"

    output_path_heatmap = Path(output_path_heatmap)
    output_path_bar = Path(output_path_bar)

    # Pivot table: Feature_Set x Classifier (Values = Accuracy_Pct)
    pivot_df = results_df.pivot(index="Feature_Set", columns="Classifier", values="Accuracy_Pct")

    # 1. Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Test Accuracy (%)"},
        linewidths=0.8,
        linecolor="#ffffff"
    )
    plt.title("Benchmarking Matrix: Feature Set $\\times$ Classifier Accuracy (%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classifier", fontsize=11, fontweight="semibold", labelpad=8)
    plt.ylabel("Feature Extraction Method", fontsize=11, fontweight="semibold", labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path_heatmap, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Benchmark heatmap saved to: {output_path_heatmap.resolve()}")

    # 2. Grouped Bar Chart
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", figsize=(12, 6), colormap="Set2", width=0.8, edgecolor="#333333")
    plt.title("Classifier Performance Across Feature Sets (Ancient Devanagari)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Feature Extraction Set", fontsize=11, fontweight="semibold")
    plt.ylabel("Recognition Accuracy (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path_bar, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Benchmark bar chart saved to: {output_path_bar.resolve()}")


def run_ablation_study(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    X_tr_full: Optional[np.ndarray] = None,
    X_te_full: Optional[np.ndarray] = None,
    output_path_csv: Optional[Path] = None,
    output_path_png: Optional[Path] = None,
    random_state: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Executes an ablation study to quantify the impact of:
      1. Preprocessing (Sauvola + CLAHE + Morphological vs Raw Grayscale Resized)
      2. PCA Dimensionality Reduction (95% Variance vs Full Raw Feature Vector)
      3. Raw Pixel Baseline
    """
    if output_path_csv is None:
        output_path_csv = config.RESULTS_DIR / "ablation_study.csv"
    if output_path_png is None:
        output_path_png = config.RESULTS_DIR / "ablation_study.png"

    output_path_csv = Path(output_path_csv)
    output_path_png = Path(output_path_png)

    print("\n" + "="*70)
    print(" [ABLATION STUDY] Running systematic component impact analysis...")
    print("="*70)

    # Base winning classifier from Narang et al. (2020): SVM-Poly
    clf_base = models.get_classifiers(random_state=random_state)["SVM-Poly"]
    ablation_records = []
    # Prepare preprocessed images if needed
    X_train_prep = preprocess.preprocess_dataset(X_train_raw)
    X_test_prep = preprocess.preprocess_dataset(X_test_raw)

    # Raw grayscale resized images (WITHOUT restoration/binarization)
    X_train_no_prep = np.array([preprocess.to_grayscale(cv2.resize(img, config.IMG_SIZE)) for img in X_train_raw])
    X_test_no_prep = np.array([preprocess.to_grayscale(cv2.resize(img, config.IMG_SIZE)) for img in X_test_raw])

    # Preprocessed features
    if X_tr_full is None or X_te_full is None:
        pipe_full = features.FeaturePipeline(feature_type="gabor_sift", random_state=random_state)
        X_tr_full = pipe_full.fit_transform(X_train_prep)
        X_te_full = pipe_full.transform(X_test_prep)

    # -------------------------------------------------------------
    # Condition 1: Full Proposed Pipeline (Preprocessed + SIFT+Gabor + PCA)
    # -------------------------------------------------------------
    t0 = time.time()
    pca_full = PCA(n_components=config.PCA_VARIANCE_RATIO, random_state=random_state)
    X_tr_pca = pca_full.fit_transform(X_tr_full)
    X_te_pca = pca_full.transform(X_te_full)

    from sklearn.base import clone
    clf_1 = clone(clf_base)
    clf_1.fit(X_tr_pca, y_train)
    t_train_1 = time.time() - t0
    preds_1 = clf_1.predict(X_te_pca)
    acc_1 = accuracy_score(y_test, preds_1)
    f1_1 = f1_score(y_test, preds_1, average="macro", zero_division=0)

    ablation_records.append({
        "Condition": "1. Full Proposed (Prep + Gabor/SIFT + PCA)",
        "Preprocessing": "Yes (CLAHE + Sauvola)",
        "Feature_Extractor": "Gabor + Dense SIFT",
        "PCA_Applied": "Yes (95% Var)",
        "Feature_Dim": X_tr_pca.shape[1],
        "Accuracy_Pct": round(acc_1 * 100.0, 2),
        "Macro_F1_Pct": round(f1_1 * 100.0, 2),
        "Train_Time_Sec": round(t_train_1, 3),
    })

    # -------------------------------------------------------------
    # Condition 2: Without PCA (Preprocessed + SIFT+Gabor + Full Vector)
    # -------------------------------------------------------------
    t0 = time.time()
    clf_2 = clone(clf_base)
    clf_2.fit(X_tr_full, y_train)
    t_train_2 = time.time() - t0
    preds_2 = clf_2.predict(X_te_full)
    acc_2 = accuracy_score(y_test, preds_2)
    f1_2 = f1_score(y_test, preds_2, average="macro", zero_division=0)

    ablation_records.append({
        "Condition": "2. Without PCA (Prep + Gabor/SIFT, Full Dim)",
        "Preprocessing": "Yes (CLAHE + Sauvola)",
        "Feature_Extractor": "Gabor + Dense SIFT",
        "PCA_Applied": "No (Full Vector)",
        "Feature_Dim": X_tr_full.shape[1],
        "Accuracy_Pct": round(acc_2 * 100.0, 2),
        "Macro_F1_Pct": round(f1_2 * 100.0, 2),
        "Train_Time_Sec": round(t_train_2, 3),
    })

    # -------------------------------------------------------------
    # Condition 3: Without Preprocessing (Raw Grayscale + SIFT+Gabor + PCA)
    # -------------------------------------------------------------
    t0 = time.time()
    pipe_no_prep = features.FeaturePipeline(feature_type="gabor_sift", random_state=random_state)
    X_tr_noprep_feat = pipe_no_prep.fit_transform(X_train_no_prep)
    X_te_noprep_feat = pipe_no_prep.transform(X_test_no_prep)

    pca_noprep = PCA(n_components=config.PCA_VARIANCE_RATIO, random_state=random_state)
    X_tr_noprep_pca = pca_noprep.fit_transform(X_tr_noprep_feat)
    X_te_noprep_pca = pca_noprep.transform(X_te_noprep_feat)

    clf_3 = clone(clf_base)
    clf_3.fit(X_tr_noprep_pca, y_train)
    t_train_3 = time.time() - t0
    preds_3 = clf_3.predict(X_te_noprep_pca)
    acc_3 = accuracy_score(y_test, preds_3)
    f1_3 = f1_score(y_test, preds_3, average="macro", zero_division=0)

    ablation_records.append({
        "Condition": "3. Without Preprocessing (Raw Gray + PCA)",
        "Preprocessing": "No (Raw Grayscale)",
        "Feature_Extractor": "Gabor + Dense SIFT",
        "PCA_Applied": "Yes (95% Var)",
        "Feature_Dim": X_tr_noprep_pca.shape[1],
        "Accuracy_Pct": round(acc_3 * 100.0, 2),
        "Macro_F1_Pct": round(f1_3 * 100.0, 2),
        "Train_Time_Sec": round(t_train_3, 3),
    })

    # -------------------------------------------------------------
    # Condition 4: Baseline (Raw Pixels Flattened + PCA + SVM-Poly)
    # -------------------------------------------------------------
    t0 = time.time()
    X_tr_pix = X_train_prep.reshape(len(X_train_prep), -1).astype(np.float32) / 255.0
    X_te_pix = X_test_prep.reshape(len(X_test_prep), -1).astype(np.float32) / 255.0

    pca_pix = PCA(n_components=config.PCA_VARIANCE_RATIO, random_state=random_state)
    X_tr_pix_pca = pca_pix.fit_transform(X_tr_pix)
    X_te_pix_pca = pca_pix.transform(X_te_pix)

    clf_4 = clone(clf_base)
    clf_4.fit(X_tr_pix_pca, y_train)
    t_train_4 = time.time() - t0
    preds_4 = clf_4.predict(X_te_pix_pca)
    acc_4 = accuracy_score(y_test, preds_4)
    f1_4 = f1_score(y_test, preds_4, average="macro", zero_division=0)

    ablation_records.append({
        "Condition": "4. Baseline (Raw Pixels + PCA)",
        "Preprocessing": "Yes (CLAHE + Sauvola)",
        "Feature_Extractor": "Pixel Intensity (Flattened)",
        "PCA_Applied": "Yes (95% Var)",
        "Feature_Dim": X_tr_pix_pca.shape[1],
        "Accuracy_Pct": round(acc_4 * 100.0, 2),
        "Macro_F1_Pct": round(f1_4 * 100.0, 2),
        "Train_Time_Sec": round(t_train_4, 3),
    })

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(output_path_csv, index=False)
    print(f"[INFO] Ablation study results table saved to: {output_path_csv.resolve()}")

    # Plot Ablation Bar Chart
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x_indices = np.arange(len(df_ablation))
    bar_width = 0.55

    colors = ["#27ae60", "#2980b9", "#f39c12", "#e74c3c"]
    bars = ax1.bar(
        x_indices,
        df_ablation["Accuracy_Pct"],
        width=bar_width,
        color=colors,
        edgecolor="#2c3e50",
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50"
        )

    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Recognition Accuracy (%)", fontsize=11, fontweight="semibold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(
        [c.replace(" ", "\n") for c in df_ablation["Condition"]],
        fontsize=9,
        fontweight="semibold"
    )
    ax1.set_title(
        "Ablation Analysis: Contribution of Preprocessing, SIFT+Gabor, and PCA",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Ablation plot saved to: {output_path_png.resolve()}")

    return df_ablation


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
