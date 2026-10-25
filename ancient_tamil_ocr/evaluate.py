"""
Evaluation and Metrics module for Ancient Tamil Palm-Leaf Character Recognition.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Implements ONLY the evaluation metrics reported in the reference paper (Section 4.1 & Tables 5, 8, 9, 10, 11):
1. Classification Accuracy: (Correctly Classified / Total Samples) * 100
2. Classification Error: (Misclassified / Total Samples) * 100
3. Average Precision
4. Average Recall
5. Average F1-Measure
6. Confusion Matrix (Figure 15 & 18)

Maintains clear scientific separation between:
- Benchmark results published in the reference paper (Table 5)
- Results produced by this implementation on the local dataset.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import config


# Reference values published in Poornima Devi & Sornam (2020), Table 5 (Page 82:16)
PAPER_REPORTED_RESULTS = {
    "Palm leaf manuscript": {
        "MABPN-BCV": {
            "Epochs": 130,
            "Error": 0.0003,
            "Average Precision": 0.96,
            "Average Recall": 0.94,
            "Average F1-Measure": 0.94,
            "Classification Accuracy": "94.1%",
        },
        "MABPN-GLCM": {
            "Epochs": 104,
            "Error": 0.0003,
            "Average Precision": 0.98,
            "Average Recall": 0.97,
            "Average F1-Measure": 0.98,
            "Classification Accuracy": "96.23%",
        },
    }
}


def compute_paper_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model: Optional[Any] = None,
    raw_outputs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes the exact set of metrics required by the reference paper:
      - Classification Accuracy
      - Classification Error
      - Average Precision (macro)
      - Average Recall (macro)
      - Average F1-Measure (macro)
      - Total Correctly Classified & Misclassified counts (Table 9)
      - Confusion Matrix (Figure 15)
      
    Phase 8 Diagnostic Metrics:
      - Unique predicted classes count & distribution
      - True classes count & distribution
      - Mean and standard deviation of continuous output activations
      - Number of finite and non-finite weights
      - Mode collapse detector
    """
    # Normalize all labels to Figure 12 canonical names to prevent string mismatch
    y_true_canonical = np.array([config.normalize_label(y) for y in y_true])
    y_pred_canonical = np.array([config.normalize_label(y) for y in y_pred])
    seen_classes = set()
    class_names_canonical = []
    for c in class_names:
        c_norm = config.normalize_label(c)
        if c_norm not in seen_classes:
            seen_classes.add(c_norm)
            class_names_canonical.append(c_norm)

    total_samples = len(y_true_canonical)
    correct_count = int(np.sum(y_true_canonical == y_pred_canonical))
    misclassified_count = total_samples - correct_count

    acc = float(accuracy_score(y_true_canonical, y_pred_canonical))
    error_rate = 1.0 - acc
    prec = float(precision_score(y_true_canonical, y_pred_canonical, average="macro", zero_division=0))
    rec = float(recall_score(y_true_canonical, y_pred_canonical, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_canonical, y_pred_canonical, average="macro", zero_division=0))

    cm = confusion_matrix(y_true_canonical, y_pred_canonical, labels=class_names_canonical)

    # Diagnostic checks (Phase 8)
    u_pred, c_pred = np.unique(y_pred_canonical, return_counts=True)
    pred_dist = dict(zip(u_pred.tolist(), [int(c) for c in c_pred]))

    u_true, c_true = np.unique(y_true_canonical, return_counts=True)
    true_dist = dict(zip(u_true.tolist(), [int(c) for c in c_true]))

    num_unique_preds = len(u_pred)
    is_collapsed = bool(num_unique_preds <= 1 and len(class_names) > 1 and total_samples > 1)

    finite_w = 0
    non_finite_w = 0
    if model is not None and hasattr(model, "W") and hasattr(model, "V"):
        finite_w = int(np.sum(np.isfinite(model.W)) + np.sum(np.isfinite(model.V)) +
                       np.sum(np.isfinite(model.bh)) + np.sum(np.isfinite(model.bo)))
        non_finite_w = int(np.sum(~np.isfinite(model.W)) + np.sum(~np.isfinite(model.V)) +
                           np.sum(~np.isfinite(model.bh)) + np.sum(~np.isfinite(model.bo)))

    raw_mean = float(np.mean(raw_outputs)) if raw_outputs is not None else None
    raw_std = float(np.std(raw_outputs)) if raw_outputs is not None else None

    return {
        "total_samples": total_samples,
        "correctly_classified": correct_count,
        "misclassified": misclassified_count,
        "classification_accuracy": acc * 100.0,
        "classification_error": error_rate * 100.0,
        "average_precision": prec,
        "average_recall": rec,
        "average_f1_measure": f1,
        "confusion_matrix": cm,
        "num_unique_predictions": num_unique_preds,
        "predicted_distribution": pred_dist,
        "true_distribution": true_dist,
        "is_collapsed": is_collapsed,
        "finite_weights": finite_w,
        "non_finite_weights": non_finite_w,
        "output_activations_mean": raw_mean,
        "output_activations_std": raw_std,
    }


def plot_paper_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str = "MABPN-GLCM",
    output_path: Optional[Path] = None
):
    """
    Plots the confusion matrix matching the style of Figure 15 in the paper.
    Diagonal shows correctly classified counts; off-diagonal shows misclassifications.
    """
    if output_path is None:
        output_path = config.RESULTS_DIR / f"confusion_matrix_{model_name.lower().replace('-', '_')}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(len(class_names)),
        yticklabels=range(len(class_names)),
        cbar=True
    )
    plt.title(
        f"Confusion Matrix for {model_name} (Tamil Palm Leaf Manuscript)\n"
        f"Reference: Poornima Devi & Sornam (2020), Figure 15",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    plt.xlabel("Predicted Character Class", fontsize=11, fontweight="semibold")
    plt.ylabel("Target Character Class", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {output_path.resolve()}")


def format_evaluation_summary(
    model_name: str,
    metrics: Dict[str, Any],
    epochs_trained: int,
    final_error: float
) -> str:
    """
    Formats the evaluation results matching the format of Table 5 in the paper,
    along with comprehensive Phase 8 diagnostic reporting.
    """
    collapse_status = "!! ALERT: COLLAPSED TO SINGLE CLASS !!" if metrics.get("is_collapsed") else "HEALTHY (Multi-class variation confirmed)"
    raw_mean_str = f"{metrics['output_activations_mean']:.4f}" if metrics.get("output_activations_mean") is not None else "N/A"
    raw_std_str = f"{metrics['output_activations_std']:.4f}" if metrics.get("output_activations_std") is not None else "N/A"

    summary = f"""
========================================================================================
 EVALUATION REPORT: {model_name}
 Reference Paper: Poornima Devi & Sornam (2020), ACM TALLIP, Article 82 (Table 5 & 9)
========================================================================================
 Implementation Results (Executed on current dataset):
   - Total Samples Evaluated:    {metrics['total_samples']}
   - Correctly Classified:      {metrics['correctly_classified']}
   - Misclassified:             {metrics['misclassified']}
   - Classification Accuracy:   {metrics['classification_accuracy']:.2f}%
   - Classification Error:      {metrics['classification_error']:.2f}%
   - Average Precision:         {metrics['average_precision']:.4f}
   - Average Recall:            {metrics['average_recall']:.4f}
   - Average F1-Measure:        {metrics['average_f1_measure']:.4f}
   - Epochs Run:                {epochs_trained}
   - Final Training Error:      {final_error:.6f}

 Phase 8 Diagnostics (Mode Collapse & Numerical Health Verification):
   - Number of Unique Predictions: {metrics['num_unique_predictions']}
   - Mode Collapse Check:          {collapse_status}
   - Predicted Class Breakdown:    {metrics['predicted_distribution']}
   - True Class Breakdown:         {metrics['true_distribution']}
   - Finite Weights:               {metrics['finite_weights']} (Non-finite: {metrics['non_finite_weights']})
   - Output Activations Mean:      {raw_mean_str} (Std: {raw_std_str})

 Benchmark Reported by Reference Paper (Table 5 - Primary Manuscript Dataset):
   - Classification Accuracy:   {PAPER_REPORTED_RESULTS['Palm leaf manuscript'].get(model_name, {}).get('Classification Accuracy', 'N/A')}
   - Average Precision:         {PAPER_REPORTED_RESULTS['Palm leaf manuscript'].get(model_name, {}).get('Average Precision', 'N/A')}
   - Average Recall:            {PAPER_REPORTED_RESULTS['Palm leaf manuscript'].get(model_name, {}).get('Average Recall', 'N/A')}
   - Average F1-Measure:        {PAPER_REPORTED_RESULTS['Palm leaf manuscript'].get(model_name, {}).get('Average F1-Measure', 'N/A')}
   - Reported Minimized Error:  0.0003
========================================================================================
"""
    return summary
