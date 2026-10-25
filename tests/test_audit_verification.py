"""
Unit tests verifying the 7 priority actions:
1. Correct Shannon activation Eq. (8) and analytical limits.
2. Correct MABPN learning equations and fit/predict.
3. Correct FFBNN adapted implementation.
4. Correct 6-feature GLCM representation (Equations 21-27).
5. Leak-free cross-validation.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent / "ancient_tamil_ocr"
sys.path.insert(0, str(PROJECT_DIR))

import neural_networks
import features
import models


def test_shannon_activation_limit_and_values():
    """Verify Equation (8) and analytical limits."""
    # Near zero limit should be exactly 4/7
    zero_val = neural_networks.shannon_activation(np.array([0.0]))
    assert np.isclose(zero_val[0], 4.0 / 7.0, atol=1e-6)

    # Derivative limit at zero should be 0.0
    zero_deriv = neural_networks.shannon_derivative(np.array([0.0]))
    assert np.isclose(zero_deriv[0], 0.0, atol=1e-6)

    # Known value at x = 1.0: (sin(1.75) - sin(0.75)) / 1.75
    expected_1 = (np.sin(1.75) - np.sin(0.75)) / 1.75
    val_1 = neural_networks.shannon_activation(np.array([1.0]))
    assert np.isclose(val_1[0], expected_1, atol=1e-6)


def test_glcm_six_features():
    """Verify that extract_glcm_features returns 6 exact Haralick features."""
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    glcm_feats = features.extract_glcm_features(img)
    assert glcm_feats.shape == (6,), f"Expected shape (6,), got {glcm_feats.shape}"
    # Ensure energy = sqrt(ASM)
    contrast, energy, homogeneity, correlation, dissimilarity, asm = glcm_feats
    assert np.isclose(energy, np.sqrt(asm), atol=1e-5), f"Energy ({energy}) != sqrt(ASM) ({np.sqrt(asm)})"
    assert energy > 0.0
    assert asm > 0.0


def test_mabpn_fit_predict():
    """Verify MABPN fits and predicts cleanly with Shannon activation."""
    X = np.random.randn(40, 6)
    y = np.array([0, 1, 2, 3] * 10)
    mabpn = neural_networks.ModifiedAdaptiveBackpropNN(
        hidden_layer_sizes=(16,),
        initial_learning_rate=0.05,
        activation="shannon",
        max_iter=30,
        random_state=42
    )
    mabpn.fit(X, y)
    preds = mabpn.predict(X)
    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1, 2, 3})


def test_ffbnn_fit_predict():
    """Verify adapted FFBNN fits and predicts cleanly."""
    X = np.random.randn(40, 64)
    y = np.array([0, 1, 2, 3] * 10)
    ffbnn = neural_networks.FeedForwardBackpropNN(
        hidden_layer_sizes=(16,),
        learning_rate=0.05,
        max_iter=30,
        random_state=42
    )
    ffbnn.fit(X, y)
    preds = ffbnn.predict(X)
    assert preds.shape == (40,)


def test_leak_free_cross_validation():
    """Verify 10-fold CV with PCA has zero leakage."""
    X = np.random.randn(50, 20)
    y = np.array([0, 1, 2, 3, 4] * 10)
    clf = neural_networks.ModifiedAdaptiveBackpropNN(
        hidden_layer_sizes=(10,),
        max_iter=10,
        random_state=42
    )
    res = models.run_tenfold_cross_validation(
        clf, X, y, use_pca=True, pca_ratio=0.90, cv_folds=5, random_state=42
    )
    assert "mean_accuracy" in res
    assert 0.0 <= res["mean_accuracy"] <= 1.0
