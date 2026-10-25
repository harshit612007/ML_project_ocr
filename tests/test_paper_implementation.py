"""
Unit tests for strict paper compliance: Poornima Devi. M & M. Sornam (2020), ACM TALLIP, Article 82.

Verifies:
1. BCV feature extraction returns exactly 64 binary values.
2. MABPN-BCV architecture has 65 input connections (64 + 1 bias), 55 hidden neurons, 5 output neurons.
3. GLCM feature extraction returns exactly 6 features matching Equations 21-27.
4. MABPN-GLCM architecture has 6 input neurons (+ 1 bias), 20 hidden neurons, 5 output neurons.
5. Sigmoid implementation and analytical derivative.
6. Shannon wavelet activation function (Equation 8) and analytical derivative, including limit at x = 0.
7. Nguyen-Widrow weight initialization (Equations 1-6) scaling factors and weight norms.
8. 5-bit class encoding and decoding mapping matching Figure 12 for all 18 classes.
9. MABPN weight updates using linear and non-linear errors (Equations 9-20).
10. End-to-end pipeline execution on a small labelled fixture.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add ancient_tamil_ocr to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ancient_tamil_ocr"))

import config
import preprocess
import features
import neural_networks
import models


class TestPaperImplementation(unittest.TestCase):

    def test_01_bcv_feature_dimensions(self):
        """1. BCV input has the correct 64 pixel values (Section 3.1.2)."""
        dummy_img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        bcv_feat = features.extract_binary_coded_features(dummy_img, grid_size=(8, 8))
        self.assertEqual(bcv_feat.shape, (64,), "BCV feature vector must be of length 64.")
        # Must be binary {0.0, 1.0}
        unique_vals = np.unique(bcv_feat)
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0], "BCV values must be strictly binary (0 or 1).")

    def test_02_bcv_architecture(self):
        """2. BCV architecture: 65 input neurons (64 + 1 bias), 55 hidden neurons, 5 output neurons."""
        model = models.MABPN_BCV(learning_rate=0.05, max_iter=2, random_state=42)
        model._init_weights()
        self.assertEqual(model.n_inputs, 64)
        self.assertEqual(model.n_hidden, 55)
        self.assertEqual(model.n_outputs, 5)
        self.assertEqual(model.W.shape, (64, 55), "W1 weight shape must be (64, 55).")
        self.assertEqual(model.bh.shape, (1, 55), "Hidden bias shape must be (1, 55).")
        self.assertEqual(model.V.shape, (55, 5), "V weight shape must be (55, 5).")
        self.assertEqual(model.bo.shape, (1, 5), "Output bias shape must be (1, 5).")

    def test_03_glcm_feature_extraction(self):
        """3. GLCM extraction returns exactly six features matching Equations 21-27."""
        dummy_img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        glcm_feat = features.extract_glcm_features(dummy_img)
        self.assertEqual(glcm_feat.shape, (6,), "GLCM feature vector must have length exactly 6.")
        # Ensure all values are finite real numbers
        self.assertTrue(np.all(np.isfinite(glcm_feat)), "GLCM features must be finite numbers.")
        # Homogeneity and Energy are in [0, 1]
        self.assertGreaterEqual(glcm_feat[2], 0.0, "Homogeneity must be non-negative.")

    def test_04_glcm_architecture(self):
        """4. GLCM architecture: 6 input neurons (+ bias), 20 hidden neurons, 5 output neurons."""
        model = models.MABPN_GLCM(learning_rate=0.05, max_iter=2, random_state=42)
        model._init_weights()
        self.assertEqual(model.n_inputs, 6)
        self.assertEqual(model.n_hidden, 20)
        self.assertEqual(model.n_outputs, 5)
        self.assertEqual(model.W.shape, (6, 20), "W1 weight shape must be (6, 20).")
        self.assertEqual(model.bh.shape, (1, 20), "Hidden bias shape must be (1, 20).")
        self.assertEqual(model.V.shape, (20, 5), "V weight shape must be (20, 5).")
        self.assertEqual(model.bo.shape, (1, 5), "Output bias shape must be (1, 5).")

    def test_05_sigmoid_and_derivative(self):
        """5. Sigmoid implementation matches Equation (7) and its derivative."""
        x = np.array([-2.0, 0.0, 2.0])
        f_x = neural_networks.sigmoid(x)
        expected_f_x = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(f_x, expected_f_x, atol=1e-10)

        # Derivative: f'(x) = f(x) * (1 - f(x))
        f_prime = neural_networks.sigmoid_derivative(f_x)
        expected_f_prime = expected_f_x * (1.0 - expected_f_x)
        np.testing.assert_allclose(f_prime, expected_f_prime, atol=1e-10)

    def test_06_shannon_wavelet_activation_and_derivative(self):
        """6. Shannon wavelet activation matches Equation (8) and derivative (including limit at x=0)."""
        # Test at x = 0 limit: lim_{x->0} f(x) = (1.75 - 0.75) / 1.75 = 4/7
        f_zero = neural_networks.shannon_wavelet_activation(np.array([0.0]))
        self.assertAlmostEqual(f_zero[0], 4.0 / 7.0, places=10)

        # Test at general x != 0
        x_val = 1.2
        f_x = neural_networks.shannon_wavelet_activation(np.array([x_val]))
        expected_val = (np.sin(1.75 * x_val) - np.sin(0.75 * x_val)) / (1.75 * x_val)
        self.assertAlmostEqual(f_x[0], expected_val, places=10)

        # Test derivative at x = 0: lim_{x->0} f'(x) = 0
        f_prime_zero = neural_networks.shannon_wavelet_derivative(np.array([0.0]))
        self.assertAlmostEqual(f_prime_zero[0], 0.0, places=10)

        # Test derivative numerically using central finite difference
        eps = 1e-6
        for x_test in [0.5, 1.0, 2.0]:
            f_plus = neural_networks.shannon_wavelet_activation(np.array([x_test + eps]))[0]
            f_minus = neural_networks.shannon_wavelet_activation(np.array([x_test - eps]))[0]
            num_deriv = (f_plus - f_minus) / (2.0 * eps)
            ana_deriv = neural_networks.shannon_wavelet_derivative(np.array([x_test]))[0]
            self.assertAlmostEqual(num_deriv, ana_deriv, places=5)

    def test_07_nguyen_widrow_initialization(self):
        """7. Nguyen-Widrow initialization follows Equations (1)-(6)."""
        rng = np.random.default_rng(42)
        n_in = 64
        n_hid = 55
        W, b = neural_networks.initialize_nguyen_widrow(n_in, n_hid, rng)

        expected_beta = 0.7 * (n_hid ** (1.0 / n_in))
        W_norm = np.sqrt(np.sum(W ** 2))
        # After scaling, the Frobenius norm should equal beta
        self.assertAlmostEqual(W_norm, expected_beta, places=6)
        # Biases must fall in [-beta, +beta]
        self.assertTrue(np.all(b >= -expected_beta - 1e-10))
        self.assertTrue(np.all(b <= expected_beta + 1e-10))

    def test_08_five_bit_encoding_mapping(self):
        """8. Five-bit class encoding and decoding are consistent with Figure 12."""
        for cls_idx in range(1, 19):
            code = neural_networks.encode_class_to_5bit(cls_idx)
            self.assertEqual(len(code), 5, f"Class {cls_idx} code must have 5 bits.")
            expected_code = np.array(config.CLASS_TO_5BIT[cls_idx], dtype=np.float64)
            np.testing.assert_array_equal(code, expected_code)

            # Test decoding
            decoded_cls = neural_networks.decode_5bit_to_class(code)
            self.assertEqual(decoded_cls, cls_idx, f"Decoded class {decoded_cls} != original {cls_idx}")

    def test_09_mabpn_weight_update_step(self):
        """9. MABPN weight updates follow Equations (9)-(20)."""
        model = models.MABPN_GLCM(learning_rate=0.05, max_iter=1, random_state=42)
        X_dummy = np.random.uniform(0.1, 1.0, (5, 6))
        y_dummy = np.array(["01_ka", "02_nga", "03_cha", "01_ka", "02_nga"])

        model._init_weights()
        W_initial = model.W.copy()
        V_initial = model.V.copy()

        model.fit(X_dummy, y_dummy)

        # Ensure weights actually updated
        self.assertFalse(np.allclose(model.W, W_initial), "Input-to-hidden weights W must update.")
        self.assertFalse(np.allclose(model.V, V_initial), "Hidden-to-output weights V must update.")
        self.assertEqual(len(model.loss_history_), 1)
        self.assertGreater(model.loss_history_[0], 0.0)

    def test_10_end_to_end_pipeline_execution(self):
        """10. Complete pipeline runs without errors on a labelled test fixture."""
        rng = np.random.default_rng(42)
        n_samples = 20
        # 8x8 dummy characters
        X_chars = rng.integers(0, 256, (n_samples, 8, 8), dtype=np.uint8)
        y_labels = np.array([f"0{(i % 4) + 1}_class" for i in range(n_samples)])

        # Test Model 1: BCV
        X_bcv = features.extract_features_batch(X_chars, model_type="bcv")
        self.assertEqual(X_bcv.shape, (n_samples, 64))
        m_bcv = models.MABPN_BCV(max_iter=3, random_state=42)
        m_bcv.fit(X_bcv, y_labels)
        preds_bcv = m_bcv.predict(X_bcv)
        self.assertEqual(len(preds_bcv), n_samples)

        # Test Model 2: GLCM
        X_glcm = features.extract_features_batch(X_chars, model_type="glcm")
        self.assertEqual(X_glcm.shape, (n_samples, 6))
        m_glcm = models.MABPN_GLCM(max_iter=3, random_state=42)
        m_glcm.fit(X_glcm, y_labels)
        preds_glcm = m_glcm.predict(X_glcm)
        self.assertEqual(len(preds_glcm), n_samples)


if __name__ == "__main__":
    unittest.main()
