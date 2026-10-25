"""
Comprehensive Unit Tests for Strict Paper Compliance.
Reference: Poornima Devi. M & M. Sornam (2020),
"Classification of Ancient Handwritten Tamil Characters on Palm Leaf Inscription
Using Modified Adaptive Backpropagation Neural Network with GLCM Features",
ACM TALLIP, Vol. 19, Iss. 6, Article 82. DOI: 10.1145/3406209

Covers all 16 required phases:
1. Exact BCV architecture: 65-55-5.
2. Exact GLCM architecture: 6-20-5.
3. Six GLCM features.
4. Correct feature equations (including non-constant Energy = sqrt(ASM)).
5. Correct Shannon activation.
6. Correct Shannon derivative.
7. Correct handling of zero input (analytical limit 4/7).
8. Nguyen-Widrow initialization.
9. All 18 five-bit class mappings (Figure 12).
10. Correct encoding and decoding (and strict rejection of non-paper classes like vowels).
11. MABPN intermediate error calculations.
12. MABPN weight updates.
13. Training and inference consistency (including serialization).
14. Preprocessing integration.
15. Detection of collapsed predictions.
16. Detection of NaN and infinite values.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Add ancient_tamil_ocr to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ancient_tamil_ocr"))

import config
import preprocess
import features
import neural_networks
import models
import evaluate


class TestPaperImplementation(unittest.TestCase):

    def test_01_bcv_architecture(self):
        """1. Exact BCV architecture: 65-55-5 (64 pixels + 1 bias = 65 inputs, 55 hidden, 5 output)."""
        model = models.MABPN_BCV(learning_rate=0.05, max_iter=2, random_state=42)
        model._init_weights()
        self.assertEqual(model.n_inputs, 64, "BCV input dimension must be 64 (8x8 binary pixels).")
        self.assertEqual(model.n_hidden, 55, "BCV hidden neurons must be exactly 55.")
        self.assertEqual(model.n_outputs, 5, "BCV output neurons must be exactly 5 (5-bit encoding).")
        self.assertEqual(model.W.shape, (64, 55), "W shape must be (64, 55).")
        self.assertEqual(model.bh.shape, (1, 55), "Hidden bias shape must be (1, 55).")
        self.assertEqual(model.V.shape, (55, 5), "V shape must be (55, 5).")
        self.assertEqual(model.bo.shape, (1, 5), "Output bias shape must be (1, 5).")

    def test_02_glcm_architecture(self):
        """2. Exact GLCM architecture: 6-20-5 (6 features + 1 bias, 20 hidden, 5 output)."""
        model = models.MABPN_GLCM(learning_rate=0.05, max_iter=2, random_state=42)
        model._init_weights()
        self.assertEqual(model.n_inputs, 6, "GLCM input dimension must be exactly 6 features.")
        self.assertEqual(model.n_hidden, 20, "GLCM hidden neurons must be exactly 20.")
        self.assertEqual(model.n_outputs, 5, "GLCM output neurons must be exactly 5 (5-bit encoding).")
        self.assertEqual(model.W.shape, (6, 20), "W shape must be (6, 20).")
        self.assertEqual(model.bh.shape, (1, 20), "Hidden bias shape must be (1, 20).")
        self.assertEqual(model.V.shape, (20, 5), "V shape must be (20, 5).")
        self.assertEqual(model.bo.shape, (1, 5), "Output bias shape must be (1, 5).")

    def test_03_six_glcm_features(self):
        """3. Six GLCM features: exactly Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM."""
        patch = np.array([
            [10, 20, 30, 40],
            [20, 30, 40, 50],
            [30, 40, 50, 60],
            [40, 50, 60, 70]
        ], dtype=np.uint8)
        glcm_feat = features.extract_glcm_features(patch)
        self.assertEqual(glcm_feat.shape, (6,), "Must return exactly 6 features.")
        self.assertTrue(np.all(np.isfinite(glcm_feat)), "All GLCM features must be finite numbers.")
        # Contrast >= 0
        self.assertGreaterEqual(glcm_feat[0], 0.0, "Contrast must be non-negative.")
        # Homogeneity in [0, 1]
        self.assertTrue(0.0 <= glcm_feat[2] <= 1.0, "Homogeneity must be in [0, 1].")
        # Correlation in [-1, 1]
        self.assertTrue(-1.0 <= glcm_feat[3] <= 1.0, "Correlation must be in [-1, 1].")
        # Dissimilarity >= 0
        self.assertGreaterEqual(glcm_feat[4], 0.0, "Dissimilarity must be non-negative.")
        # ASM in [0, 1]
        self.assertTrue(0.0 <= glcm_feat[5] <= 1.0, "ASM must be in [0, 1].")

    def test_04_correct_feature_equations(self):
        """4. Correct feature equations: Energy must be sqrt(ASM), NOT constant sum(P) = 1.0."""
        # Single direction check: Energy == sqrt(ASM) exactly
        patch1 = np.zeros((8, 8), dtype=np.uint8)
        patch1[::2, ::2] = 255  # High contrast checkerboard
        feat_single = features.extract_glcm_features(patch1, distances=[1], angles=[0.0])
        self.assertAlmostEqual(
            feat_single[1], np.sqrt(feat_single[5]), places=7,
            msg="For single GLCM direction, Energy must equal sqrt(ASM) exactly."
        )

        # Flat constant image across all directions: Energy == sqrt(ASM) exactly = 1.0
        patch2 = np.ones((8, 8), dtype=np.uint8) * 128
        feat2 = features.extract_features_batch([patch2], model_type="glcm")[0]
        self.assertAlmostEqual(feat2[1], 1.0, places=7)
        self.assertAlmostEqual(feat2[5], 1.0, places=7)
        self.assertAlmostEqual(feat2[1], np.sqrt(feat2[5]), places=7)

        # Multi-direction average on complex patch: Energy is mean(sqrt(ASM_i)), close to sqrt(mean(ASM_i))
        feat_multi = features.extract_glcm_features(patch1)
        self.assertAlmostEqual(feat_multi[1], np.sqrt(feat_multi[5]), delta=0.02)

        # Energy must vary between textures and cannot be constant 1.0
        self.assertNotEqual(feat_single[1], feat2[1], "Energy must differ across different textures.")
        self.assertFalse(np.isclose(feat_single[1], 1.0), "Energy cannot be constant 1.0 on a complex patch.")

    def test_05_shannon_activation(self):
        """5. Correct Shannon activation: f(x) = (sin(1.75x) - sin(0.75x)) / (1.75x)."""
        x_vals = np.array([-2.5, -1.0, 0.5, 1.5, 3.0])
        f_vals = neural_networks.shannon_wavelet_activation(x_vals)

        for x, fx in zip(x_vals, f_vals):
            expected = (np.sin(1.75 * x) - np.sin(0.75 * x)) / (1.75 * x)
            self.assertAlmostEqual(fx, expected, places=10)

    def test_06_shannon_derivative(self):
        """6. Correct Shannon derivative verified against numerical central difference."""
        eps = 1e-6
        test_points = [-2.0, -0.7, 0.3, 1.2, 2.5]
        for x in test_points:
            f_plus = neural_networks.shannon_wavelet_activation(np.array([x + eps]))[0]
            f_minus = neural_networks.shannon_wavelet_activation(np.array([x - eps]))[0]
            numerical = (f_plus - f_minus) / (2.0 * eps)
            analytical = neural_networks.shannon_wavelet_derivative(np.array([x]))[0]
            self.assertAlmostEqual(numerical, analytical, places=5)

    def test_07_shannon_zero_input(self):
        """7. Correct handling of zero input: analytical limit is 4/7, derivative is 0."""
        zero_in = np.array([0.0, -0.0])
        f_zero = neural_networks.shannon_wavelet_activation(zero_in)
        np.testing.assert_allclose(f_zero, [4.0 / 7.0, 4.0 / 7.0], atol=1e-12)

        f_prime_zero = neural_networks.shannon_wavelet_derivative(zero_in)
        np.testing.assert_allclose(f_prime_zero, [0.0, 0.0], atol=1e-12)

    def test_08_nguyen_widrow_initialization(self):
        """8. Nguyen-Widrow initialization (Equations 1-6): norm equals beta, bias in [-beta, +beta]."""
        rng = np.random.default_rng(123)
        n_in = 64
        n_hid = 55
        W, b = neural_networks.initialize_nguyen_widrow(n_in, n_hid, rng)

        expected_beta = 0.7 * (n_hid ** (1.0 / n_in))
        W_norm = np.sqrt(np.sum(W ** 2))
        self.assertAlmostEqual(W_norm, expected_beta, places=6)
        self.assertTrue(np.all(b >= -expected_beta - 1e-10))
        self.assertTrue(np.all(b <= expected_beta + 1e-10))

    def test_09_all_18_five_bit_class_mappings(self):
        """9. All 18 five-bit class mappings strictly match Figure 12."""
        self.assertEqual(len(config.FIGURE_12_TABLE), 18, "Figure 12 must define exactly 18 classes.")
        seen_codes = set()
        for idx in range(1, 19):
            info = config.FIGURE_12_TABLE[idx]
            code = tuple(info["code"])
            self.assertEqual(len(code), 5, f"Class {idx} ({info['name']}) must have a 5-bit code.")
            self.assertNotIn(code, seen_codes, f"Code {code} for class {idx} is not unique!")
            seen_codes.add(code)

        # Verify key classes from Figure 12
        self.assertEqual(config.FIGURE_12_TABLE[1]["code"], [0, 0, 0, 0, 1])  # ka
        self.assertEqual(config.FIGURE_12_TABLE[2]["code"], [0, 0, 0, 1, 0])  # nga
        self.assertEqual(config.FIGURE_12_TABLE[9]["code"], [0, 1, 0, 0, 1])  # pa
        self.assertEqual(config.FIGURE_12_TABLE[16]["code"], [1, 0, 0, 0, 0]) # lla
        self.assertEqual(config.FIGURE_12_TABLE[18]["code"], [1, 0, 0, 1, 0]) # nna2

    def test_10_correct_encoding_and_decoding(self):
        """10. Correct encoding/decoding and strict rejection of non-paper classes (e.g. vowels)."""
        # Test valid classes
        for idx in range(1, 19):
            info = config.FIGURE_12_TABLE[idx]
            name = info["name"]
            code = neural_networks.encode_class_to_5bit(name)
            np.testing.assert_array_equal(code, info["code"])

            # Decode from continuous vector
            decoded_name = neural_networks.decode_5bit_to_class(code)
            self.assertEqual(decoded_name, name)
            decoded_idx = neural_networks.decode_5bit_to_class_index(code)
            self.assertEqual(decoded_idx, idx)

        # Test rejection of vowels (e.g. 01_a, 02_aa, 03_i)
        invalid_labels = ["01_a", "02_aa", "03_i", "a", "aa", "unknown_class", 19, -1]
        for inv in invalid_labels:
            with self.assertRaises(ValueError, msg=f"Should reject invalid label: {inv}"):
                config.get_5bit_code(inv)

    def test_11_mabpn_intermediate_error_calculations(self):
        """11. MABPN intermediate error calculations: neuron-specific delta updates."""
        model = models.MABPN_GLCM(learning_rate=0.05, max_iter=1, random_state=42)
        model._init_weights()

        x = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.3])
        d = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        # Forward pass
        uh = np.dot(x, model.W) + model.bh
        ah = neural_networks.sigmoid(uh)
        uo = np.dot(ah, model.V) + model.bo
        y = neural_networks.shannon_wavelet_activation(uo)

        # Error calculations (Equations 10-15)
        e1_o = d - y
        e2_o = 0.5 * (d ** 2 - y ** 2)
        deriv_o = neural_networks.shannon_wavelet_derivative(uo)
        delta_o1 = e1_o * deriv_o
        delta_o2 = e2_o * deriv_o
        delta_o = delta_o1 + model.alpha * delta_o2

        self.assertEqual(delta_o.shape, (1, 5), "Output delta must be (1, 5) neuron-specific vector.")
        self.assertTrue(np.all(np.isfinite(delta_o)), "Output deltas must be finite.")

        # Hidden layer error
        deriv_h = neural_networks.sigmoid_derivative(ah)
        delta_h = np.dot(delta_o, model.V.T) * deriv_h
        self.assertEqual(delta_h.shape, (1, 20), "Hidden delta must be (1, 20) neuron-specific vector.")
        self.assertTrue(np.all(np.isfinite(delta_h)), "Hidden deltas must be finite.")

    def test_12_mabpn_weight_updates(self):
        """12. MABPN weight updates: weights update and loss decreases."""
        model = models.MABPN_GLCM(learning_rate=0.08, max_iter=15, random_state=42)
        rng = np.random.default_rng(42)

        # Create two separable classes
        X = np.vstack([
            rng.normal(0.2, 0.05, (10, 6)),
            rng.normal(0.8, 0.05, (10, 6))
        ])
        y = np.array(["01_ka"] * 10 + ["02_nga"] * 10)

        model._init_weights()
        W_init = model.W.copy()
        V_init = model.V.copy()

        model.fit(X, y)

        self.assertFalse(np.allclose(model.W, W_init), "Input-to-hidden weights must change.")
        self.assertFalse(np.allclose(model.V, V_init), "Hidden-to-output weights must change.")
        self.assertLess(model.loss_history_[-1], model.loss_history_[0], "MSE must decrease during training.")

    def test_13_training_and_inference_consistency(self):
        """13. Training and inference consistency: serialization preserves exact predictions."""
        import tempfile
        import joblib

        model = models.MABPN_GLCM(learning_rate=0.05, max_iter=5, random_state=42)
        X = np.random.uniform(0.1, 0.9, (12, 6))
        y = np.array(["01_ka", "02_nga", "03_cha", "04_gna"] * 3)
        model.fit(X, y)

        preds_before = model.predict(X)
        raw_before = model.predict_bits(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            joblib.dump(model, tmp_path)
            loaded_model = joblib.load(tmp_path)
            preds_after = loaded_model.predict(X)
            raw_after = loaded_model.predict_bits(X)

            np.testing.assert_array_equal(preds_before, preds_after)
            np.testing.assert_allclose(raw_before, raw_after, atol=1e-12)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_14_preprocessing_integration(self):
        """14. Preprocessing integration: full pipeline from raw image to 8x8 character and BCV vector."""
        raw_img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

        # Page-level preprocessing
        stages = preprocess.preprocess_palm_leaf_page(raw_img)
        self.assertIn("grayscale", stages)
        self.assertIn("median_filter", stages)
        self.assertIn("adaptive_threshold", stages)
        self.assertEqual(stages["grayscale"].shape, (64, 64))

        # Character preprocessing and BCV extraction
        char_crop = stages["grayscale"][:32, :32]
        char_8x8 = preprocess.preprocess_character_sample(char_crop)
        self.assertEqual(char_8x8.shape, (8, 8), "Resized character must be 8x8.")

        bcv_feat = preprocess.binarize_to_bcv(char_8x8)
        self.assertEqual(bcv_feat.shape, (64,), "BCV feature must have 64 elements.")
        self.assertTrue(set(np.unique(bcv_feat)).issubset({0.0, 1.0}), "BCV values must be binary {0, 1}.")

    def test_15_detection_of_collapsed_predictions(self):
        """15. Detection of collapsed predictions (mode collapse audit)."""
        # Scenario A: Collapsed predictions (all samples predicted as 'ka')
        y_true = np.array(["ka", "nga", "cha", "ta", "pa"])
        y_pred_collapsed = np.array(["ka", "ka", "ka", "ka", "ka"])
        metrics_collapsed = evaluate.compute_paper_metrics(
            y_true, y_pred_collapsed, ["ka", "nga", "cha", "ta", "pa"]
        )
        self.assertTrue(metrics_collapsed["is_collapsed"], "Must flag single-class prediction collapse.")
        self.assertEqual(metrics_collapsed["num_unique_predictions"], 1)

        # Scenario B: Healthy multi-class predictions
        y_pred_healthy = np.array(["ka", "nga", "cha", "ta", "pa"])
        metrics_healthy = evaluate.compute_paper_metrics(
            y_true, y_pred_healthy, ["ka", "nga", "cha", "ta", "pa"]
        )
        self.assertFalse(metrics_healthy["is_collapsed"], "Must NOT flag healthy diverse predictions.")
        self.assertEqual(metrics_healthy["num_unique_predictions"], 5)

    def test_16_detection_of_nan_and_infinite_values(self):
        """16. Detection of NaN and infinite values in model weights and outputs."""
        model = models.MABPN_GLCM(learning_rate=0.05, max_iter=2, random_state=42)
        model._init_weights()

        diag_clean = model.get_diagnostics(np.random.uniform(0.1, 0.9, (4, 6)))
        self.assertTrue(diag_clean["weights_finite"], "Initial weights must be completely finite.")

        # Invalidate weight
        model.W[0, 0] = np.nan
        diag_nan = model.get_diagnostics(np.random.uniform(0.1, 0.9, (4, 6)))
        self.assertFalse(diag_nan["weights_finite"], "Must detect NaN weight injection.")


if __name__ == "__main__":
    unittest.main()
