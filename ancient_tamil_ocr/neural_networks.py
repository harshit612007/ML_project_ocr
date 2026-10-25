"""
Modified Adaptive Backpropagation Network (MABPN) for Ancient Tamil Palm-Leaf Character Recognition.
Reference: Poornima Devi. M and M. Sornam (2020), "Classification of Ancient Handwritten
Tamil Characters on Palm Leaf Inscription Using Modified Adaptive Backpropagation Neural
Network with GLCM Features", ACM Transactions on Asian and Low-Resource Language
Information Processing, Vol. 19, Iss. 6, Article 82.
DOI: https://doi.org/10.1145/3406209

Strictly implements:
1. Nguyen-Widrow Weight Initialization (Equations 1-6)
2. Sigmoid Activation for Input-to-Hidden Layer (Equation 7)
3. Shannon Wavelet Activation for Hidden-to-Output Layer (Equation 8)
4. Modified Adaptive Backpropagation with Linear and Non-linear Errors (Equations 9-20)
5. 5-bit Output Class Encoding and Nearest-Codeword Decoding (Figure 12)
6. Model 1 (MABPN-BCV): Architecture (65-55-5)
7. Model 2 (MABPN-GLCM): Architecture (6-20-5)
"""

from typing import Tuple, List, Optional, Dict, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import config


# =============================================================================
# 1. ACTIVATION FUNCTIONS & DERIVATIVES (EQUATIONS 7 & 8)
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function for input-to-hidden layer (Equation 7, Page 82:9):
      f(h_j) = 1 / (1 + exp(-h_j))
    """
    x_clip = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def sigmoid_derivative(f_x: np.ndarray) -> np.ndarray:
    """
    Gradient of sigmoid activation function f'(h_j) = f(h_j) * (1 - f(h_j)).
    Passed activation output f_x = sigmoid(x).
    """
    return f_x * (1.0 - f_x)


def shannon_wavelet_activation(x: np.ndarray) -> np.ndarray:
    """
    Shannon wavelet activation function for hidden-to-output layer (Equation 8, Page 82:9):
      f(x) = (sin(1.75 * x) - sin(0.75 * x)) / (1.75 * x)
      
    Derived using the inverse Fourier transform in the real domain (Carlo Cattani, 2008).
    Infinitely differentiable and bounded in the frequency domain.
    
    Mathematical limit as x -> 0 (via L'Hopital's rule):
      lim_{x -> 0} f(x) = (1.75*cos(0) - 0.75*cos(0)) / 1.75 = 1.0 / 1.75 = 4/7 (~0.57142857)
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    
    zero_mask = np.abs(x) < 1e-12
    non_zero = ~zero_mask
    
    # Mathematical limit at x = 0
    out[zero_mask] = 4.0 / 7.0
    
    # General case x != 0
    x_nz = x[non_zero]
    out[non_zero] = (np.sin(1.75 * x_nz) - np.sin(0.75 * x_nz)) / (1.75 * x_nz)
    
    return out


def shannon_wavelet_derivative(x: np.ndarray) -> np.ndarray:
    """
    Exact analytical derivative of the Shannon wavelet activation function f'(x):
      f'(x) = d/dx [ (sin(1.75 * x) - sin(0.75 * x)) / (1.75 * x) ]
            = (1.75 * cos(1.75 * x) - 0.75 * cos(0.75 * x)) / (1.75 * x)
              - (sin(1.75 * x) - sin(0.75 * x)) / (1.75 * x^2)
              
    Mathematical limit as x -> 0:
      Since f(x) is an even function (f(-x) = f(x)), its derivative f'(x) is an odd function.
      lim_{x -> 0} f'(x) = 0.0.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    
    zero_mask = np.abs(x) < 1e-12
    non_zero = ~zero_mask
    
    # Mathematical limit at x = 0
    out[zero_mask] = 0.0
    
    # General case x != 0
    x_nz = x[non_zero]
    denom1 = 1.75 * x_nz
    denom2 = 1.75 * (x_nz ** 2)
    
    term1 = (1.75 * np.cos(1.75 * x_nz) - 0.75 * np.cos(0.75 * x_nz)) / denom1
    term2 = (np.sin(1.75 * x_nz) - np.sin(0.75 * x_nz)) / denom2
    out[non_zero] = term1 - term2
    
    return out


def inverse_shannon_target(y: np.ndarray) -> np.ndarray:
    """
    Computes linear target equivalent ly_i = f^{-1}(y_i) for Equation (15).
    For binary target bits y_i in {0, 1}:
      - When y_i = 1: maps to x = 0.0 (where f(0) = 4/7 is the peak activation of Shannon wavelet).
      - When y_i = 0: maps to the fundamental root of f(x) = 0, which occurs when
        sin(1.75 * x) = sin(0.75 * x) => x = pi / 2.5 = 0.4 * pi (~1.256637).
    """
    y = np.asarray(y, dtype=np.float64)
    # Binary mapping: 1 -> 0.0 (maximum), 0 -> 0.4 * pi (root)
    root = 0.4 * np.pi
    return np.where(y > 0.5, 0.0, root)


# =============================================================================
# 2. NGUYEN-WIDROW WEIGHT INITIALIZATION (EQUATIONS 1-6)
# =============================================================================

def initialize_nguyen_widrow(
    n_in: int,
    n_hidden: int,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nguyen-Widrow weight initialization algorithm (Equations 1-6, Page 82:8).
    
    Steps:
      1. Initialize weights w_{ij} with uniform random numbers in (-0.5, +0.5).
      2. Calculate scaling factor beta = 0.7 * (n_hidden)^(1 / n_in) (Eq 1).
      3. Compute Euclidean norm N = sqrt(sum_{i=1}^{n_in} sum_{j=1}^{n_hidden} w_{ij}^2) (Eq 2).
      4. Adjust weights: w_{ij} = beta * w_{ij} / N (Eq 3).
      5. Initialize bias values uniformly in (-beta, +beta) (Step 11).
    """
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)

    # 1. Random initial weights in (-0.5, +0.5)
    W = rng.uniform(-0.5, 0.5, (n_in, n_hidden)).astype(np.float64)

    # 2. Scaling factor beta (Equation 1)
    beta = 0.7 * (n_hidden ** (1.0 / max(1, n_in)))

    # 3. Norm N over all weights (Equation 2)
    N = np.sqrt(np.sum(W ** 2))
    if N < 1e-12:
        N = 1e-12

    # 4. Adjusted weights (Equation 3)
    W = beta * (W / N)

    # 5. Bias initialized in (-beta, +beta) (Step 11)
    b = rng.uniform(-beta, beta, (1, n_hidden)).astype(np.float64)

    return W, b


# =============================================================================
# 3. FIVE-BIT CLASS ENCODING AND DECODING (FIGURE 12)
# =============================================================================

def encode_class_to_5bit(class_index: int) -> np.ndarray:
    """
    Encodes 1-based character class label (1 to 18) into the paper's 5-bit vector (Figure 12).
    """
    if class_index in config.CLASS_TO_5BIT:
        return np.array(config.CLASS_TO_5BIT[class_index], dtype=np.float64)
    # Default binary representation
    bits = [int(b) for b in f"{class_index:05b}"]
    return np.array(bits[-5:], dtype=np.float64)


def decode_5bit_to_class(output_vector: np.ndarray) -> int:
    """
    Decodes continuous 5-bit network output into the nearest class label (1 to 18)
    using minimum Euclidean distance to the 18 valid codewords in Figure 12.
    """
    output_vector = np.asarray(output_vector, dtype=np.float64).flatten()
    best_class = 1
    min_dist = float("inf")

    for cls_idx, code in config.CLASS_TO_5BIT.items():
        code_arr = np.array(code, dtype=np.float64)
        dist = np.sum((output_vector - code_arr) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_class = cls_idx

    return best_class


# =============================================================================
# 4. MODIFIED ADAPTIVE BACKPROPAGATION NETWORK (MABPN) CLASS
# =============================================================================

class ModifiedAdaptiveBackpropNN(BaseEstimator, ClassifierMixin):
    """
    Modified Adaptive Backpropagation Network (MABPN) Classifier.
    Reference: Poornima Devi & Sornam (2020), ACM TALLIP, Article 82.
    
    Implements Equations (9)-(20):
      - Input-to-hidden: Sigmoid activation
      - Hidden-to-output: Shannon wavelet activation
      - Error calculation: Linear errors (e2) + Non-linear errors (e1)
      - Weight updates: Combined gradient update using error scaling terms (mu, lambda).
    """

    def __init__(
        self,
        n_inputs: int = 6,
        n_hidden: int = 20,
        n_outputs: int = 5,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        tol: float = config.CONVERGENCE_ERROR_TARGET,
        random_state: int = config.RANDOM_SEED
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Network weights and biases
        self.W: Optional[np.ndarray] = None   # Input-to-hidden weights (n_inputs x n_hidden)
        self.bh: Optional[np.ndarray] = None  # Hidden layer biases (1 x n_hidden)
        self.V: Optional[np.ndarray] = None   # Hidden-to-output weights (n_hidden x n_outputs)
        self.bo: Optional[np.ndarray] = None  # Output layer biases (1 x n_outputs)

        # Classes and history tracking
        self.classes_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    def _init_weights(self):
        """Initializes weights using Nguyen-Widrow technique (Equations 1-6)."""
        rng = np.random.default_rng(self.random_state)
        # Input to hidden (Equations 1-3)
        self.W, self.bh = initialize_nguyen_widrow(self.n_inputs, self.n_hidden, rng)
        # Hidden to output (Equations 4-6)
        self.V, self.bo = initialize_nguyen_widrow(self.n_hidden, self.n_outputs, rng)

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for a single input vector x (shape: 1 x n_inputs):
          uh = x * W + bh
          ah = sigmoid(uh) (Equation 7)
          uo = ah * V + bo
          d  = shannon_wavelet_activation(uo) (Equation 8)
        """
        uh = np.dot(x, self.W) + self.bh
        ah = sigmoid(uh)
        uo = np.dot(ah, self.V) + self.bo
        d = shannon_wavelet_activation(uo)
        return uh, ah, uo, d

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains MABPN on training set (X, y) using Equations (9)-(20).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_feat = X.shape
        self.n_inputs = n_feat

        # Map labels to 1-based class indices for Figure 12 5-bit encoding
        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        label_to_idx = {label: i + 1 for i, label in enumerate(unique_labels)}

        # Encode target outputs into 5-bit vectors (Figure 12)
        Y_5bit = np.zeros((n_samples, self.n_outputs), dtype=np.float64)
        for i in range(n_samples):
            cls_1based = label_to_idx[y[i]]
            Y_5bit[i] = encode_class_to_5bit(cls_1based)

        # Initialize network weights with Nguyen-Widrow
        self._init_weights()
        self.loss_history_ = []

        alpha = self.learning_rate

        for epoch in range(self.max_iter):
            epoch_error_sq = 0.0

            # Online stochastic / sample-by-sample update matching paper equations (9)-(20)
            indices = np.arange(n_samples)

            for idx in indices:
                x_i = X[idx:idx + 1]       # shape: (1, n_inputs)
                y_i = Y_5bit[idx:idx + 1]  # shape: (1, n_outputs)

                # 1. Forward Propagation
                uh, ah, uo, d = self._forward(x_i)

                # 2. Output Layer Errors (Equations 14-18)
                # Non-linear error (Eq 14): e1i = yi - di
                e1_o = y_i - d

                # Linear error (Eq 15): e2i = lyi - ui, where lyi = f^{-1}(yi)
                ly_i = inverse_shannon_target(y_i)
                e2_o = ly_i - uo

                # Activation derivative at output layer: f'(uo)
                f_prime_uo = shannon_wavelet_derivative(uo)

                # Output delta terms (Eq 16 & 17)
                delta_o1 = e1_o * f_prime_uo
                delta_o2 = e2_o * f_prime_uo

                # Output total error gradient lambda (Eq 18)
                # delta_o_tot = (delta_o1 + delta_o2) / n
                delta_o_tot = (delta_o1 + delta_o2)
                lambda_grad = np.mean(delta_o_tot)

                # 3. Hidden Layer Errors (Equations 9-13)
                # e1j = sum_r f'(ur) * e1r * wrj (Eq 9, where wrj are hidden-to-output weights V)
                e1_h = np.dot(e1_o * f_prime_uo, self.V.T)

                # Sigmoid activation gradient at hidden layer: f'(uh)
                f_prime_uh = sigmoid_derivative(ah)

                # e2j = f'(uj) * sum_r e2r * wrj (Eq 10)
                e2_h = f_prime_uh * np.dot(e2_o, self.V.T)

                # Hidden delta terms (Eq 11 & 12)
                delta_h1 = e1_h * f_prime_uh
                delta_h2 = e2_h * f_prime_uh

                # Hidden total error gradient mu (Eq 13)
                delta_h_tot = delta_h1 + delta_h2
                mu_grad = np.mean(delta_h_tot)

                # 4. Weight Updates (Equations 19 & 20)
                # Delta V for hidden-to-output weights:
                # Delta V_jk = lambda * e1k * f'(uk) * ah_j + lambda * alpha * e2k * ah_j
                grad_V_nonlinear = np.dot(ah.T, e1_o * f_prime_uo)
                grad_V_linear = np.dot(ah.T, e2_o)
                delta_V = lambda_grad * grad_V_nonlinear + lambda_grad * alpha * grad_V_linear
                delta_bo = lambda_grad * (e1_o * f_prime_uo) + lambda_grad * alpha * e2_o

                # Delta W for input-to-hidden weights:
                # Delta W_ij = mu * e1j * f'(uj) * x_i + mu * alpha * e2j * x_i
                grad_W_nonlinear = np.dot(x_i.T, e1_h * f_prime_uh)
                grad_W_linear = np.dot(x_i.T, e2_h)
                delta_W = mu_grad * grad_W_nonlinear + mu_grad * alpha * grad_W_linear
                delta_bh = mu_grad * (e1_h * f_prime_uh) + mu_grad * alpha * e2_h

                # Update weights
                self.V += delta_V
                self.bo += delta_bo
                self.W += delta_W
                self.bh += delta_bh

                # Track error
                epoch_error_sq += np.sum(e1_o ** 2)

            mse = float(epoch_error_sq / (n_samples * self.n_outputs))
            self.loss_history_.append(mse)

            # Stopping condition from Figure 9: check error < 0.0001
            if mse < self.tol:
                break

        return self

    def predict_bits(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous 5-bit output predictions for input X."""
        X = np.asarray(X, dtype=np.float64)
        uh = np.dot(X, self.W) + self.bh
        ah = sigmoid(uh)
        uo = np.dot(ah, self.V) + self.bo
        return shannon_wavelet_activation(uo)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for test samples X using nearest-codeword decoding.
        """
        pred_bits = self.predict_bits(X)
        predictions = []

        for row in pred_bits:
            pred_class_1based = decode_5bit_to_class(row)
            # Map back to original label
            if pred_class_1based <= len(self.classes_):
                label = self.classes_[pred_class_1based - 1]
            else:
                label = self.classes_[-1]
            predictions.append(label)

        return np.array(predictions)


# =============================================================================
# 5. SPECIFIC PAPER PROPOSED MODELS 1 & 2
# =============================================================================

class MABPN_BCV(ModifiedAdaptiveBackpropNN):
    """
    Model 1: MABPN with Binary Coded Value (BCV) input (Section 3.1.2).
    Architecture:
      - 65 input neurons (64 binary pixels + 1 bias)
      - 55 hidden neurons
      - 5 output neurons (5-bit class encoding)
    """
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 150,
        tol: float = config.CONVERGENCE_ERROR_TARGET,
        random_state: int = config.RANDOM_SEED
    ):
        super().__init__(
            n_inputs=config.BCV_INPUT_DIM,       # 64
            n_hidden=config.BCV_HIDDEN_NEURONS,  # 55
            n_outputs=config.BCV_OUTPUT_NEURONS, # 5
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )


class MABPN_GLCM(ModifiedAdaptiveBackpropNN):
    """
    Model 2: MABPN with GLCM feature extraction input (Section 3.1.3).
    Architecture:
      - 6 input neurons (Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM) + 1 bias
      - 20 hidden neurons
      - 5 output neurons (5-bit class encoding)
    """
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 120,
        tol: float = config.CONVERGENCE_ERROR_TARGET,
        random_state: int = config.RANDOM_SEED
    ):
        super().__init__(
            n_inputs=config.GLCM_INPUT_DIM,       # 6
            n_hidden=config.GLCM_HIDDEN_NEURONS,  # 20
            n_outputs=config.GLCM_OUTPUT_NEURONS, # 5
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
