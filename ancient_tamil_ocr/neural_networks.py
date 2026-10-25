"""
Neural Network Implementations for Ancient Tamil Palm-Leaf Manuscript Recognition.

Directly implements the neural architectures from the two reference papers:
1. Paper 1 (Ramya & Parvathavarthini, 2014):
   Feed Forward Back Propagation Neural Network (FFBNN)
   - Standard random weight initialization
   - Sigmoid activation function
   - Standard backpropagation with momentum and fixed learning rate

2. Paper 2 (Poornima Devi & Sornam, 2020):
   Modified Adaptive Backpropagation Network (MABPN)
   - Nguyen-Widrow weight initialization algorithm
   - Shannon activation function f(x) = sin(x) - sin(sin(x))
   - Adaptive learning rate mechanism (updates eta dynamically based on error trend)
"""

from typing import Tuple, List, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


# =============================================================================
# Activation Functions & Derivatives
# =============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable standard logistic sigmoid activation."""
    z_clipped = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given activation output a = sigmoid(z)."""
    return a * (1.0 - a)


def shannon_activation(z: np.ndarray) -> np.ndarray:
    """
    Exact Shannon Wavelet Activation Function (Poornima Devi & Sornam, 2020, Equation 8):
      f(x) = [sin(1.75x) - sin(0.75x)] / (1.75x)
      
    Analytical Limit as x -> 0 (L'Hopital's Rule):
      lim_{x -> 0} f(x) = (1.75 - 0.75) / 1.75 = 1.0 / 1.75 = 4/7 approx 0.57142857
    """
    z = np.asarray(z, dtype=np.float64)
    abs_z = np.abs(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (np.sin(1.75 * z) - np.sin(0.75 * z)) / (1.75 * z)
    return np.where(abs_z < 1e-12, 4.0 / 7.0, f)


def shannon_derivative(z: np.ndarray) -> np.ndarray:
    """
    Exact Analytical Derivative of Shannon Wavelet Activation (Equation 8):
      f'(x) = [1.75*cos(1.75x) - 0.75*cos(0.75x)] / (1.75x) - [sin(1.75x) - sin(0.75x)] / (1.75x^2)
      
    Analytical Limit as x -> 0:
      Because f(x) is an even function, f'(x) is odd: lim_{x -> 0} f'(x) = 0.0
    """
    z = np.asarray(z, dtype=np.float64)
    abs_z = np.abs(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (1.75 * np.cos(1.75 * z) - 0.75 * np.cos(0.75 * z)) / (1.75 * z)
        term2 = (np.sin(1.75 * z) - np.sin(0.75 * z)) / (1.75 * (z ** 2))
        f_prime = term1 - term2
    return np.where(abs_z < 1e-12, 0.0, f_prime)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for multi-class probability outputs."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.maximum(1e-12, np.sum(exp_z, axis=1, keepdims=True))


# =============================================================================
# Weight Initialization Algorithms
# =============================================================================

def initialize_random_weights(
    n_in: int,
    n_out: int,
    rng: np.random.Generator,
    scale: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard random weight initialization (Paper 1 - Ramya & Parvathavarthini, 2014):
      Weights uniformly distributed in [-scale, +scale].
      Biases initialized to zeros or small uniform values.
    """
    W = rng.uniform(-scale, scale, (n_in, n_out)).astype(np.float64)
    b = rng.uniform(-0.1, 0.1, (1, n_out)).astype(np.float64)
    return W, b


def initialize_nguyen_widrow(
    n_in: int,
    n_hidden: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nguyen-Widrow weight initialization algorithm (Paper 2 - Poornima Devi & Sornam, 2020):
    Distributes the active regions of hidden layer neurons uniformly across the input space.
    
    Steps:
      1. Initialize weights w_ij uniformly in [-0.5, 0.5].
      2. For each hidden neuron j, calculate Euclidean norm: ||w_j|| = sqrt(sum_i w_ij^2).
      3. Compute scaling factor beta = 0.7 * (n_hidden)^(1 / n_in).
      4. Re-scale weights: w_ij = beta * w_ij / ||w_j||.
      5. Initialize biases b_j uniformly in [-beta, beta].
    """
    # 1. Random initial weights
    W = rng.uniform(-0.5, 0.5, (n_in, n_hidden)).astype(np.float64)
    
    # 2. Euclidean norm per hidden neuron (column-wise)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    
    # 3. Scaling factor beta
    # Safeguard against overflow when n_in is large
    power = 1.0 / max(1, n_in)
    beta = 0.7 * (n_hidden ** power)
    
    # 4. Scale weights
    W = beta * (W / norms)
    
    # 5. Biases uniformly distributed in [-beta, +beta]
    b = rng.uniform(-beta, beta, (1, n_hidden)).astype(np.float64)
    
    return W, b


# =============================================================================
# Paper 1: Feed Forward Back Propagation Neural Network (FFBNN)
# Ramya & Parvathavarthini (2014)
# =============================================================================

class FeedForwardBackpropNN(BaseEstimator, ClassifierMixin):
    """
    Adapted Implementation: Feed Forward Back Propagation Neural Network (FFBNN).
    Reference: Ramya & Parvathavarthini (2014), "Ancient Tamil Character Recognition
    System using Feed Forward Back Propagation Neural Network", Journal of Computer Science.
    
    Paper 1 Adaptation Notes:
      - The original 2014 paper tested FFBNN on binary-coded character grid inputs with
        standard momentum backpropagation.
      - In this multi-classifier benchmark pipeline, FFBNN is adapted to process the project's
        continuous feature representations (Gabor, SIFT, HOG, PCA, and GLCM) in addition to
        binary grids, providing a direct empirical baseline against the 2020 MABPN model.
      
    Architecture & Learning Rule:
      - 3-Layer Architecture: Input -> Hidden (Sigmoid) -> Output (Softmax probabilities)
      - Standard random uniform weight initialization [-0.5, 0.5]
      - Gradient descent backpropagation with momentum (v = momentum * v - lr * grad)
      - Cross-entropy loss tracking with early stopping tolerance
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64,),
        learning_rate: float = 0.05,
        momentum: float = 0.8,
        max_iter: int = 400,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Internal state
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.label_binarizer: Optional[LabelBinarizer] = None
        self.classes_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the FFBNN on training dataset (X, y)."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Encode labels to one-hot vectors
        self.label_binarizer = LabelBinarizer()
        Y_onehot = self.label_binarizer.fit_transform(y)
        if Y_onehot.shape[1] == 1:
            Y_onehot = np.hstack([1.0 - Y_onehot, Y_onehot])
        self.classes_ = self.label_binarizer.classes_
        n_classes = Y_onehot.shape[1]

        n_hidden = self.hidden_layer_sizes[0]

        # Initialize weights with standard random uniform distribution (Paper 1)
        self.W1, self.b1 = initialize_random_weights(n_features, n_hidden, rng, scale=0.5)
        self.W2, self.b2 = initialize_random_weights(n_hidden, n_classes, rng, scale=0.5)

        # Velocity matrices for momentum
        v_W1 = np.zeros_like(self.W1)
        v_b1 = np.zeros_like(self.b1)
        v_W2 = np.zeros_like(self.W2)
        v_b2 = np.zeros_like(self.b2)

        self.loss_history_ = []

        for epoch in range(self.max_iter):
            # 1. Forward Propagation
            # Input -> Hidden
            z1 = np.dot(X, self.W1) + self.b1
            a1 = sigmoid(z1)  # Hidden activations

            # Hidden -> Output
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = softmax(z2)  # Output probabilities

            # 2. Loss Calculation (Cross-Entropy with epsilon clipping)
            probs_clip = np.clip(a2, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(np.sum(Y_onehot * np.log(probs_clip), axis=1))
            self.loss_history_.append(float(loss))

            # Early stopping check
            if loss < self.tol:
                break

            # 3. Backpropagation (Output Layer Error)
            # Softmax + Cross-Entropy gradient simplifies to: delta2 = a2 - y
            delta2 = (a2 - Y_onehot) / n_samples

            # Gradients for layer 2
            grad_W2 = np.dot(a1.T, delta2)
            grad_b2 = np.sum(delta2, axis=0, keepdims=True)

            # Backpropagation (Hidden Layer Error)
            delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(a1)

            # Gradients for layer 1
            grad_W1 = np.dot(X.T, delta1)
            grad_b1 = np.sum(delta1, axis=0, keepdims=True)

            # 4. Weight Updates with Momentum
            v_W2 = self.momentum * v_W2 - self.learning_rate * grad_W2
            v_b2 = self.momentum * v_b2 - self.learning_rate * grad_b2
            v_W1 = self.momentum * v_W1 - self.learning_rate * grad_W1
            v_b1 = self.momentum * v_b1 - self.learning_rate * grad_b1

            self.W2 += v_W2
            self.b2 += v_b2
            self.W1 += v_W1
            self.b1 += v_b1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities for test samples X."""
        X = np.asarray(X, dtype=np.float64)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return softmax(z2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for test samples X."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]


# =============================================================================
# Paper 2: Modified Adaptive Backpropagation Network (MABPN)
# Poornima Devi & Sornam (2020)
# =============================================================================

class ModifiedAdaptiveBackpropNN(BaseEstimator, ClassifierMixin):
    """
    Modified Adaptive Backpropagation Network (MABPN).
    Reference: Poornima Devi & Sornam (2020), ACM TALLIP, Article 82.
    
    Faithful Implementation of Equations (1)-(20):
      1. Nguyen-Widrow Weight Initialization (Equations 1-6):
         Scales initial weights by beta = 0.7 * (n_hidden)^(1/n_in) to distribute
         active regions evenly and avoid activation saturation.
      2. Exact Shannon Wavelet Activation Function (Equation 8):
         f(x) = [sin(1.75x) - sin(0.75x)] / (1.75x) with analytical limit 4/7 at x=0.
      3. Dual Linear & Non-Linear Error Formulation (Equations 9-18):
         - Non-linear direct error: e_1k = d_k - y_k (Eq. 14)
         - Linear quadratic error:   e_2k = 0.5 * (d_k^2 - y_k^2) (Eq. 15)
         - Output deltas: delta_o1k = e_1k * f'(u_ok), delta_o2k = e_2k * f'(u_ok) (Eq. 16-17)
         - Combined output delta: delta_ok = delta_o1k + alpha * delta_o2k
         - Backpropagated hidden delta: delta_hj = (sum_k delta_ok * v_jk) * g'(u_hj) (Eq. 9-12)
      4. Neuron-Specific Adaptive Weight Updates (Equations 19-20):
         - Delta v_jk = eta * a_hj * delta_ok
         - Delta w_ij = eta * x_i * delta_hj
      5. Adaptive Learning Rate Control:
         Dynamically increases eta by lr_inc when error decreases, and decodes eta
         by lr_dec when error rises, guaranteeing rapid and stable convergence.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64,),
        initial_learning_rate: float = 0.05,
        alpha: float = 0.05,
        lr_inc: float = 1.05,
        lr_dec: float = 0.70,
        max_error_ratio: float = 1.04,
        momentum: float = 0.85,
        activation: str = "shannon",
        max_iter: int = 400,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.initial_learning_rate = initial_learning_rate
        self.alpha = alpha
        self.lr_inc = lr_inc
        self.lr_dec = lr_dec
        self.max_error_ratio = max_error_ratio
        self.momentum = momentum
        self.activation = activation
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Internal state
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.label_binarizer: Optional[LabelBinarizer] = None
        self.classes_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.lr_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the MABPN using Nguyen-Widrow init, Shannon activation, and adaptive MABPN updates."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Encode labels to target vectors
        self.label_binarizer = LabelBinarizer()
        Y_onehot = self.label_binarizer.fit_transform(y)
        if Y_onehot.shape[1] == 1:
            Y_onehot = np.hstack([1.0 - Y_onehot, Y_onehot])
        self.classes_ = self.label_binarizer.classes_
        n_classes = Y_onehot.shape[1]

        n_hidden = self.hidden_layer_sizes[0]

        # 1. Nguyen-Widrow Weight Initialization (Equations 1-6)
        self.W1, self.b1 = initialize_nguyen_widrow(n_features, n_hidden, rng)
        self.W2, self.b2 = initialize_nguyen_widrow(n_hidden, n_classes, rng)

        # Momentum velocity buffers
        v_W1 = np.zeros_like(self.W1)
        v_b1 = np.zeros_like(self.b1)
        v_W2 = np.zeros_like(self.W2)
        v_b2 = np.zeros_like(self.b2)

        current_lr = self.initial_learning_rate
        prev_loss = np.inf
        self.loss_history_ = []
        self.lr_history_ = []

        for epoch in range(self.max_iter):
            # 2. Forward Propagation
            # Input -> Hidden: Sigmoid (Equation 7)
            uh = np.dot(X, self.W1) + self.b1
            ah = sigmoid(uh)

            # Hidden -> Output: Shannon Wavelet (Equation 8) or Softmax for probabilities
            uo = np.dot(ah, self.W2) + self.b2
            if self.activation == "shannon":
                yo = shannon_activation(uo)
            else:
                yo = sigmoid(uo)

            # 3. Error Calculations (Equations 14-17)
            # e1: direct error (Equation 14)
            e1 = Y_onehot - yo
            # e2: non-linear quadratic error (Equation 15)
            e2 = 0.5 * (Y_onehot ** 2 - yo ** 2)

            # Mean squared error loss
            mse_loss = float(np.mean(e1 ** 2))
            self.loss_history_.append(mse_loss)
            self.lr_history_.append(float(current_lr))

            # Early stopping check
            if mse_loss < self.tol:
                break

            # 4. Adaptive Learning Rate Adjustment (Section 3.1.2)
            if epoch > 0:
                if mse_loss < prev_loss:
                    current_lr = min(current_lr * self.lr_inc, 1.0)
                elif mse_loss > prev_loss * self.max_error_ratio:
                    current_lr = max(current_lr * self.lr_dec, 1e-4)
                    v_W1 *= 0.5
                    v_W2 *= 0.5
            prev_loss = mse_loss

            # 5. Output Layer Deltas (Equations 16-17)
            if self.activation == "shannon":
                deriv_o = shannon_derivative(uo)
            else:
                deriv_o = sigmoid_derivative(yo)

            delta_o1 = e1 * deriv_o
            delta_o2 = e2 * deriv_o
            delta_o = (delta_o1 + self.alpha * delta_o2) / n_samples

            # 6. Hidden Layer Deltas Backpropagation (Equations 9-12)
            deriv_h = sigmoid_derivative(ah)
            delta_h = np.dot(delta_o, self.W2.T) * deriv_h

            # 7. Neuron-Specific Weight Updates with Adaptive Learning Rate (Equations 19-20)
            grad_W2 = np.dot(ah.T, delta_o)
            grad_b2 = np.sum(delta_o, axis=0, keepdims=True)
            grad_W1 = np.dot(X.T, delta_h)
            grad_b1 = np.sum(delta_h, axis=0, keepdims=True)

            v_W2 = self.momentum * v_W2 + current_lr * grad_W2
            v_b2 = self.momentum * v_b2 + current_lr * grad_b2
            v_W1 = self.momentum * v_W1 + current_lr * grad_W1
            v_b1 = self.momentum * v_b1 + current_lr * grad_b1

            self.W2 += v_W2
            self.b2 += v_b2
            self.W1 += v_W1
            self.b1 += v_b1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes normalized class probabilities for test samples X."""
        X = np.asarray(X, dtype=np.float64)
        uh = np.dot(X, self.W1) + self.b1
        ah = sigmoid(uh)
        uo = np.dot(ah, self.W2) + self.b2
        if self.activation == "shannon":
            raw_out = shannon_activation(uo)
            # Softmax normalize over the outputs for probabilistic interpretation
            return softmax(raw_out)
        else:
            return softmax(uo)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for test samples X."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]
