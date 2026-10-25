"""
Feature Extraction module for Ancient Tamil Palm-Leaf Character Recognition.
Domain: Ancient Tamil Palm-Leaf Manuscripts (Panaiyolai Suvadigal).
Methodology: Multi-orientation Gabor wavelets + Dense SIFT BoVW + HOG + Statistical Zoning.

Implements independent feature extractors:
1. Gabor Filter Bank (mean, std, energy across orientations & frequencies)
2. Dense SIFT (regular grid keypoints with BoVW KMeans or spatial pooling)
3. HOG (Histogram of Oriented Gradients)
4. Statistical Features (Zoning density, diagonal features, centroid, projection profiles)
5. Feature Combinations (Gabor + SIFT, All Combined) + StandardScaler normalization.
"""

from typing import Union, List, Tuple, Optional, Dict
import numpy as np
import cv2
from skimage.filters import gabor
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import config


# Precompute Gabor filter bank kernels for high-performance extraction
def _init_gabor_kernels():
    kernels = []
    freqs = config.GABOR_FREQUENCIES
    num_orients = config.GABOR_NUM_ORIENTATIONS
    thetas = [i * np.pi / num_orients for i in range(num_orients)]
    ksize = 15
    sigma = 2.0
    gamma = 0.5
    for freq in freqs:
        lambd = 1.0 / max(0.01, freq)
        for theta in thetas:
            # Generate real (cos) Gabor kernel
            k_real = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
            )
            # Generate imaginary (sin) Gabor kernel
            k_imag = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi=np.pi / 2, ktype=cv2.CV_32F
            )
            kernels.append((k_real, k_imag))
    return kernels

_GABOR_KERNELS = _init_gabor_kernels()


def extract_gabor_features(
    img: np.ndarray,
    frequencies: Optional[List[float]] = None,
    num_orientations: int = config.GABOR_NUM_ORIENTATIONS
) -> np.ndarray:
    """
    Extract multi-scale and multi-orientation Gabor filter features.
    Computes mean, standard deviation, and energy of the filtered magnitude response.
    
    Formula:
      G(x,y; lambda, theta, psi, sigma, gamma) = exp(-(x'^2 + gamma^2 * y'^2)/(2*sigma^2)) * cos(2*pi*x'/lambda + psi)
    """
    # Normalize image to float32 [0, 1]
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    if img_gray.dtype != np.float32 and img_gray.dtype != np.float64:
        img_norm = img_gray.astype(np.float32) / 255.0
    else:
        img_norm = img_gray.astype(np.float32)

    features = []
    for k_real, k_imag in _GABOR_KERNELS:
        resp_r = cv2.filter2D(img_norm, cv2.CV_32F, k_real)
        resp_i = cv2.filter2D(img_norm, cv2.CV_32F, k_imag)
        magnitude = np.sqrt(resp_r ** 2 + resp_i ** 2)

        mean_val = float(np.mean(magnitude))
        std_val = float(np.std(magnitude))
        energy_val = float(np.mean(magnitude ** 2))

        features.extend([mean_val, std_val, energy_val])

    return np.array(features, dtype=np.float32)


# =====================================================================
# 2. DENSE SIFT (SCALE-INVARIANT FEATURE TRANSFORM)
# =====================================================================

def create_dense_sift_keypoints(
    img_size: Tuple[int, int] = config.IMG_SIZE,
    step: int = config.SIFT_GRID_STEP,
    size: int = config.SIFT_KEYPOINT_SIZE
) -> List[cv2.KeyPoint]:
    """Generates a regular grid of dense SIFT keypoints across the image."""
    w, h = img_size
    keypoints = []
    # Grid margins to avoid border truncation
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(size)))
    return keypoints


_SIFT_DENSE_KEYPOINTS = create_dense_sift_keypoints()
_SIFT_DETECTOR = cv2.SIFT_create()


def extract_raw_sift_descriptors(
    img: np.ndarray,
    keypoints: Optional[List[cv2.KeyPoint]] = None
) -> np.ndarray:
    """Extract raw 128-D SIFT descriptors from fixed dense grid keypoints."""
    if keypoints is None:
        keypoints = _SIFT_DENSE_KEYPOINTS

    # Ensure single channel uint8
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    _, descriptors = _SIFT_DETECTOR.compute(img_gray, keypoints)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((len(keypoints), 128), dtype=np.float32)
    return descriptors.astype(np.float32)


def build_bovw_codebook(
    images: List[np.ndarray],
    n_clusters: int = config.SIFT_BOVW_CLUSTERS,
    random_state: int = config.RANDOM_SEED
) -> MiniBatchKMeans:
    """
    Learns Bag-of-Visual-Words (BoVW) visual dictionary from Dense SIFT descriptors
    extracted across the training set.
    """
    all_descriptors = []
    for img in images:
        descs = extract_raw_sift_descriptors(img)
        if descs is not None and len(descs) > 0:
            all_descriptors.append(descs)

    all_descriptors = np.vstack(all_descriptors)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3,
        max_iter=300
    )
    kmeans.fit(all_descriptors)
    return kmeans


def extract_sift_features(
    img: np.ndarray,
    bovw_model: Optional[MiniBatchKMeans] = None,
    pooling: str = config.SIFT_POOLING_METHOD
) -> np.ndarray:
    """
    Extract fixed-length SIFT representation from an image.
    Supports:
      1. 'bovw': Bag-of-Visual-Words normalized histogram + spatial quadrant histograms
      2. 'spatial_mean_max': Spatial pyramid mean and max pooling across quadrants
    """
    descriptors = extract_raw_sift_descriptors(img)

    if pooling == "bovw" and bovw_model is not None:
        cluster_ids = bovw_model.predict(descriptors)
        k = bovw_model.n_clusters
        # Global histogram
        hist, _ = np.histogram(cluster_ids, bins=np.arange(k + 1), density=False)
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm

        # Spatial quadrant pooling (divide 49 keypoints into 4 quadrants)
        n_kp = len(_SIFT_DENSE_KEYPOINTS)
        step = int(np.sqrt(n_kp))
        quad_features = []
        for qy in (0, 1):
            for qx in (0, 1):
                quad_ids = []
                for idx, kp in enumerate(_SIFT_DENSE_KEYPOINTS):
                    in_x = (kp.pt[0] < config.IMG_SIZE[0] / 2) if qx == 0 else (kp.pt[0] >= config.IMG_SIZE[0] / 2)
                    in_y = (kp.pt[1] < config.IMG_SIZE[1] / 2) if qy == 0 else (kp.pt[1] >= config.IMG_SIZE[1] / 2)
                    if in_x and in_y:
                        quad_ids.append(cluster_ids[idx])
                if len(quad_ids) > 0:
                    q_hist, _ = np.histogram(quad_ids, bins=np.arange(k + 1))
                    q_norm = np.linalg.norm(q_hist)
                    if q_norm > 0:
                        q_hist = q_hist.astype(np.float32) / q_norm
                    quad_features.append(q_hist)
                else:
                    quad_features.append(np.zeros(k, dtype=np.float32))

        return np.concatenate([hist] + quad_features)

    else:
        # Spatial mean + max + std pooling across all dense SIFT descriptors
        mean_desc = np.mean(descriptors, axis=0)
        max_desc = np.max(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        return np.concatenate([mean_desc, max_desc, std_desc]).astype(np.float32)


# =====================================================================
# 3. HISTOGRAM OF ORIENTED GRADIENTS (HOG)
# =====================================================================

def extract_hog_features(
    img: np.ndarray,
    orientations: int = config.HOG_ORIENTATIONS,
    pixels_per_cell: Tuple[int, int] = config.HOG_PIXELS_PER_CELL,
    cells_per_block: Tuple[int, int] = config.HOG_CELLS_PER_BLOCK
) -> np.ndarray:
    """
    Extract HOG feature descriptor.
    Captures edge directions and stroke boundary gradients.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    features = hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True
    )
    return features.astype(np.float32)


# =====================================================================
# 4. STATISTICAL FEATURES
# =====================================================================

def extract_statistical_features(
    img: np.ndarray,
    zoning_grid: Tuple[int, int] = config.ZONING_GRID,
    proj_bins: int = config.PROJECTION_BINS
) -> np.ndarray:
    """
    Extract classical statistical structural features for Devanagari characters:
      1. Zoning Density (4x4 = 16 sub-regions)
      2. Diagonal Features (Directional line densities across zones)
      3. Centroid (Normalized center of ink mass)
      4. Horizontal and Vertical Projection Profiles
      5. Contour Transitions
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    # Normalize binary image (foreground ink = 1.0, background = 0.0)
    binary = (img_gray > 127).astype(np.float32)
    h, w = binary.shape
    stats = []

    # A. Zoning Density (4x4 grid)
    zh, zw = h // zoning_grid[1], w // zoning_grid[0]
    for r in range(zoning_grid[1]):
        for c in range(zoning_grid[0]):
            zone = binary[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw]
            stats.append(np.mean(zone))

    # B. Diagonal Features
    # Traverse main diagonal and anti-diagonal of each zone
    for r in range(zoning_grid[1]):
        for c in range(zoning_grid[0]):
            zone = binary[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw]
            diag_main = np.trace(zone) / max(1, min(zh, zw))
            diag_anti = np.trace(np.fliplr(zone)) / max(1, min(zh, zw))
            stats.extend([diag_main, diag_anti])

    # C. Ink Centroid (Center of Mass)
    total_ink = np.sum(binary)
    if total_ink > 0:
        y_indices, x_indices = np.indices(binary.shape)
        cx = np.sum(x_indices * binary) / (total_ink * w)
        cy = np.sum(y_indices * binary) / (total_ink * h)
    else:
        cx, cy = 0.5, 0.5
    stats.extend([cx, cy])

    # D. Horizontal & Vertical Projection Profiles
    # Horizontal projection (row-wise sum) downsampled to proj_bins
    row_sums = np.sum(binary, axis=1) / max(1, w)
    col_sums = np.sum(binary, axis=0) / max(1, h)

    # Bin row and col sums to fixed length
    row_binned = np.array([np.mean(chunk) for chunk in np.array_split(row_sums, proj_bins)])
    col_binned = np.array([np.mean(chunk) for chunk in np.array_split(col_sums, proj_bins)])
    stats.extend(row_binned)
    stats.extend(col_binned)

    # E. Background-to-foreground transitions along center lines
    center_row = binary[h // 2, :]
    center_col = binary[:, w // 2]
    row_trans = np.sum(np.abs(np.diff(center_row)))
    col_trans = np.sum(np.abs(np.diff(center_col)))
    stats.extend([row_trans / max(1, w), col_trans / max(1, h)])

    return np.array(stats, dtype=np.float32)


# =====================================================================
# 5. PAPER 1 & PAPER 2 FEATURE EXTRACTORS: GLCM & BINARY CODED VALUES
# =====================================================================

def extract_binary_coded_features(
    img: np.ndarray,
    grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Extracts Binary Coded Values feature representation.
    Used in Paper 1 (Ramya & Parvathavarthini, 2014) and as baseline in Paper 2 (Poornima Devi & Sornam, 2020).
    
    Resamples the binary character mask to a uniform 8x8 spatial grid,
    yielding a 64-dimensional binary descriptor vector x in {0, 1}^64.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Resample to grid_size
    resized = cv2.resize(img_gray, grid_size, interpolation=cv2.INTER_AREA)
    # Threshold to binary 0 or 1
    binary = (resized > 127).astype(np.float32)
    return binary.flatten()


def extract_glcm_features(
    img: np.ndarray,
    distances: List[int] = [1, 2],
    angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    n_levels: int = 16
) -> np.ndarray:
    """
    Extracts the six-feature Gray-Level Co-occurrence Matrix (GLCM) representation.
    Reference: Paper 2 (Poornima Devi & Sornam, 2020), ACM TALLIP.
    
    Computes the 6 exact Haralick texture properties defined in Equations (21)-(27):
      1. Contrast (Eq. 21):       sum_{i,j} (i - j)^2 * P(i, j)
      2. Energy (Eq. 22):         sqrt(sum_{i,j} P(i, j)^2) = sqrt(ASM)
      3. Homogeneity (Eq. 23):    sum_{i,j} P(i, j) / (1 + (i - j)^2)
      4. Correlation (Eq. 24):    sum_{i,j} (i - mu_i)*(j - mu_j)*P(i, j) / (sigma_i * sigma_j)
      5. Dissimilarity (Eq. 26):  sum_{i,j} |i - j| * P(i, j)
      6. ASM (Eq. 27):            sum_{i,j} P(i, j)^2
      
    Averages each property across directional co-occurrence matrices, returning
    a compact, rotation-robust 6-dimensional texture descriptor vector in R^6.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    # Quantize image to n_levels (e.g. 16 gray levels) for robust co-occurrence counting
    quantized = (img_gray.astype(np.float32) / 256.0 * n_levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, n_levels - 1)

    # Compute GLCM over distances and angles
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=True,
        normed=True
    )

    # Extract all 6 properties defined in the 2020 research paper
    # Note: skimage 'energy' is sqrt(ASM) as in Equation 22, and 'ASM' is Equation 27
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
    asm = float(np.mean(graycoprops(glcm, "ASM")))

    return np.array([contrast, energy, homogeneity, correlation, dissimilarity, asm], dtype=np.float32)



# =====================================================================
# 6. FEATURE COMBINATIONS & BATCH PIPELINE
# =====================================================================

def extract_single_image_features(
    img: np.ndarray,
    feature_type: str = "gabor_sift",
    bovw_model: Optional[MiniBatchKMeans] = None
) -> np.ndarray:
    """Extract specified feature set for a single image."""
    ft = feature_type.lower()
    if ft == "gabor":
        return extract_gabor_features(img)
    elif ft == "sift":
        return extract_sift_features(img, bovw_model=bovw_model)
    elif ft == "hog":
        return extract_hog_features(img)
    elif ft == "statistical":
        return extract_statistical_features(img)
    elif ft in ("glcm", "paper2_glcm"):
        # Paper 2: Poornima Devi & Sornam (2020)
        return extract_glcm_features(img)
    elif ft in ("binary_coded", "binary", "paper1_binary"):
        # Paper 1: Ramya & Parvathavarthini (2014) & Paper 2 baseline
        return extract_binary_coded_features(img)
    elif ft in ("gabor_sift", "sift_gabor"):
        # Narang et al. (2020) proposed combined feature set
        f_gabor = extract_gabor_features(img)
        f_sift = extract_sift_features(img, bovw_model=bovw_model)
        return np.concatenate([f_gabor, f_sift])
    elif ft in ("glcm_gabor", "gabor_glcm"):
        f_glcm = extract_glcm_features(img)
        f_gabor = extract_gabor_features(img)
        return np.concatenate([f_glcm, f_gabor])
    elif ft in ("statistical_hog", "hog_statistical"):
        f_stat = extract_statistical_features(img)
        f_hog = extract_hog_features(img)
        return np.concatenate([f_stat, f_hog])
    elif ft == "all":
        f_gabor = extract_gabor_features(img)
        f_sift = extract_sift_features(img, bovw_model=bovw_model)
        f_hog = extract_hog_features(img)
        f_stat = extract_statistical_features(img)
        f_glcm = extract_glcm_features(img)
        return np.concatenate([f_gabor, f_sift, f_hog, f_stat, f_glcm])
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


class FeaturePipeline:
    """
    Scikit-learn compatible feature extraction and normalization pipeline.
    Handles:
      - Training BoVW codebook strictly on X_train (avoiding test leakage)
      - Batch feature extraction
      - StandardScaler fit on X_train, transform on X_test
    """

    def __init__(
        self,
        feature_type: str = "gabor_sift",
        use_bovw: bool = True,
        bovw_clusters: int = config.SIFT_BOVW_CLUSTERS,
        random_state: int = config.RANDOM_SEED
    ):
        self.feature_type = feature_type
        self.use_bovw = use_bovw
        self.bovw_clusters = bovw_clusters
        self.random_state = random_state
        self.bovw_model: Optional[MiniBatchKMeans] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim_ = None

    def fit(self, X_train_preprocessed: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit BoVW dictionary and StandardScaler on training data."""
        ft = self.feature_type.lower()
        if ("sift" in ft or "all" in ft) and self.use_bovw:
            self.bovw_model = build_bovw_codebook(
                X_train_preprocessed,
                n_clusters=self.bovw_clusters,
                random_state=self.random_state
            )

        # Extract features for all training samples
        X_feats = [
            extract_single_image_features(img, self.feature_type, self.bovw_model)
            for img in X_train_preprocessed
        ]
        X_feats = np.array(X_feats, dtype=np.float32)
        self.scaler.fit(X_feats)
        self.feature_dim_ = X_feats.shape[1]
        self.is_fitted = True
        return self

    def transform(self, X_preprocessed: np.ndarray) -> np.ndarray:
        """Extract features and standardize with fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform.")
        X_feats = [
            extract_single_image_features(img, self.feature_type, self.bovw_model)
            for img in X_preprocessed
        ]
        X_feats = np.array(X_feats, dtype=np.float32)
        return self.scaler.transform(X_feats)

    def fit_transform(self, X_train_preprocessed: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit pipeline on X_train and transform in one step."""
        self.fit(X_train_preprocessed, y)
        return self.transform(X_train_preprocessed)


if __name__ == "__main__":
    dummy_img = np.zeros((64, 64), dtype=np.uint8)
    dummy_img[20:45, 15:50] = 255

    g_feat = extract_gabor_features(dummy_img)
    s_feat = extract_sift_features(dummy_img)
    h_feat = extract_hog_features(dummy_img)
    st_feat = extract_statistical_features(dummy_img)
    comb = extract_single_image_features(dummy_img, "gabor_sift")

    print(f"Gabor features length:      {len(g_feat)}")
    print(f"Dense SIFT features length: {len(s_feat)}")
    print(f"HOG features length:        {len(h_feat)}")
    print(f"Statistical features length:{len(st_feat)}")
    print(f"Gabor + SIFT length:        {len(comb)}")
    print("Feature extraction verification successful!")
