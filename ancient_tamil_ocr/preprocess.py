"""
Preprocessing module for Ancient Tamil Palm-Leaf Character Images.
Domain: Ancient Tamil Palm-Leaf Manuscripts (Panaiyolai Suvadigal).
Pipeline: Grayscale -> Resize (64x64) -> Denoise -> CLAHE -> Binarization -> Morphological Cleanup.
"""

from pathlib import Path
from typing import Union, Tuple, Dict, List
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola, threshold_otsu

import config


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Ensure image is single-channel grayscale uint8."""
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.squeeze().astype(np.uint8)


def resize_image(img: np.ndarray, target_size: Tuple[int, int] = config.IMG_SIZE) -> np.ndarray:
    """Resize image to target dimensions (width, height) using INTER_AREA."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def denoise_image(
    img: np.ndarray,
    method: str = config.DENOISE_METHOD
) -> np.ndarray:
    """
    Apply spatial filtering to reduce fiber grain and scanning artifacts.
    Supports Gaussian smoothing and Median filtering.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(
            img,
            config.DENOISE_GAUSSIAN_KERNEL,
            config.DENOISE_GAUSSIAN_SIGMA
        )
    elif method == "median":
        return cv2.medianBlur(img, config.DENOISE_MEDIAN_KSIZE)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    return img


def enhance_contrast_clahe(
    img: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    grid_size: Tuple[int, int] = config.CLAHE_TILE_GRID_SIZE
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Essential for palm-leaf manuscripts with uneven illumination and aged leaf patina.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def binarize_palm_leaf(
    img: np.ndarray,
    method: str = config.BINARIZE_METHOD
) -> np.ndarray:
    """
    Binarize degraded palm-leaf character images.
    Returns: Binary image with foreground ink as 255 and background as 0.
    
    Sauvola adaptive thresholding is exceptionally effective for historical manuscripts
    as it adapts to local mean and standard deviation:
      T = m * (1 + k * ((s / R) - 1))
    """
    if method == "sauvola":
        # Calculate local Sauvola threshold map
        # Sauvola operates on float [0, 1]
        img_norm = img.astype(np.float32) / 255.0
        thresh = threshold_sauvola(
            img_norm,
            window_size=config.SAUVOLA_WINDOW_SIZE,
            k=config.SAUVOLA_K
        )
        binary_mask = img_norm < thresh  # Ink is darker than background
        return (binary_mask * 255).astype(np.uint8)

    elif method == "otsu":
        # Global Otsu with inversion (dark ink -> 255)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    elif method == "adaptive_gaussian":
        # OpenCV Adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 4
        )
        return binary

    else:
        raise ValueError(f"Unknown binarization method: {method}")


def morphological_cleanup(
    binary_img: np.ndarray,
    kernel_size: Tuple[int, int] = config.MORPH_KERNEL_SIZE
) -> np.ndarray:
    """
    Morphological operations to clean degraded character binary images:
      - Morphological Open: Removes salt/fiber speckle noise from background
      - Morphological Close: Connects hairline fractures in weathered ink strokes
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Open: Erode then dilate (eliminates isolated white speckles)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    # Close: Dilate then erode (bridges hairline breaks in ancient strokes)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = config.IMG_SIZE,
    denoise_method: str = config.DENOISE_METHOD,
    binarize_method: str = config.BINARIZE_METHOD,
    apply_clahe: bool = True,
    apply_morph: bool = config.APPLY_MORPH_CLEANUP
) -> np.ndarray:
    """
    Full end-to-end preprocessing pipeline for a single character sample.
    Converts raw image into a standardized, clean 64x64 binary representation.
    """
    # 1. Load image if path provided
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_input}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()

    # 2. Grayscale conversion
    gray = to_grayscale(img)

    # 3. Resize to 64x64
    resized = resize_image(gray, target_size)

    # 4. Denoise
    denoised = denoise_image(resized, method=denoise_method)

    # 5. Contrast Enhancement (CLAHE)
    enhanced = enhance_contrast_clahe(denoised) if apply_clahe else denoised

    # 6. Degraded Manuscript Binarization
    binary = binarize_palm_leaf(enhanced, method=binarize_method)

    # 7. Morphological Cleanup
    cleaned = morphological_cleanup(binary) if apply_morph else binary

    return cleaned


def get_pipeline_stages(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = config.IMG_SIZE
) -> Dict[str, np.ndarray]:
    """
    Extracts all intermediate stages of the preprocessing pipeline for analysis and visualization.
    """
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.asarray(image_input, dtype=np.uint8).copy()

    gray = to_grayscale(img_rgb)
    resized_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    resized_gray = resize_image(gray, target_size)
    denoised = denoise_image(resized_gray, method=config.DENOISE_METHOD)
    clahe_enh = enhance_contrast_clahe(denoised)
    binarized = binarize_palm_leaf(clahe_enh, method=config.BINARIZE_METHOD)
    cleaned = morphological_cleanup(binarized)

    return {
        "Original": resized_rgb,
        "Grayscale": resized_gray,
        "Denoised": denoised,
        "CLAHE Enhanced": clahe_enh,
        "Sauvola Binarized": binarized,
        "Morph Cleaned": cleaned,
    }


def preprocess_dataset(
    images_raw: Union[List[np.ndarray], np.ndarray],
    target_size: Tuple[int, int] = config.IMG_SIZE,
    binarize_method: str = config.BINARIZE_METHOD
) -> np.ndarray:
    """Preprocess a batch of raw character images."""
    preprocessed = [
        preprocess_image(img, target_size=target_size, binarize_method=binarize_method)
        for img in images_raw
    ]
    return np.array(preprocessed, dtype=np.uint8)


def visualize_preprocessing_steps(
    sample_images: List[np.ndarray],
    class_labels: List[str],
    output_path: Union[str, Path] = None,
    max_samples: int = 5
):
    """
    Generates and saves a publication-quality figure showing the original palm-leaf
    degraded samples versus each progressive stage of preprocessing.
    """
    if output_path is None:
        output_path = config.RESULTS_DIR / "original_vs_preprocessed.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = min(len(sample_images), max_samples)
    stage_names = [
        "1. Original Palm-Leaf",
        "2. Grayscale (64x64)",
        "3. CLAHE Enhanced",
        "4. Sauvola Binarized",
        "5. Morph Cleaned",
    ]

    fig, axes = plt.subplots(n_samples, len(stage_names), figsize=(14, 2.7 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        stages = get_pipeline_stages(sample_images[i])
        stage_keys = [
            "Original",
            "Grayscale",
            "CLAHE Enhanced",
            "Sauvola Binarized",
            "Morph Cleaned",
        ]

        for j, (key, title) in enumerate(zip(stage_keys, stage_names)):
            ax = axes[i, j]
            img = stages[key]

            if j == 0:
                ax.imshow(img)
                ax.set_ylabel(
                    f"Class: {class_labels[i]}",
                    fontsize=12,
                    fontweight="bold",
                    color="#2c3e50"
                )
            else:
                ax.imshow(img, cmap="gray")

            if i == 0:
                ax.set_title(title, fontsize=12, fontweight="semibold", pad=8)

            ax.set_xticks([])
            ax.set_yticks([])
            # Subtle border around each subplot
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(1.0)

    fig.suptitle(
        "Ancient Tamil Palm-Leaf Preprocessing Pipeline\n"
        "(Grayscale $\\rightarrow$ Resizing $\\rightarrow$ Denoising $\\rightarrow$ CLAHE $\\rightarrow$ Sauvola Binarization $\\rightarrow$ Morphological Clean)",
        fontsize=14,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Preprocessing pipeline visualization saved to: {output_path.resolve()}")


# =============================================================================
# CHARACTER SEGMENTATION ALGORITHMS (PAPER 1 & PAPER 2)
# =============================================================================

def segment_characters_sliding_window(
    binary_img: np.ndarray,
    window_size: Tuple[int, int] = (64, 64),
    step_size: int = 16,
    min_ink_density: float = 0.05
) -> List[Tuple[int, int, int, int]]:
    """
    Sliding Window Character Segmentation.
    Reference: Paper 1 (Ramya & Parvathavarthini, 2014), Journal of Computer Science.
    
    Slides a fixed-size inspection window across the palm-leaf manuscript line,
    measuring foreground ink pixel density to isolate individual character frames.
    
    Returns:
      List of bounding boxes (x, y, w, h) for identified character candidates.
    """
    h, w = binary_img.shape[:2]
    win_w, win_h = window_size
    candidates = []

    # Ensure binary mask has foreground as 255
    foreground = (binary_img > 127).astype(np.float32)

    for y in range(0, max(1, h - win_h + 1), max(1, win_h // 2)):
        for x in range(0, max(1, w - win_w + 1), step_size):
            patch = foreground[y:y + win_h, x:x + win_w]
            ink_density = float(np.mean(patch))
            if ink_density >= min_ink_density:
                candidates.append((x, y, win_w, win_h))

    return candidates


def segment_characters_convex_hull(
    binary_img: np.ndarray,
    min_dim: int = 14,
    max_dim: int = 120
) -> List[Tuple[int, int, int, int]]:
    """
    Contour-Based Convex Hull Bounding Box Character Segmentation.
    Reference: Paper 2 (Poornima Devi & Sornam, 2020), ACM TALLIP.
    
    Computes external contours from the restored manuscript, calculates
    the convex hull for each isolated character contour, and encloses
    irregular ancient curved glyphs within a tight convex hull bounding box.
    
    Returns:
      List of bounding boxes (x, y, w, h) sorted left-to-right along the leaf.
    """
    if binary_img.ndim == 3:
        binary_mask = cv2.cvtColor(binary_img, cv2.COLOR_RGB2GRAY)
    else:
        binary_mask = binary_img

    # Find external contours of ink strokes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Compute Convex Hull around the contour (Paper 2)
        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(hull)
        
        # Filter out tiny noise specks or excessively large page artifacts
        if min_dim <= w <= max_dim and min_dim <= h <= max_dim:
            boxes.append((x, y, w, h))

    # Sort left-to-right along the manuscript strip
    boxes = sorted(boxes, key=lambda b: (b[0] // 50, b[0]))
    return boxes


if __name__ == "__main__":
    import dataset
    X_tr, _, _, _, class_names, _ = dataset.load_character_dataset()
    sample_imgs = [X_tr[i * 40] for i in range(min(5, len(class_names)))]
    sample_labels = [class_names[i] for i in range(min(5, len(class_names)))]
    visualize_preprocessing_steps(sample_imgs, sample_labels)
    print("Preprocessing verification successful!")
