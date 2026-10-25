"""
Dataset loader and verification module for Ancient Tamil Palm-Leaf Character Recognition.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Paper Experimental Dataset Setup (Section 4.1.1):
  - Primary Source: U.V. Swaminatha Iyer Library, Chennai, Tamil Nadu.
  - Historical Manuscript: 'Kuzhanthai Pini Marunthu' (Medical Palm-Leaf Manuscript, 175 leaves).
  - Character Set: 18 Meyyezhuthu (Body-Letters / Consonants) classes (Figure 12).
  - Total Characters: 1,756 characters.
  - Training Set: 1,229 characters (70%).
  - Testing Set: 527 characters (30%).
  
Note on Dataset Availability:
  The authors note (Page 82:5): "The palm leaf dataset used in this work is a primary dataset
  that is not available online."
  If the authentic library dataset is not present in data/, this module reports that status
  transparently and provides a controlled 18-class test fixture generator solely for verifying
  code correctness, input/output tensor dimensions, and pipeline execution.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_dataset_status(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Checks whether the primary 18-class palm-leaf dataset from U.V. Swaminatha Iyer Library is present.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    data_dir = Path(data_dir)

    is_present = False
    class_folders = []
    total_images = 0

    if data_dir.exists():
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        class_folders = sorted([d.name for d in subdirs])
        for sf in subdirs:
            total_images += len(list(sf.glob("*.png")) + list(sf.glob("*.jpg")) + list(sf.glob("*.jpeg")))

        # Check if matching the 18 classes and 1,756 samples
        if len(class_folders) == 18 and total_images >= 1700:
            is_present = True

    return {
        "is_authentic_primary_dataset": is_present,
        "data_dir": str(data_dir),
        "num_classes_found": len(class_folders),
        "classes_found": class_folders,
        "total_images_found": total_images,
        "expected_classes": len(config.TAMIL_CLASSES_18),
        "expected_total_images": 1756,
        "reference_source": "U.V. Swaminatha Iyer Library, Chennai (Kuzhanthai Pini Marunthu)",
    }


def find_system_tamil_font() -> Optional[str]:
    """Finds available system font supporting Tamil script for the test fixture generator."""
    candidate_fonts = [
        r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\NirmalaB.ttc",
        r"C:\Windows\Fonts\latha.ttf",
        r"C:\Windows\Fonts\vijaya.ttf",
        r"C:\Windows\Fonts\arialuni.ttf",
        "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidate_fonts:
        if os.path.exists(font_path):
            return font_path
    return None


def generate_synthetic_test_fixture(
    output_dir: Path,
    samples_per_class: int = 20,
    random_seed: int = config.RANDOM_SEED
) -> Path:
    """
    Generates a controlled synthetic test fixture of the 18 Tamil consonant classes
    solely for unit testing, dimension verification, and pipeline smoke testing.
    
    IMPORTANT: This is explicitly labeled as a synthetic test fixture and NOT
    the authentic 'Kuzhanthai Pini Marunthu' manuscript dataset from the library.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    font_path = find_system_tamil_font()
    pil_font = None
    if font_path:
        try:
            pil_font = ImageFont.truetype(font_path, 28)
        except Exception:
            pil_font = None

    for cls in config.TAMIL_CLASSES_18:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        glyph = config.TAMIL_GLYPH_MAP_18.get(cls, cls)

        for i in range(samples_per_class):
            # Create simulated palm-leaf character patch (8x8 final size)
            canvas = Image.new("L", (32, 32), color=220)
            draw = ImageDraw.Draw(canvas)
            if pil_font:
                draw.text((4, 2), glyph, fill=30, font=pil_font)
            else:
                # Fallback: draw basic geometric stroke pattern
                draw.ellipse([6, 6, 26, 26], outline=30, width=2)
                draw.line([6, 16, 26, 16], fill=30, width=2)

            patch = np.array(canvas, dtype=np.uint8)
            # Add minor noise
            noise = rng.normal(0, 5, patch.shape).astype(np.int16)
            patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Resize to paper 8x8 size
            char_8x8 = cv2.resize(patch, config.CHAR_SIZE, interpolation=cv2.INTER_AREA)

            img_file = cls_dir / f"{cls}_test_{i+1:03d}.png"
            Image.fromarray(char_8x8).save(img_file)

    logger.info(f"Generated 18-class test fixture ({samples_per_class} samples/class) in: {output_dir}")
    return output_dir


def audit_dataset(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Audits every class folder in data_dir against the 18 reference classes in Figure 12.
    Detects valid Meyyezhuthu consonants vs rejected unknown/vowel classes.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    data_dir = Path(data_dir)

    all_subdirs = [d for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []
    valid_classes = []
    rejected_classes = []
    class_details = {}

    for d in all_subdirs:
        if d.name == "18_classes_test_fixture":
            continue
        try:
            canonical = config.normalize_label(d.name)
            valid_classes.append(d.name)
            class_details[d.name] = {
                "canonical": canonical,
                "status": "VALID_FIGURE_12_CONSONANT",
                "code": config.get_5bit_code(canonical).tolist()
            }
        except ValueError as err:
            rejected_classes.append(d.name)
            class_details[d.name] = {
                "canonical": None,
                "status": "REJECTED_NON_FIGURE_12",
                "reason": str(err)
            }

    status = check_dataset_status(data_dir)
    status["valid_classes"] = valid_classes
    status["rejected_classes"] = rejected_classes
    status["class_details"] = class_details
    return status


def load_character_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    test_size: float = config.TRAIN_TEST_SPLIT_RATIO,
    random_state: int = config.RANDOM_SEED,
    allow_test_fixture: bool = True,
    strict_figure12: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Loads character images for Model 1 (MABPN-BCV) and Model 2 (MABPN-GLCM).
    Strictly verifies class labels against the 18 Meyyezhuthu classes in Figure 12.
    
    Returns:
      X_train_raw: Training character images (shape: N_train x 8 x 8)
      X_test_raw:  Testing character images (shape: N_test x 8 x 8)
      y_train:     Training class labels
      y_test:      Testing class labels
      class_names: List of class names
      metadata:    Status dictionary detailing dataset provenance
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    data_dir = Path(data_dir)

    status = check_dataset_status(data_dir)
    target_dir = data_dir

    if not status["is_authentic_primary_dataset"]:
        print("\n" + "=" * 76)
        print(" [DATASET PROVENANCE NOTICE]")
        print(" The primary palm-leaf dataset described in the reference paper:")
        print("   - Source: U.V. Swaminatha Iyer Library, Chennai, Tamil Nadu")
        print("   - Manuscript: 'Kuzhanthai Pini Marunthu' (175 leaves, 18 classes, 1,756 characters)")
        print("   - Split: 70% Train (1,229), 30% Test (527)")
        print(" is a physical archival collection that was NOT released online by the authors.")
        print(f" Current repository folder '{data_dir.resolve()}' does not contain the 1,756 manuscript files.")
        if allow_test_fixture:
            print(" >> Using the controlled 18-class test fixture (Figure 12) for pipeline & math verification.")
            print(" >> NOTE: Results on test fixture will NOT match the paper's reported 96.23% manuscript accuracy.")
        print("=" * 76 + "\n")

        # If data_dir has the legacy 10-folder dataset (which contains vowels '01_a', etc.)
        # and allow_test_fixture is True, switch to the 18-class test fixture
        fixture_dir = data_dir / "18_classes_test_fixture"
        if allow_test_fixture:
            if not fixture_dir.exists() or len([d for d in fixture_dir.iterdir() if d.is_dir()]) < 18:
                generate_synthetic_test_fixture(fixture_dir, samples_per_class=30, random_seed=random_state)
            target_dir = fixture_dir

    subdirs = [d for d in target_dir.iterdir() if d.is_dir() and d.name != "18_classes_test_fixture"]

    # Filter and validate classes
    valid_subdirs = []
    for d in subdirs:
        try:
            config.normalize_label(d.name)
            valid_subdirs.append(d)
        except ValueError as err:
            if strict_figure12:
                logger.warning(f"Rejecting unknown class folder '{d.name}': {err}")
            else:
                valid_subdirs.append(d)

    if len(valid_subdirs) == 0:
        raise FileNotFoundError(f"No valid Figure 12 character classes found in '{target_dir}'.")

    ordered_figure12_names = [info["name"] for info in config.FIGURE_12_TABLE.values()]
    subdirs_by_canonical = {}
    for d in valid_subdirs:
        c_name = config.normalize_label(d.name)
        subdirs_by_canonical[c_name] = d

    # Order class_names according to Figure 12 table
    class_names = [c for c in ordered_figure12_names if c in subdirs_by_canonical]

    images = []
    labels = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for cls_name in class_names:
        cls_path = subdirs_by_canonical[cls_name]
        files = [p for p in cls_path.iterdir() if p.suffix.lower() in valid_exts]
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != config.CHAR_SIZE:
                    img = cv2.resize(img, config.CHAR_SIZE, interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(cls_name)

    if len(images) == 0:
        raise FileNotFoundError(f"No valid character images found in '{target_dir}'.")

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)

    # 70:30 Stratified Split as in the reference paper (Section 4.1, Table 4)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(f"Loaded dataset from: {target_dir.name}")
    logger.info(f"Total samples: {len(X)} across {len(class_names)} canonical classes.")
    logger.info(f"Stratified Split (70:30): {len(X_train)} Train, {len(X_test)} Test.")

    status["loaded_from"] = str(target_dir)
    status["num_classes_loaded"] = len(class_names)
    status["total_images_loaded"] = len(X)

    return X_train, X_test, y_train, y_test, class_names, status
