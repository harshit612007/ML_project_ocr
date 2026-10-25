"""
Model registry for Ancient Tamil Palm-Leaf Manuscript Character Recognition.
Reference: Poornima Devi. M and M. Sornam (2020), ACM TALLIP, Article 82.

Provides strictly the two proposed models from the reference paper:
1. Model 1: MABPN-BCV (Binary Coded Value input)
   - Architecture: 65 inputs (64 pixels + 1 bias), 55 hidden, 5 output neurons.
2. Model 2: MABPN-GLCM (GLCM 6-feature input)
   - Architecture: 6 inputs (+ 1 bias), 20 hidden, 5 output neurons.
"""

from typing import Dict, Any
import config
from neural_networks import MABPN_BCV, MABPN_GLCM


def get_paper_models(random_state: int = config.RANDOM_SEED) -> Dict[str, Any]:
    """
    Returns the two proposed neural network models described in the reference paper:
      - 'MABPN-BCV': Model 1 (Section 3.1.2)
      - 'MABPN-GLCM': Model 2 (Section 3.1.3)
    """
    return {
        "MABPN-BCV": MABPN_BCV(
            learning_rate=0.05,
            max_iter=130,  # 130 epochs reported in Table 5 for MABPN-BCV
            tol=config.CONVERGENCE_ERROR_TARGET,
            random_state=random_state
        ),
        "MABPN-GLCM": MABPN_GLCM(
            learning_rate=0.05,
            max_iter=104,  # 104 epochs reported in Table 5 for MABPN-GLCM
            tol=config.CONVERGENCE_ERROR_TARGET,
            random_state=random_state
        ),
    }


def get_classifiers(random_state: int = config.RANDOM_SEED) -> Dict[str, Any]:
    """Alias for backwards compatibility with pipeline callers."""
    return get_paper_models(random_state=random_state)
