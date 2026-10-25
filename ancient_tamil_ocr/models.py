"""
Classifier models, hyperparameter tuning, and 10-fold Cross-Validation for Ancient Tamil Palm-Leaf OCR.
Strictly Classical Machine Learning & Shallow Neural Networks (Zero Deep Learning).

Models:
  - SVM (Linear, RBF, Polynomial kernels)
  - k-Nearest Neighbors (k-NN)
  - Random Forest
  - MLPClassifier (Classical Shallow Neural Network, 1-2 hidden layers)
"""

from typing import Dict, Any, Tuple, Optional
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from neural_networks import FeedForwardBackpropNN, ModifiedAdaptiveBackpropNN


def get_classifiers(random_state: int = config.RANDOM_SEED) -> Dict[str, Any]:
    """
    Instantiates the neural network architectures and classical machine learning classifiers.
    Neural networks are placed first as the primary research architectures:
      - MABPN (Paper 2: Poornima Devi & Sornam, 2020) - Primary Research Model
      - FFBNN (Paper 1: Ramya & Parvathavarthini, 2014) - Adapted Research Model
    followed by classical comparative baseline classifiers (SVM, RF, k-NN, Classical MLP).
    """
    return {
        "MABPN (Paper 2: 2020)": ModifiedAdaptiveBackpropNN(
            hidden_layer_sizes=(64,),
            initial_learning_rate=0.05,
            activation="shannon",
            max_iter=300,
            random_state=random_state
        ),
        "FFBNN (Paper 1: 2014)": FeedForwardBackpropNN(
            hidden_layer_sizes=(64,),
            learning_rate=0.05,
            momentum=0.85,
            max_iter=300,
            random_state=random_state
        ),
        "SVM-Poly": SVC(
            kernel="poly",
            degree=3,
            C=1.0,
            coef0=1.0,
            probability=True,
            random_state=random_state
        ),
        "SVM-RBF": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=random_state
        ),
        "SVM-Linear": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=random_state
        ),
        "k-NN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="euclidean"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=1
        ),
        "MLP (Shallow NN)": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="lbfgs",
            alpha=0.001,
            max_iter=500,
            random_state=random_state
        ),
        "MLP-2Layer": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="lbfgs",
            alpha=0.001,
            max_iter=500,
            random_state=random_state
        ),
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """Defines search spaces for GridSearchCV hyperparameter optimization."""
    return {
        "SVM-Poly": {
            "C": [0.1, 1.0, 10.0],
            "degree": [2, 3],
            "coef0": [0.0, 1.0],
        },
        "SVM-RBF": {
            "C": [0.5, 1.0, 10.0, 50.0],
            "gamma": ["scale", "auto", 0.01],
        },
        "SVM-Linear": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "k-NN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 12, 20],
        },
        "MLP (Shallow NN)": {
            "hidden_layer_sizes": [(100,), (128, 64)],
            "alpha": [0.0001, 0.001],
        },
    }


def tune_hyperparameters(
    model_name: str,
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_pca: bool = False,
    pca_ratio: float = config.PCA_VARIANCE_RATIO,
    cv_folds: int = 5,
    random_state: int = config.RANDOM_SEED
) -> Tuple[Any, Dict[str, Any]]:
    """
    Performs grid search cross-validation to discover optimal hyperparameters.
    Eliminates CV leakage: If use_pca is True, PCA is wrapped in an sklearn Pipeline
    so PCA is fitted strictly on training folds within each cross-validation split.
    """
    grids = get_hyperparameter_grids()
    param_grid = grids.get(model_name, {})

    if not param_grid:
        if use_pca:
            pca = PCA(n_components=pca_ratio, random_state=random_state)
            X_train = pca.fit_transform(X_train)
        base_model.fit(X_train, y_train)
        return base_model, {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if use_pca:
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("pca", PCA(n_components=pca_ratio, random_state=random_state)),
            ("clf", base_model)
        ])
        pipe_param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=pipe_param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_.named_steps["clf"]
        best_params = {k.replace("clf__", ""): v for k, v in grid_search.best_params_.items()}
        return best_model, best_params
    else:
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_


def run_tenfold_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    use_pca: bool = False,
    pca_ratio: float = config.PCA_VARIANCE_RATIO,
    cv_folds: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_SEED
) -> Dict[str, Any]:
    """
    Executes Stratified 10-Fold Cross-Validation.
    Eliminates cross-validation leakage: When use_pca=True, PCA is fitted strictly
    on the current fold's training samples (X_tr_fold) and transforms X_val_fold.
    Validation fold samples never leak into PCA decomposition.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_fold, X_val_fold = X[train_idx], X[val_idx]
        y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

        if use_pca:
            # Fit PCA strictly on the training fold to eliminate CV data leakage
            pca_fold = PCA(n_components=pca_ratio, random_state=random_state)
            X_tr_fold = pca_fold.fit_transform(X_tr_fold)
            X_val_fold = pca_fold.transform(X_val_fold)

        # Fit clone/fresh model instance on current fold
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_tr_fold, y_tr_fold)

        preds = fold_model.predict(X_val_fold)

        acc = accuracy_score(y_val_fold, preds)
        prec = precision_score(y_val_fold, preds, average="macro", zero_division=0)
        rec = recall_score(y_val_fold, preds, average="macro", zero_division=0)
        f1 = f1_score(y_val_fold, preds, average="macro", zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_precision": float(np.mean(fold_precisions)),
        "std_precision": float(np.std(fold_precisions)),
        "mean_recall": float(np.mean(fold_recalls)),
        "std_recall": float(np.std(fold_recalls)),
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "fold_accuracies": [float(x) for x in fold_accuracies],
        "fold_f1s": [float(x) for x in fold_f1s],
    }



if __name__ == "__main__":
    clfs = get_classifiers()
    print(f"Instantiated {len(clfs)} classifiers: {list(clfs.keys())}")
