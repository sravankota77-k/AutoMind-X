"""Synthetic dataset generation (scikit-learn) with train / validation / test splits."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    """Standardised train / validation / test arrays plus the problem dimensions."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int


def make_dataset(cfg) -> Dataset:
    """Generate synthetic data and split it 60/20/20 (stratified). The scaler is fit on train only: no leakage."""
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_classes=cfg.n_classes,
        class_sep=cfg.class_sep,
        flip_y=cfg.flip_y,
        random_state=cfg.seed,
    )
    # 60% train (fit weights) / 20% validation (evolution sees this) / 20% test (final honest check)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=cfg.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=cfg.seed
    )
    scaler = StandardScaler().fit(X_train)  # fit on train only -> no leakage
    return Dataset(
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        y_train,
        y_val,
        y_test,
        cfg.n_features,
        cfg.n_classes,
    )
