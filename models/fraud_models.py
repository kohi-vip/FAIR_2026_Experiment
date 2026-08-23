"""Model factories for the credit-card fraud detection experiments.

The module keeps model construction separate from data preparation, training,
and evaluation so the same configured estimators can be reused consistently.
"""

from __future__ import annotations

from typing import Any

import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from pytorch_tabnet.classifier import TabNetClassifier
except ModuleNotFoundError as exc:
    # Keep the traditional model factory usable when only TabNet is missing.
    if exc.name and exc.name.startswith("pytorch_tabnet"):
        TabNetClassifier = None  # type: ignore[assignment,misc]
    else:
        raise


def get_traditional_models(random_state: int = 42) -> dict[str, Any]:
    """Return the 12 initialized traditional classifiers used in the study.

    The insertion order reflects the four research categories: linear and
    statistical models, a single decision tree, ensemble models, then distance
    and probability models.

    Args:
        random_state: Seed used by classifiers that support reproducibility.

    Returns:
        A mapping from stable model names to initialized classifier instances.
    """
    return {
        # Group 1: Linear & Statistical Models
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "LDA": LinearDiscriminantAnalysis(),
        # Group 2: Single Decision Tree
        "Decision_Tree": DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
        ),
        # Group 3: Ensemble Learning - Bagging
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        # Group 3: Ensemble Learning - Boosting
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Gradient_Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            random_state=random_state,
            verbose=0,
        ),
        # Group 4: Distance & Probability
        "Naive_Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def get_tabnet_model(random_state: int = 42) -> Any:
    """Return a TabNet classifier configured with the paper's parameters.

    Args:
        random_state: Seed passed to TabNet for reproducible initialization.

    Raises:
        ModuleNotFoundError: If ``pytorch-tabnet`` is not installed.

    Returns:
        An initialized :class:`TabNetClassifier`.
    """
    if TabNetClassifier is None:
        raise ModuleNotFoundError(
            "pytorch_tabnet is not installed. Run: pip install pytorch-tabnet"
        )

    return TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="sparsemax",
        seed=random_state,
    )
