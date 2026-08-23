"""Verify model dependencies and constructor compatibility.

Run this module directly with ``python verify_imports.py``. It only initializes
the estimators; it does not load data or train any model.
"""

from __future__ import annotations

try:
    from fraud_models import get_tabnet_model, get_traditional_models
except ModuleNotFoundError as exc:
    # Preserve a readable checklist even when a required package is absent.
    IMPORT_ERROR: ModuleNotFoundError | None = exc
    get_tabnet_model = None  # type: ignore[assignment]
    get_traditional_models = None  # type: ignore[assignment]
else:
    IMPORT_ERROR = None


MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "Group 1 - Linear & Statistical Models": (
        "Logistic_Regression",
        "LDA",
    ),
    "Group 2 - Single Decision Tree": ("Decision_Tree",),
    "Group 3 - Ensemble Learning": (
        "Random_Forest",
        "Extra_Trees",
        "AdaBoost",
        "Gradient_Boosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ),
    "Group 4 - Distance & Probability": (
        "Naive_Bayes",
        "KNN",
    ),
}


def verify_traditional_models() -> dict[str, bool]:
    """Initialize all traditional models and return group-level results."""
    group_status = {group_name: True for group_name in MODEL_GROUPS}

    if get_traditional_models is None:
        missing_package = IMPORT_ERROR.name if IMPORT_ERROR else "unknown"
        print(f"[FAILED] Missing required package: {missing_package}")
        return {group_name: False for group_name in MODEL_GROUPS}

    try:
        models = get_traditional_models()
    except Exception as exc:  # Report dependency/constructor issues cleanly.
        print(f"[FAILED] Could not initialize traditional models: {exc}")
        return {group_name: False for group_name in MODEL_GROUPS}

    expected_names = {
        model_name
        for group_model_names in MODEL_GROUPS.values()
        for model_name in group_model_names
    }
    unexpected_names = set(models).difference(expected_names)
    if unexpected_names:
        print(f"[FAILED] Unexpected model(s): {sorted(unexpected_names)}")

    for group_name, model_names in MODEL_GROUPS.items():
        print(f"\n{group_name}")
        for model_name in model_names:
            model = models.get(model_name)
            is_ready = model is not None
            group_status[group_name] &= is_ready
            status = "READY" if is_ready else "FAILED"
            class_name = type(model).__name__ if is_ready else "not initialized"
            print(f"  [{status}] {model_name}: {class_name}")

    return group_status


def verify_tabnet() -> bool:
    """Initialize TabNet and provide an actionable warning if it is missing."""
    print("\nTabNet - Deep Learning")
    if get_tabnet_model is None:
        missing_package = IMPORT_ERROR.name if IMPORT_ERROR else "unknown"
        if missing_package.startswith("pytorch_tabnet"):
            print(
                "  [FAILED] pytorch_tabnet is missing. "
                "Please run: pip install pytorch-tabnet"
            )
        else:
            print(
                "  [FAILED] fraud_models could not be imported because "
                f"{missing_package} is missing."
            )
        return False

    try:
        model = get_tabnet_model()
    except ModuleNotFoundError:
        print(
            "  [FAILED] pytorch_tabnet is missing. "
            "Please run: pip install pytorch-tabnet"
        )
        return False
    except Exception as exc:
        print(f"  [FAILED] TabNet initialization error: {exc}")
        return False

    print(f"  [READY] TabNet: {type(model).__name__}")
    return True


def print_checklist(group_status: dict[str, bool], tabnet_ready: bool) -> None:
    """Print the final structured readiness checklist."""
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION CHECKLIST")
    print("=" * 60)
    for group_name, is_ready in group_status.items():
        status = "READY" if is_ready else "FAILED"
        print(f"[{status:^6}] {group_name}")
    tabnet_status = "READY" if tabnet_ready else "FAILED"
    print(f"[{tabnet_status:^6}] TabNet - Deep Learning")
    print("=" * 60)


def main() -> int:
    """Run all constructor checks and return a process-compatible exit code."""
    group_status = verify_traditional_models()
    tabnet_ready = verify_tabnet()
    print_checklist(group_status, tabnet_ready)
    return 0 if all(group_status.values()) and tabnet_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
