"""Reusable EDA and preprocessing pipeline for the fraud datasets.

The module is the executable Python counterpart of ``EDA.ipynb``. It can be
imported from a Kaggle notebook so EDA, CSV generation, and model training can
run as separate stages without relying on notebook state.

Kaggle example::

    from Notebook.EDA import run_eda, run_preprocessing, load_processed_fold

    run_eda(datasets=("MLG_ULB",))
    run_preprocessing(datasets=("MLG_ULB",), folds=(1,))
    X_train, y_train, X_valid, y_valid = load_processed_fold(
        "MLG_ULB", fold=1, variant="without_duplicates"
    )

Command-line examples::

    python Notebook/EDA.py eda --datasets MLG_ULB IEEE_CIS Sparkov
    python Notebook/EDA.py preprocess --datasets MLG_ULB --folds 1
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - depends on the runtime image.
    SMOTE = None  # type: ignore[assignment,misc]


RANDOM_STATE = 42
N_SPLITS = 5
SUPPORTED_DATASETS = ("MLG_ULB", "IEEE_CIS", "Sparkov")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

RELATIVE_DATA_FILES = {
    "creditcard.csv": Path("data/Raw_data/MLG_ULB/creditcard.csv"),
    "train_transaction.csv": Path(
        "data/Raw_data/ieee-fraud-detection/train_transaction.csv"
    ),
    "train_identity.csv": Path(
        "data/Raw_data/ieee-fraud-detection/train_identity.csv"
    ),
    "fraudTrain.csv": Path("data/Raw_data/Sparkov/fraudTrain.csv"),
    "fraudTest.csv": Path("data/Raw_data/Sparkov/fraudTest.csv"),
}

KAGGLE_DATA_HINTS = {
    "creditcard.csv": KAGGLE_INPUT_ROOT
    / "mlg-ulb-creditcardfraud"
    / "creditcard.csv",
    "train_transaction.csv": KAGGLE_INPUT_ROOT
    / "ieee-fraud-detection"
    / "train_transaction.csv",
    "train_identity.csv": KAGGLE_INPUT_ROOT
    / "ieee-fraud-detection"
    / "train_identity.csv",
    "fraudTrain.csv": KAGGLE_INPUT_ROOT / "fraud-detection" / "fraudTrain.csv",
    "fraudTest.csv": KAGGLE_INPUT_ROOT / "fraud-detection" / "fraudTest.csv",
}

KAGGLE_DATASET_RESOURCES = {
    "MLG_ULB": {
        "handle": "mlg-ulb/creditcardfraud",
        "files": ("creditcard.csv",),
    },
    "Sparkov": {
        "handle": "kartik2112/fraud-detection",
        "files": ("fraudTrain.csv", "fraudTest.csv"),
    },
}
KAGGLE_COMPETITION_RESOURCES = {
    "IEEE_CIS": {
        "handle": "ieee-fraud-detection",
        "files": ("train_transaction.csv", "train_identity.csv"),
    }
}
_DOWNLOADED_DATA_FILES: dict[str, Path] = {}


@dataclass(frozen=True)
class FoldArtifacts:
    """Paths generated for one processed train/validation fold."""

    dataset: str
    fold: int
    train_csv: Path
    validation_csv: Path
    validation_reference_csv: Path
    metadata_csv: Path
    variant: str | None = None


def is_kaggle_runtime() -> bool:
    """Return whether the current process is running in a Kaggle filesystem."""
    return KAGGLE_INPUT_ROOT.is_dir() and KAGGLE_WORKING_ROOT.is_dir()


def get_project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the writable project root for Kaggle or the local repository."""
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    elif is_kaggle_runtime():
        root = KAGGLE_WORKING_ROOT / "FAIR_2026_Experiment"
    else:
        root = Path(__file__).resolve().parent.parent
    root.mkdir(parents=True, exist_ok=True)
    return root


def find_data_file(
    filename: str,
    project_root: str | Path | None = None,
) -> Path:
    """Find a raw CSV in Kaggle Inputs or the repository data directory."""
    if filename not in RELATIVE_DATA_FILES:
        raise KeyError(f"No path configuration exists for {filename!r}.")

    registered_path = _DOWNLOADED_DATA_FILES.get(filename)
    if registered_path is not None and registered_path.is_file():
        return registered_path

    if is_kaggle_runtime():
        hinted_path = KAGGLE_DATA_HINTS[filename]
        if hinted_path.is_file():
            return hinted_path
        matches = list(KAGGLE_INPUT_ROOT.rglob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            rendered = "\n".join(str(path) for path in matches)
            raise RuntimeError(
                f"Multiple Kaggle Inputs contain {filename!r}:\n{rendered}"
            )

    workspace_path = get_project_root(project_root) / RELATIVE_DATA_FILES[filename]
    if workspace_path.is_file():
        return workspace_path

    raw_data_root = get_project_root(project_root) / "data" / "Raw_data"
    workspace_matches = list(raw_data_root.rglob(filename))
    if len(workspace_matches) == 1:
        return workspace_matches[0]
    if len(workspace_matches) > 1:
        rendered = "\n".join(str(path) for path in workspace_matches)
        raise RuntimeError(
            f"Multiple downloaded files are named {filename!r}:\n{rendered}"
        )
    raise FileNotFoundError(
        f"Could not find {filename!r} at {workspace_path}"
        + (" or under /kaggle/input" if is_kaggle_runtime() else "")
    )


def _resolve_kagglehub_file(
    download_result: str | Path,
    filename: str,
    destination: Path,
) -> Path:
    """Resolve a file whether KaggleHub attached it or used ``output_dir``."""
    returned_path = Path(download_result).resolve()
    candidates: list[Path] = []

    if returned_path.is_file() and returned_path.name == filename:
        candidates.append(returned_path)
    elif returned_path.is_dir():
        direct_return = returned_path / filename
        if direct_return.is_file():
            candidates.append(direct_return)
        candidates.extend(returned_path.rglob(filename))

    direct_destination = destination / filename
    if direct_destination.is_file():
        candidates.append(direct_destination)
    if destination.is_dir():
        candidates.extend(destination.rglob(filename))
    if KAGGLE_INPUT_ROOT.is_dir():
        candidates.extend(KAGGLE_INPUT_ROOT.rglob(filename))

    candidates = list(
        dict.fromkeys(path.resolve() for path in candidates if path.is_file())
    )
    if not candidates:
        raise FileNotFoundError(
            f"KaggleHub finished but {filename!r} was not found. "
            f"Returned path: {returned_path}; output directory: {destination}"
        )
    return candidates[0]


def download_kaggle_data(
    datasets: Sequence[str] = SUPPORTED_DATASETS,
    *,
    force_download: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Download selected raw files with Kaggle's official ``kagglehub`` API.

    Public datasets authenticate automatically inside Kaggle notebooks. The
    IEEE-CIS competition requires the current Kaggle account to accept its
    rules before its files can be downloaded.
    """
    datasets = _validate_datasets(datasets)
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Missing kagglehub. Run: pip install kagglehub") from exc

    root = get_project_root(project_root)
    resolved: dict[str, Path] = {}

    for dataset in datasets:
        if dataset in KAGGLE_DATASET_RESOURCES:
            resource = KAGGLE_DATASET_RESOURCES[dataset]
            for filename in resource["files"]:
                destination = root / RELATIVE_DATA_FILES[filename].parent
                destination.mkdir(parents=True, exist_ok=True)
                existing = destination / filename
                download_result: str | Path = existing
                if not existing.is_file() or force_download:
                    print(f"[DOWNLOAD] {dataset}: {filename}")
                    download_result = kagglehub.dataset_download(
                        resource["handle"],
                        path=filename,
                        output_dir=str(destination),
                        force_download=force_download,
                    )
                resolved[filename] = _resolve_kagglehub_file(
                    download_result, filename, destination
                )
                _DOWNLOADED_DATA_FILES[filename] = resolved[filename]

        if dataset in KAGGLE_COMPETITION_RESOURCES:
            resource = KAGGLE_COMPETITION_RESOURCES[dataset]
            for filename in resource["files"]:
                destination = root / RELATIVE_DATA_FILES[filename].parent
                destination.mkdir(parents=True, exist_ok=True)
                existing = destination / filename
                download_result = existing
                if not existing.is_file() or force_download:
                    print(f"[DOWNLOAD] {dataset}: {filename}")
                    try:
                        download_result = kagglehub.competition_download(
                            resource["handle"],
                            path=filename,
                            output_dir=str(destination),
                            force_download=force_download,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Không tải được IEEE-CIS. Hãy mở competition "
                            "ieee-fraud-detection bằng đúng tài khoản Kaggle, "
                            "chấp nhận rules rồi chạy lại cell tải dữ liệu."
                        ) from exc
                resolved[filename] = _resolve_kagglehub_file(
                    download_result, filename, destination
                )
                _DOWNLOADED_DATA_FILES[filename] = resolved[filename]

    for filename, path in resolved.items():
        print(f"[READY] {filename}: {path}")
    return resolved


def _validate_datasets(datasets: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(datasets)
    unknown = sorted(set(normalized).difference(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError(
            f"Unsupported datasets: {unknown}. Expected one of {SUPPORTED_DATASETS}."
        )
    return normalized


def _validate_folds(folds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(fold) for fold in folds))
    invalid = [fold for fold in normalized if fold not in range(1, N_SPLITS + 1)]
    if invalid:
        raise ValueError(f"Fold numbers must be between 1 and {N_SPLITS}: {invalid}")
    return normalized


def _summarize_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    target: str | None,
    show_plots: bool,
) -> pd.DataFrame:
    """Print schema, missingness, duplicates, and target imbalance."""
    print(f"\n{'=' * 24} {dataset_name} {'=' * 24}")
    print(f"Shape: {len(dataframe):,} rows x {dataframe.shape[1]:,} columns")
    print(f"Fully duplicated rows: {dataframe.duplicated().sum():,}")

    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    identifier_columns = [
        column
        for column in dataframe.columns
        if column.lower() in {"transactionid", "cc_num", "trans_num", "unnamed: 0"}
        or column.lower().endswith("_id")
    ]
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    print(f"Identifier-like columns: {identifier_columns}")

    missing = dataframe.isna().sum()
    missing = missing[missing.gt(0)].sort_values(ascending=False)
    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_rate_pct": (missing / len(dataframe) * 100).round(4),
        }
    )
    print(f"Columns containing missing values: {len(missing_report):,}")
    if not missing_report.empty:
        print(missing_report.head(20).to_string())

    if target is not None:
        if target not in dataframe.columns:
            raise KeyError(f"Target {target!r} is missing from {dataset_name}.")
        counts = dataframe[target].value_counts(dropna=False).sort_index()
        rates = (
            dataframe[target]
            .value_counts(dropna=False, normalize=True)
            .sort_index()
            .mul(100)
            .round(6)
        )
        print(pd.DataFrame({"count": counts, "rate_pct": rates}).to_string())
        if show_plots:
            axis = counts.plot(
                kind="bar", logy=True, color=["#4C78A8", "#E45756"]
            )
            axis.set_title(f"{dataset_name} - {target} distribution (log scale)")
            axis.set_xlabel(target)
            axis.set_ylabel("Samples")
            plt.tight_layout()
            plt.show()

    if show_plots and not missing_report.empty:
        axis = (
            missing_report.head(20)
            .sort_values("missing_rate_pct")["missing_rate_pct"]
            .plot(kind="barh", color="#F58518")
        )
        axis.set_title(f"{dataset_name} - 20 columns with most missing data")
        axis.set_xlabel("Missing rate (%)")
        plt.tight_layout()
        plt.show()

    print(dataframe.head().to_string())
    return missing_report


def _plot_amount_by_target(
    dataframe: pd.DataFrame,
    amount_column: str,
    target: str,
    dataset_name: str,
) -> None:
    if amount_column not in dataframe.columns or target not in dataframe.columns:
        return
    _, axis = plt.subplots(figsize=(9, 4))
    for label, color in ((0, "#4C78A8"), (1, "#E45756")):
        values = pd.to_numeric(
            dataframe.loc[dataframe[target].eq(label), amount_column],
            errors="coerce",
        ).dropna()
        axis.hist(
            np.log1p(values[values.ge(0)]),
            bins=60,
            alpha=0.55,
            density=True,
            label=str(label),
            color=color,
        )
    axis.set_title(f"{dataset_name} - log1p({amount_column}) by {target}")
    axis.set_xlabel(f"log1p({amount_column})")
    axis.set_ylabel("Density")
    axis.legend(title=target)
    plt.tight_layout()
    plt.show()


def run_eda(
    datasets: Sequence[str] = SUPPORTED_DATASETS,
    *,
    show_plots: bool = True,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run EDA for selected datasets and return their missing-value reports."""
    datasets = _validate_datasets(datasets)
    reports: dict[str, pd.DataFrame] = {}
    print(f"Environment: {'Kaggle' if is_kaggle_runtime() else 'Local'}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Output root: {get_project_root(project_root)}")

    if "MLG_ULB" in datasets:
        dataframe = pd.read_csv(find_data_file("creditcard.csv", project_root))
        reports["MLG_ULB"] = _summarize_dataset(
            dataframe, "MLG-ULB", "Class", show_plots
        )
        if show_plots:
            _plot_amount_by_target(dataframe, "Amount", "Class", "MLG-ULB")
        del dataframe
        gc.collect()

    if "IEEE_CIS" in datasets:
        transaction = pd.read_csv(
            find_data_file("train_transaction.csv", project_root)
        )
        identity = pd.read_csv(find_data_file("train_identity.csv", project_root))
        reports["IEEE_CIS_transaction"] = _summarize_dataset(
            transaction, "IEEE-CIS train_transaction", "isFraud", show_plots
        )
        reports["IEEE_CIS_identity"] = _summarize_dataset(
            identity, "IEEE-CIS train_identity", None, show_plots
        )
        matching_ids = identity["TransactionID"].isin(transaction["TransactionID"]).sum()
        print(f"IEEE identity IDs present in transaction: {matching_ids:,}/{len(identity):,}")
        if show_plots:
            _plot_amount_by_target(
                transaction,
                "TransactionAmt",
                "isFraud",
                "IEEE-CIS train_transaction",
            )
        del transaction, identity
        gc.collect()

    if "Sparkov" in datasets:
        train = pd.read_csv(find_data_file("fraudTrain.csv", project_root))
        test = pd.read_csv(find_data_file("fraudTest.csv", project_root))
        reports["Sparkov_train"] = _summarize_dataset(
            train, "Sparkov fraudTrain", "is_fraud", show_plots
        )
        reports["Sparkov_test"] = _summarize_dataset(
            test, "Sparkov fraudTest", "is_fraud", show_plots
        )
        schema_difference = sorted(set(train.columns).symmetric_difference(test.columns))
        print(f"Sparkov train/test schema difference: {schema_difference}")
        if show_plots:
            _plot_amount_by_target(train, "amt", "is_fraud", "Sparkov fraudTrain")
        del train, test
        gc.collect()

    return reports


def _make_dense_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:  # scikit-learn < 1.2
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def _make_smote(**kwargs: Any) -> Any:
    """Construct SMOTE only when preprocessing is requested."""
    if SMOTE is None:
        raise ImportError(
            "Missing imbalanced-learn. Run: pip install imbalanced-learn"
        )
    return SMOTE(**kwargs)


def _build_mixed_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    numeric_fill_value: float = -999,
    categorical_fill_value: str = "Unknown",
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=numeric_fill_value)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=categorical_fill_value),
            ),
            ("onehot", _make_dense_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, list(numeric_columns)),
            ("cat", categorical_pipeline, list(categorical_columns)),
        ]
    )


def _fold_indices(features: pd.DataFrame, target: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(splitter.split(features, target))


def _write_definition(
    output_directory: Path,
    config: dict[str, Any],
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    row_count: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in config.items()]
    ).to_csv(output_directory / "config.csv", index=False)
    validation_fold = np.zeros(row_count, dtype=np.int8)
    for fold_number, (_, valid_indices) in enumerate(fold_indices, 1):
        validation_fold[np.asarray(valid_indices)] = fold_number
    pd.DataFrame(
        {
            "source_index": np.arange(row_count),
            "validation_fold": validation_fold,
        }
    ).to_csv(output_directory / "fold_assignments.csv", index=False)


def _ensure_writable_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}. Pass overwrite=True to replace them."
        )


def _write_fold(
    *,
    dataset: str,
    fold: int,
    output_directory: Path,
    feature_names: Sequence[str],
    train_features: np.ndarray,
    train_target: np.ndarray | pd.Series,
    valid_features: np.ndarray,
    valid_target: np.ndarray | pd.Series,
    target_name: str,
    reference: pd.DataFrame,
    metadata: dict[str, Any],
    overwrite: bool,
    variant: str | None = None,
) -> FoldArtifacts:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_path = output_directory / f"train_fold_{fold:02d}.csv"
    valid_path = output_directory / f"validation_fold_{fold:02d}.csv"
    reference_path = output_directory / f"validation_reference_fold_{fold:02d}.csv"
    metadata_path = output_directory / f"metadata_fold_{fold:02d}.csv"
    paths = (train_path, valid_path, reference_path, metadata_path)
    _ensure_writable_outputs(paths, overwrite)

    train_frame = pd.DataFrame(train_features, columns=feature_names)
    train_frame[target_name] = np.asarray(train_target, dtype=np.int8)
    valid_frame = pd.DataFrame(valid_features, columns=feature_names)
    valid_frame[target_name] = np.asarray(valid_target, dtype=np.int8)

    train_frame.to_csv(train_path, index=False)
    valid_frame.to_csv(valid_path, index=False)
    reference.to_csv(reference_path, index=False)
    pd.DataFrame([{**metadata, "dataset": dataset, "fold": fold}]).to_csv(
        metadata_path, index=False
    )
    print(f"[READY] {dataset} fold {fold}: {output_directory}")
    return FoldArtifacts(
        dataset=dataset,
        fold=fold,
        train_csv=train_path,
        validation_csv=valid_path,
        validation_reference_csv=reference_path,
        metadata_csv=metadata_path,
        variant=variant,
    )


def preprocess_mlg(
    *,
    folds: Sequence[int] = (1,),
    variants: Sequence[str] = ("with_duplicates", "without_duplicates"),
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> list[FoldArtifacts]:
    """Create leakage-safe MLG-ULB train/validation fold CSVs."""
    folds = _validate_folds(folds)
    allowed_variants = {"with_duplicates", "without_duplicates"}
    unknown_variants = sorted(set(variants).difference(allowed_variants))
    if unknown_variants:
        raise ValueError(f"Unsupported MLG duplicate variants: {unknown_variants}")

    dataframe = pd.read_csv(find_data_file("creditcard.csv", project_root))
    expected = ["Time", *[f"V{index}" for index in range(1, 29)], "Amount", "Class"]
    missing = [column for column in expected if column not in dataframe.columns]
    if missing:
        raise ValueError(f"MLG-ULB is missing required columns: {missing}")

    frames = {
        "with_duplicates": dataframe.reset_index(drop=True),
        "without_duplicates": dataframe.drop_duplicates().reset_index(drop=True),
    }
    output_root = (
        get_project_root(project_root)
        / "data"
        / "Processed_data"
        / "MLG_ULB"
        / "reproduction"
    )
    artifacts: list[FoldArtifacts] = []
    config = {
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "standard_scaler": True,
        "smote": True,
        "smote_k_neighbors": 5,
        "smote_sampling_strategy": 1.0,
    }

    for variant in variants:
        variant_frame = frames[variant]
        features = variant_frame.drop(columns=["Class"])
        target = variant_frame["Class"].astype(int)
        indices = _fold_indices(features, target)
        output_directory = output_root / variant
        variant_config = {
            **config,
            "duplicate_variant": variant,
            "remove_duplicates": variant == "without_duplicates",
        }
        _write_definition(output_directory, variant_config, indices, len(features))

        for fold in folds:
            train_indices, valid_indices = indices[fold - 1]
            scaler = StandardScaler()
            train_features = scaler.fit_transform(features.iloc[train_indices])
            valid_features = scaler.transform(features.iloc[valid_indices])
            train_target = target.iloc[train_indices]
            valid_target = target.iloc[valid_indices].to_numpy()
            smote = _make_smote(
                sampling_strategy=1.0,
                k_neighbors=5,
                random_state=RANDOM_STATE,
            )
            train_features, train_target = smote.fit_resample(
                train_features, train_target
            )
            artifacts.append(
                _write_fold(
                    dataset="MLG_ULB",
                    fold=fold,
                    output_directory=output_directory,
                    feature_names=features.columns,
                    train_features=train_features,
                    train_target=train_target,
                    valid_features=valid_features,
                    valid_target=valid_target,
                    target_name="Class",
                    reference=pd.DataFrame({"source_index": valid_indices}),
                    metadata={
                        "duplicate_variant": variant,
                        "train_rows_before_smote": len(train_indices),
                        "train_rows_after_smote": len(train_target),
                        "valid_rows": len(valid_indices),
                    },
                    overwrite=overwrite,
                    variant=variant,
                )
            )

    del dataframe, frames
    gc.collect()
    return artifacts


def _estimate_dense_gb(
    features: pd.DataFrame,
    target: pd.Series,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> float:
    estimated_columns = len(numeric_columns) + sum(
        features[column].nunique(dropna=False) for column in categorical_columns
    )
    estimated_balanced_rows = int(target.value_counts().max()) * 2
    return estimated_balanced_rows * estimated_columns * 8 / 1024**3


def preprocess_ieee(
    *,
    folds: Sequence[int],
    overwrite: bool = False,
    max_estimated_dense_gb: float = 8.0,
    project_root: str | Path | None = None,
) -> list[FoldArtifacts]:
    """LEFT JOIN and preprocess selected IEEE-CIS folds."""
    folds = _validate_folds(folds)
    transaction = pd.read_csv(find_data_file("train_transaction.csv", project_root))
    identity = pd.read_csv(find_data_file("train_identity.csv", project_root))
    dataframe = transaction.merge(identity, on="TransactionID", how="left")
    if len(dataframe) != len(transaction):
        raise RuntimeError("IEEE LEFT JOIN changed the transaction row count.")

    target = dataframe["isFraud"].astype(int)
    transaction_ids = dataframe["TransactionID"].copy()
    features = dataframe.drop(columns=["isFraud", "TransactionID"])
    indices = _fold_indices(features, target)
    output_directory = (
        get_project_root(project_root)
        / "data"
        / "Processed_data"
        / "IEEE_CIS"
        / "reproduction"
    )
    config = {
        "join": "left",
        "missing_threshold": 0.50,
        "numeric_fill_value": -999,
        "categorical_fill_value": "Unknown",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "smote_sampling_strategy": 1.0,
    }
    _write_definition(output_directory, config, indices, len(features))
    artifacts: list[FoldArtifacts] = []

    for fold in folds:
        train_indices, valid_indices = indices[fold - 1]
        train_frame = features.iloc[train_indices].copy()
        valid_frame = features.iloc[valid_indices].copy()
        drop_columns = train_frame.columns[train_frame.isna().mean().gt(0.50)].tolist()
        train_frame = train_frame.drop(columns=drop_columns)
        valid_frame = valid_frame.drop(columns=drop_columns)
        categorical = train_frame.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numeric = [column for column in train_frame.columns if column not in categorical]
        estimated_gb = _estimate_dense_gb(
            train_frame, target.iloc[train_indices], numeric, categorical
        )
        print(f"IEEE_CIS fold {fold} estimated dense RAM: {estimated_gb:.2f} GB")
        if estimated_gb > max_estimated_dense_gb:
            raise MemoryError(
                f"IEEE_CIS fold {fold} needs about {estimated_gb:.2f} GB before overhead; "
                f"limit is {max_estimated_dense_gb:.2f} GB."
            )

        preprocessor = _build_mixed_preprocessor(numeric, categorical)
        train_features = preprocessor.fit_transform(train_frame)
        valid_features = preprocessor.transform(valid_frame)
        smote = _make_smote(
            random_state=RANDOM_STATE,
            k_neighbors=5,
            sampling_strategy=1.0,
        )
        train_features, train_target = smote.fit_resample(
            train_features, target.iloc[train_indices]
        )
        artifacts.append(
            _write_fold(
                dataset="IEEE_CIS",
                fold=fold,
                output_directory=output_directory,
                feature_names=preprocessor.get_feature_names_out(),
                train_features=train_features,
                train_target=train_target,
                valid_features=valid_features,
                valid_target=target.iloc[valid_indices],
                target_name="isFraud",
                reference=pd.DataFrame(
                    {
                        "source_index": valid_indices,
                        "TransactionID": transaction_ids.iloc[valid_indices].to_numpy(),
                    }
                ),
                metadata={
                    "dropped_columns": "|".join(drop_columns),
                    "estimated_dense_train_gb": estimated_gb,
                    "train_rows_after_smote": len(train_target),
                    "valid_rows": len(valid_indices),
                },
                overwrite=overwrite,
            )
        )

    del transaction, identity, dataframe, features, target
    gc.collect()
    return artifacts


def _prepare_sparkov_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    dataframe = dataframe.copy()
    if "Unnamed: 0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Unnamed: 0"])
    transaction_ids = (
        dataframe["trans_num"].copy() if "trans_num" in dataframe.columns else None
    )
    dataframe = dataframe.drop(
        columns=[
            column
            for column in ("trans_num", "cc_num")
            if column in dataframe.columns
        ]
    )
    transaction_time = pd.to_datetime(
        dataframe["trans_date_trans_time"], errors="coerce"
    )
    dataframe["transaction_hour"] = transaction_time.dt.hour
    dataframe["transaction_day"] = transaction_time.dt.day
    dataframe["transaction_month"] = transaction_time.dt.month
    dataframe["transaction_weekday"] = transaction_time.dt.dayofweek
    dataframe["is_weekend"] = (transaction_time.dt.dayofweek >= 5).astype("int8")
    dataframe = dataframe.drop(columns=["trans_date_trans_time"])
    dataframe = dataframe.drop(
        columns=[
            column
            for column in ("first", "last", "street")
            if column in dataframe.columns
        ]
    )
    target = dataframe.pop("is_fraud").astype(int)
    return dataframe, target, transaction_ids


def preprocess_sparkov(
    *,
    folds: Sequence[int],
    overwrite: bool = False,
    max_estimated_dense_gb: float = 8.0,
    project_root: str | Path | None = None,
) -> list[FoldArtifacts]:
    """Feature-engineer and preprocess selected Sparkov folds."""
    folds = _validate_folds(folds)
    dataframe = pd.read_csv(find_data_file("fraudTrain.csv", project_root))
    features, target, transaction_ids = _prepare_sparkov_features(dataframe)
    categorical = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    indices = _fold_indices(features, target)
    output_directory = (
        get_project_root(project_root)
        / "data"
        / "Processed_data"
        / "Sparkov"
        / "reproduction"
    )
    config = {
        "drop_unnamed_index": True,
        "drop_trans_num": True,
        "drop_cc_num": True,
        "engineer_time_features": True,
        "drop_high_cardinality_personal": True,
        "keep_unix_time": True,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "smote_sampling_strategy": 1.0,
    }
    _write_definition(output_directory, config, indices, len(features))
    artifacts: list[FoldArtifacts] = []

    for fold in folds:
        train_indices, valid_indices = indices[fold - 1]
        train_frame = features.iloc[train_indices]
        valid_frame = features.iloc[valid_indices]
        estimated_gb = _estimate_dense_gb(
            train_frame, target.iloc[train_indices], numeric, categorical
        )
        print(f"Sparkov fold {fold} estimated dense RAM: {estimated_gb:.2f} GB")
        if estimated_gb > max_estimated_dense_gb:
            raise MemoryError(
                f"Sparkov fold {fold} needs about {estimated_gb:.2f} GB before overhead; "
                f"limit is {max_estimated_dense_gb:.2f} GB."
            )

        preprocessor = _build_mixed_preprocessor(numeric, categorical)
        train_features = preprocessor.fit_transform(train_frame)
        valid_features = preprocessor.transform(valid_frame)
        smote = _make_smote(
            random_state=RANDOM_STATE,
            k_neighbors=5,
            sampling_strategy=1.0,
        )
        train_features, train_target = smote.fit_resample(
            train_features, target.iloc[train_indices]
        )
        reference_data: dict[str, Any] = {"source_index": valid_indices}
        if transaction_ids is not None:
            reference_data["trans_num"] = transaction_ids.iloc[valid_indices].to_numpy()
        artifacts.append(
            _write_fold(
                dataset="Sparkov",
                fold=fold,
                output_directory=output_directory,
                feature_names=preprocessor.get_feature_names_out(),
                train_features=train_features,
                train_target=train_target,
                valid_features=valid_features,
                valid_target=target.iloc[valid_indices],
                target_name="is_fraud",
                reference=pd.DataFrame(reference_data),
                metadata={
                    "estimated_dense_train_gb": estimated_gb,
                    "train_rows_after_smote": len(train_target),
                    "valid_rows": len(valid_indices),
                },
                overwrite=overwrite,
            )
        )

    del dataframe, features, target
    gc.collect()
    return artifacts


def run_preprocessing(
    datasets: Sequence[str] = ("MLG_ULB",),
    *,
    folds: Sequence[int] = (1,),
    overwrite: bool = False,
    max_estimated_dense_gb: float = 8.0,
    project_root: str | Path | None = None,
) -> list[FoldArtifacts]:
    """Preprocess selected datasets and return every generated CSV bundle."""
    datasets = _validate_datasets(datasets)
    folds = _validate_folds(folds)
    artifacts: list[FoldArtifacts] = []
    if "MLG_ULB" in datasets:
        artifacts.extend(
            preprocess_mlg(
                folds=folds,
                overwrite=overwrite,
                project_root=project_root,
            )
        )
    if "IEEE_CIS" in datasets:
        artifacts.extend(
            preprocess_ieee(
                folds=folds,
                overwrite=overwrite,
                max_estimated_dense_gb=max_estimated_dense_gb,
                project_root=project_root,
            )
        )
    if "Sparkov" in datasets:
        artifacts.extend(
            preprocess_sparkov(
                folds=folds,
                overwrite=overwrite,
                max_estimated_dense_gb=max_estimated_dense_gb,
                project_root=project_root,
            )
        )
    return artifacts


def load_processed_fold(
    dataset: str,
    *,
    fold: int,
    variant: str | None = None,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load generated CSVs as model-ready train and validation arrays."""
    _validate_datasets((dataset,))
    _validate_folds((fold,))
    target_names = {"MLG_ULB": "Class", "IEEE_CIS": "isFraud", "Sparkov": "is_fraud"}
    output_directory = (
        get_project_root(project_root)
        / "data"
        / "Processed_data"
        / dataset
        / "reproduction"
    )
    if dataset == "MLG_ULB":
        selected_variant = variant or "without_duplicates"
        if selected_variant not in {"with_duplicates", "without_duplicates"}:
            raise ValueError(f"Invalid MLG duplicate variant: {selected_variant}")
        output_directory /= selected_variant
    elif variant is not None:
        raise ValueError("variant is only valid for MLG_ULB.")

    train_path = output_directory / f"train_fold_{fold:02d}.csv"
    valid_path = output_directory / f"validation_fold_{fold:02d}.csv"
    missing = [path for path in (train_path, valid_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Processed fold files do not exist: {missing}. Run run_preprocessing first."
        )
    target_name = target_names[dataset]
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    return (
        train.drop(columns=[target_name]),
        train[target_name].astype(int),
        valid.drop(columns=[target_name]),
        valid[target_name].astype(int),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda_parser = subparsers.add_parser("eda", help="Run exploratory data analysis.")
    eda_parser.add_argument(
        "--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS)
    )
    eda_parser.add_argument("--no-plots", action="store_true")

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Generate model-ready fold CSV files."
    )
    preprocess_parser.add_argument(
        "--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=["MLG_ULB"]
    )
    preprocess_parser.add_argument("--folds", nargs="+", type=int, default=[1])
    preprocess_parser.add_argument("--overwrite", action="store_true")
    preprocess_parser.add_argument("--max-estimated-dense-gb", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and print machine-readable artifact paths."""
    if sys.version_info < (3, 10):
        raise RuntimeError("EDA.py requires Python 3.10 or newer.")
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "eda":
        run_eda(arguments.datasets, show_plots=not arguments.no_plots)
        return 0

    artifacts = run_preprocessing(
        arguments.datasets,
        folds=arguments.folds,
        overwrite=arguments.overwrite,
        max_estimated_dense_gb=arguments.max_estimated_dense_gb,
    )
    print(json.dumps([asdict(item) for item in artifacts], default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
