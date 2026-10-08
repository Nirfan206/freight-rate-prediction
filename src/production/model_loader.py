"""Load the promoted model from the registry. Never trains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from production.config import Settings, get_settings
from production.logging_config import get_logger
from production.model_registry import RegistryError, get_production_model

log = get_logger("model_loader")


class ModelNotAvailableError(RuntimeError):
    """The production model could not be located or loaded."""


@dataclass
class LoadedModel:
    version: str
    model: Any                 # sklearn Pipeline (imputer -> scaler -> Ridge), trained on log1p(rate)
    feature_columns: list
    cleaning_stats: Any        # data_cleaning.CleaningStatistics
    metadata: dict


def load_artifact_file(model_file: Path) -> dict:
    """Unpickle an artifact. SECURITY: pickles can execute code - only load registry
    files you trust (this repository's models/registry), never user-supplied files."""
    import joblib
    import production  # noqa: F401  (ensures src/ is on sys.path for the pickled class)

    artifact = joblib.load(model_file)
    for key in ("model", "feature_columns", "cleaning_stats"):
        if key not in artifact:
            raise ModelNotAvailableError(f"model artifact {model_file} is missing '{key}'")
    return artifact


def load_model(model_file: Path, metadata: dict, version: str) -> LoadedModel:
    import sklearn

    artifact = load_artifact_file(model_file)
    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        log.warning(
            "scikit-learn version differs from the one the model was trained with",
            extra={"event": "sklearn_version_mismatch", "trained_with": trained_with, "runtime": sklearn.__version__},
        )
    return LoadedModel(version, artifact["model"], list(artifact["feature_columns"]), artifact["cleaning_stats"], metadata)


def load_production_model(settings: Optional[Settings] = None) -> LoadedModel:
    settings = settings or get_settings()
    try:
        located = get_production_model(settings.registry_path, settings.production_model_version)
    except RegistryError as exc:
        raise ModelNotAvailableError(str(exc)) from exc
    try:
        loaded = load_model(located["model_file"], located["metadata"], located["version"])
    except ModelNotAvailableError:
        raise
    except Exception as exc:
        raise ModelNotAvailableError(f"failed to load model {located['version']}: {type(exc).__name__}: {exc}") from exc
    log.info(
        "production model loaded",
        extra={"event": "model_loaded", "model_version": loaded.version, "source": located["source"],
               "feature_count": len(loaded.feature_columns)},
    )
    return loaded
