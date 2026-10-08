"""Central configuration. Every environment-specific value comes from here.

Settings are read from environment variables (see ``.env.example``). Nothing here
is a secret; the project needs no credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_NAME = "freight_rate_ridge"
TRUE_VALUES = {"1", "true", "yes", "on"}


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in TRUE_VALUES


@dataclass(frozen=True)
class Settings:
    root: Path
    registry_path: Path
    monitoring_path: Path
    data_dir: Path
    production_model_version: Optional[str]  # explicit override of production.json
    log_level: str
    api_host: str
    api_port: int
    log_predictions: bool
    log_input_features: bool
    model_name: str = DEFAULT_MODEL_NAME

    @property
    def prediction_log_path(self) -> Path:
        return self.monitoring_path / "predictions" / "predictions.jsonl"

    @property
    def drift_dir(self) -> Path:
        return self.monitoring_path / "drift"

    @property
    def performance_dir(self) -> Path:
        return self.monitoring_path / "performance"

    @property
    def retraining_dir(self) -> Path:
        return self.monitoring_path / "retraining"

    @property
    def training_data_path(self) -> Path:
        return self.data_dir / "train-test.csv"


def get_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build settings from the environment. Called per use so tests can override."""
    env = os.environ if env is None else env
    override = (env.get("PRODUCTION_MODEL_VERSION") or "").strip() or None
    level = (env.get("LOG_LEVEL") or "INFO").strip().upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        level = "INFO"
    return Settings(
        root=ROOT,
        registry_path=_resolve(env.get("MODEL_REGISTRY_PATH") or "models/registry", ROOT),
        monitoring_path=_resolve(env.get("MONITORING_PATH") or "monitoring", ROOT),
        data_dir=_resolve(env.get("DATA_DIR") or "data", ROOT),
        production_model_version=override,
        log_level=level,
        api_host=env.get("API_HOST") or "0.0.0.0",
        api_port=int(env.get("API_PORT") or 8000),
        log_predictions=_as_bool(env.get("PREDICTION_LOGGING"), True),
        log_input_features=_as_bool(env.get("LOG_INPUT_FEATURES"), True),
    )
