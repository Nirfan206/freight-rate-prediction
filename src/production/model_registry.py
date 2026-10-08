"""Lightweight file-based model registry (no MLflow needed).

    models/registry/
        model_v1/{model.joblib, metadata.json}
        model_v2/...
        production.json        <- which version is live

Rules: versions are immutable (never overwritten, never deleted by this code);
promotion/rollback only rewrites ``production.json`` after validation.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from production.config import DEFAULT_MODEL_NAME, ROOT, get_settings
from production.logging_config import get_logger

log = get_logger("registry")

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
PRODUCTION_FILENAME = "production.json"
VERSION_RE = re.compile(r"^v(\d+)$")

REQUIRED_METADATA_KEYS = (
    "model_name", "version", "algorithm", "alpha", "training_rows", "feature_count",
    "training_date", "training_data_period", "target_transformation", "metrics",
    "sklearn_version", "python_version", "feature_names", "git_commit",
)
REQUIRED_METRIC_KEYS = ("MAE", "RMSE", "R2")


class RegistryError(Exception):
    """Registry operation failed (message is safe to show to an operator)."""


# ----------------------------------------------------------------------------- helpers

def _registry(path: Optional[Path]) -> Path:
    return Path(path) if path is not None else get_settings().registry_path


def version_dir(registry_path: Path, version: str) -> Path:
    if not VERSION_RE.match(version):
        raise RegistryError(f"invalid version '{version}' (expected like 'v1')")
    return registry_path / f"model_{version}"


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_metadata(registry_path: Path, version: str) -> dict:
    path = version_dir(registry_path, version) / METADATA_FILENAME
    if not path.is_file():
        raise RegistryError(f"metadata.json missing for {version}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"metadata.json for {version} is not valid JSON: {exc}") from exc


def _version_numbers(registry_path: Path) -> list[int]:
    if not registry_path.is_dir():
        return []
    nums = []
    for child in registry_path.iterdir():
        m = re.match(r"^model_v(\d+)$", child.name)
        if child.is_dir() and m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def read_production_record(registry_path: Optional[Path] = None) -> Optional[dict]:
    path = _registry(registry_path) / PRODUCTION_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"production.json is not valid JSON: {exc}") from exc


# ----------------------------------------------------------------------------- listing

def list_models(registry_path: Optional[Path] = None) -> list[dict]:
    """All registered versions (oldest first) with a short summary."""
    reg = _registry(registry_path)
    record = read_production_record(reg)
    prod = record.get("production_version") if record else None
    rows = []
    for num in _version_numbers(reg):
        version = f"v{num}"
        try:
            meta = read_metadata(reg, version)
        except RegistryError:
            meta = None
        metrics = (meta or {}).get("metrics") or {}
        rows.append({
            "version": version,
            "path": str(version_dir(reg, version)),
            "is_production": version == prod,
            "complete": bool(meta) and (version_dir(reg, version) / MODEL_FILENAME).is_file(),
            "algorithm": (meta or {}).get("algorithm"),
            "alpha": (meta or {}).get("alpha"),
            "training_date": (meta or {}).get("training_date"),
            "training_rows": (meta or {}).get("training_rows"),
            "feature_count": (meta or {}).get("feature_count"),
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2"),
        })
    return rows


# ----------------------------------------------------------------------------- register

def register_model(
    metadata: dict[str, Any],
    registry_path: Optional[Path] = None,
    *,
    artifact: Optional[dict] = None,
    model_file: Optional[Path] = None,
) -> dict:
    """Store a new immutable model version and return its final metadata.

    Provide either ``artifact`` (dict with model / feature_columns / cleaning_stats,
    serialised with joblib) or ``model_file`` (an existing .joblib copied byte-for-byte).
    Does NOT promote the version.
    """
    import joblib
    import sklearn

    if (artifact is None) == (model_file is None):
        raise RegistryError("provide exactly one of artifact or model_file")
    reg = _registry(registry_path)
    reg.mkdir(parents=True, exist_ok=True)

    nums = _version_numbers(reg)
    version = f"v{(nums[-1] + 1) if nums else 1}"
    meta = dict(metadata)
    meta.setdefault("model_name", DEFAULT_MODEL_NAME)
    meta["version"] = version
    meta.setdefault("training_date", _utc_now())
    meta.setdefault("sklearn_version", sklearn.__version__)
    meta.setdefault("python_version", platform.python_version())
    meta.setdefault("git_commit", get_git_commit())
    meta["registered_at"] = _utc_now()

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise RegistryError(f"metadata missing required key(s): {missing}")
    bad_metrics = [k for k in REQUIRED_METRIC_KEYS if k not in (meta.get("metrics") or {})]
    if bad_metrics:
        raise RegistryError(f"metadata.metrics missing: {bad_metrics}")
    if int(meta["feature_count"]) != len(meta["feature_names"]):
        raise RegistryError("feature_count does not match len(feature_names)")

    target = version_dir(reg, version)
    target.mkdir(parents=True, exist_ok=False)  # never overwrite an existing version
    try:
        if model_file is not None:
            shutil.copyfile(model_file, target / MODEL_FILENAME)
        else:
            joblib.dump(artifact, target / MODEL_FILENAME)
        _atomic_write_json(target / METADATA_FILENAME, meta)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)  # only the directory created just above
        raise
    log.info("model registered", extra={"event": "model_registered", "model_version": version})
    return meta


# ----------------------------------------------------------------------------- validation

def validate_version(version: str, registry_path: Optional[Path] = None, *, smoke: bool = True) -> list[str]:
    """Return a list of problems (empty list = safe to promote)."""
    reg = _registry(registry_path)
    problems: list[str] = []
    try:
        vdir = version_dir(reg, version)
    except RegistryError as exc:
        return [str(exc)]
    if not vdir.is_dir():
        return [f"version {version} is not registered"]
    model_path = vdir / MODEL_FILENAME
    if not model_path.is_file():
        problems.append("model file (model.joblib) does not exist")
    try:
        meta = read_metadata(reg, version)
    except RegistryError as exc:
        return problems + [str(exc)]

    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        problems.append(f"metadata missing required key(s): {missing}")
    metrics = meta.get("metrics") or {}
    missing_metrics = [k for k in REQUIRED_METRIC_KEYS if metrics.get(k) is None]
    if missing_metrics:
        problems.append(f"metrics missing: {missing_metrics}")
    if meta.get("version") != version:
        problems.append(f"metadata version '{meta.get('version')}' != directory version '{version}'")
    if "feature_names" in meta and "feature_count" in meta and int(meta["feature_count"]) != len(meta["feature_names"]):
        problems.append("metadata feature_count != len(feature_names)")

    if smoke and model_path.is_file() and not problems:
        from production.predictor import expected_feature_count, smoke_test_artifact
        try:
            expected = expected_feature_count()
            if int(meta["feature_count"]) != expected:
                problems.append(
                    f"feature count {meta['feature_count']} does not match the current feature pipeline ({expected})"
                )
            problems.extend(smoke_test_artifact(model_path, meta))
        except Exception as exc:  # unreadable pickle, version mismatch, ...
            problems.append(f"smoke prediction failed: {type(exc).__name__}: {exc}")
    return problems


# ----------------------------------------------------------------------------- promote / rollback

def _set_production(version: str, reg: Path, action: str, reason: Optional[str]) -> dict:
    current = read_production_record(reg)
    previous = current.get("production_version") if current else None
    meta = read_metadata(reg, version)
    history = list((current or {}).get("history", []))
    history.append({"timestamp": _utc_now(), "action": action, "from": previous, "to": version, "reason": reason})
    record = {
        "model_name": meta.get("model_name", DEFAULT_MODEL_NAME),
        "production_version": version,
        "status": "production",
        "promoted_at": history[-1]["timestamp"],
        "previous_version": previous,
        "history": history,
    }
    _atomic_write_json(reg / PRODUCTION_FILENAME, record)
    log.info("production model changed", extra={"event": f"model_{action}", "model_version": version, "previous_version": previous})
    return record


def promote_model(version: str, registry_path: Optional[Path] = None, *, reason: Optional[str] = None, smoke: bool = True) -> dict:
    reg = _registry(registry_path)
    problems = validate_version(version, reg, smoke=smoke)
    if problems:
        raise RegistryError(f"cannot promote {version}: " + "; ".join(problems))
    return _set_production(version, reg, "promote", reason)


def rollback_model(version: Optional[str] = None, registry_path: Optional[Path] = None, *, reason: Optional[str] = None, smoke: bool = True) -> dict:
    """Point production at ``version`` (default: the previously promoted version)."""
    reg = _registry(registry_path)
    current = read_production_record(reg)
    if version is None:
        version = (current or {}).get("previous_version")
        if not version:
            raise RegistryError("no previous version recorded; pass an explicit version")
    problems = validate_version(version, reg, smoke=smoke)
    if problems:
        raise RegistryError(f"cannot roll back to {version}: " + "; ".join(problems))
    return _set_production(version, reg, "rollback", reason)


def get_production_model(registry_path: Optional[Path] = None, override_version: Optional[str] = None) -> dict:
    """Locate the production model (paths + metadata). Does not unpickle anything."""
    reg = _registry(registry_path)
    if override_version:
        version, source = override_version, "PRODUCTION_MODEL_VERSION override"
    else:
        record = read_production_record(reg)
        if not record or not record.get("production_version"):
            raise RegistryError(
                f"no production model configured: {reg / PRODUCTION_FILENAME} is missing or empty. "
                "Register and promote a model first (see scripts/promote_model.py)."
            )
        version, source = record["production_version"], PRODUCTION_FILENAME
    vdir = version_dir(reg, version)
    model_path = vdir / MODEL_FILENAME
    if not model_path.is_file():
        raise RegistryError(f"production model {version} ({source}) not found at {model_path}")
    return {"version": version, "source": source, "model_file": model_path, "metadata": read_metadata(reg, version)}
