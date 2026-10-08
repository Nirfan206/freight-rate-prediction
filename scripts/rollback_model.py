"""Point production back at an earlier registered version (nothing is deleted).

    python scripts/rollback_model.py --version v1
    python scripts/rollback_model.py            # roll back to the previously promoted version

The API loads the model at startup: restart it (or the container) to pick up the change.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import json

from production.config import get_settings
from production.logging_config import configure_logging
from production.model_registry import RegistryError, rollback_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=False, help="registered version, e.g. v1")
    parser.add_argument("--registry", type=Path, default=None, help="registry path (default: MODEL_REGISTRY_PATH)")
    parser.add_argument("--reason", default=None, help="optional note stored in production.json history")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        record = rollback_model(args.version, args.registry or settings.registry_path, reason=args.reason)
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in record.items() if k != "history"}, indent=2))
    print("Restart the API to load the new production model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
