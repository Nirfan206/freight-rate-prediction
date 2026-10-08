"""FastAPI service for the production freight-rate model.

Importing this package makes ``src/`` importable (the saved model artifact and the
assessment cleaning/feature code live there). It does not import FastAPI.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))
