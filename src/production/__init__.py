"""Production ML components (additive layer on top of the assessment pipeline).

The assessment modules in ``src/`` are plain top-level modules (``data_cleaning``,
``features`` ...). The saved model artifact pickles ``data_cleaning.CleaningStatistics``,
so ``src/`` must be importable as a path root. This package guarantees that.

Keep this file light: it must not import heavy dependencies.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
