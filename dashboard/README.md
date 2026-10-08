# Dashboard (additional layer)

Read-only report UI for this project. It reads existing files in `data/`, `outputs/`, `artifacts/model_metadata.json`,
`validation_predictions.csv` and runs `score.py` into a temp folder. It never trains, never loads the model pickle,
and never writes outside `dashboard/dist/`.

    python dashboard/build_dashboard.py      # needs pandas + numpy only
    # open dashboard/dist/index.html (single self-contained file, works offline)

Rebuild after re-running any pipeline step. Missing optional files show an empty state instead of crashing.
