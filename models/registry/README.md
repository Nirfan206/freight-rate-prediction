# Model registry (file based)

```
models/registry/
  model_v1/model.joblib      # joblib artifact: {model, feature_columns, cleaning_stats}
  model_v1/metadata.json     # name, version, algorithm, alpha, rows, features, metrics, dates, versions, interval, drift reference
  model_v2/...
  production.json            # which version the API serves
```

* **v1** is a byte-for-byte copy of the assessment artifact `artifacts/ridge_model.joblib`.
* Versions are immutable: registration never overwrites, nothing here deletes a version.
* `train_production.py` registers a new version but never promotes it.
* `promote_model.py` / `rollback_model.py` validate (model file, metadata keys, metrics, feature count vs the
  current feature pipeline, smoke prediction) and then rewrite `production.json` atomically; each change is
  appended to its `history`.
* The API loads the model at startup: restart it after promote/rollback.
* **Security / portability:** `.joblib` files are pickles. Only load files from a registry you control. A pickle
  is tied to the scikit-learn version that wrote it (`metadata.sklearn_version`); the loader logs a warning on mismatch
  and the promotion smoke test refuses incompatible artifacts.
* `metadata.python_version` / `git_commit` describe the environment that *registered* the version.

```
python scripts/register_assessment_model.py   # one-off: register the assessment artifact (no-op if present)
python scripts/train_production.py            # register next version
python scripts/promote_model.py  --version v2
python scripts/rollback_model.py --version v1
```
