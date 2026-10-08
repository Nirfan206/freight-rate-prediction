#!/usr/bin/env python3
"""Build dashboard/dist/index.html from the repository's existing outputs.

Read-only on the ML project: no training, no model loading, no file in data/,
src/, artifacts/, outputs/ or validation_predictions.csv is modified. The scorer
is run into a temporary directory. Usage: python dashboard/build_dashboard.py
"""
import base64, json, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
P = lambda *a: ROOT.joinpath(*a)
missing = []


def rd(*a, **kw):
    try:
        return pd.read_csv(P(*a), **kw)
    except Exception:
        missing.append("/".join(a))


def recs(df):
    return [] if df is None else json.loads(df.to_json(orient="records"))


tr, va = rd("data", "train-test.csv"), rd("data", "validation.csv")
if tr is None or va is None:
    sys.exit("ERROR: data/train-test.csv and data/validation.csv are required.")
for d in (tr, va):
    d["date"] = pd.to_datetime(d["date"])
meta = json.loads(P("artifacts", "model_metadata.json").read_text())
err = rd("outputs", "experiments", "error_overall.csv").iloc[0]
tune = rd("outputs", "experiments", "ridge_tuning_results.csv").sort_values("alpha")
fin = tune[tune.alpha == meta["alpha"]].iloc[0]
ex, tv = rd("outputs", "experiments", "model_experiment_results.csv"), rd("outputs", "experiments", "temporal_validation_results.csv")
sp = rd("outputs", "eda", "spearman_correlation.csv", index_col=0)
es = rd("outputs", "eda", "eda_summary.csv").set_index("metric").value
dec = rd("outputs", "predictions", "december_predictions.csv")
vp = rd("validation_predictions.csv")

# ---- data profile -------------------------------------------------------
def prof(d):
    return dict(rows=len(d), start=str(d.date.min().date()), end=str(d.date.max().date()),
                miss_w=int(d.weight.isna().sum()), neg_w=int((d.weight < 0).sum()),
                cap=int((d.weight.abs() >= 47500).sum()), miss_mi=int(d.market_index.isna().sum()),
                dup=int(d.drop(columns="load_id").duplicated().sum()), dup_id=int(d.load_id.duplicated().sum()),
                bad_dist=int((d.distance <= 0).sum()), equip=d.equipment.value_counts().to_dict(), cities=int(d.pickup.nunique()))

pre = tr[tr.date < "2025-09-01"]
q99 = float(tr.posted_rate.quantile(.99))
cnt, edges = np.histogram(tr.posted_rate, bins=30, range=(0, q99))
mi = pd.concat([tr.assign(s="train"), va.assign(s="assessment")])
mi = mi.groupby([mi.date.dt.strftime("%Y-%m"), "s"]).market_index.mean().reset_index()
new_mask = ~va.pickup.isin(set(tr.pickup)) | ~va.delivery.isin(set(tr.delivery))
newc = sorted((set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)))

# ---- features -----------------------------------------------------------
GROUPS = [  # first match wins; text is read from src/data_cleaning.py + src/features.py
    ("Quality flags", r"weight_missing|weight_negative|weight_at_|market_index_missing|distance_invalid|weight_raw",
     "Raw weight / market index / distance checks -> 0-1 flags (and the untouched raw weight) so the model can see which values were repaired."),
    ("Equipment", r"^equipment_", "Equipment text -> one-hot columns (Dry Van, Reefer, Flatbed, Unknown), plus each column x log distance and x market index."),
    ("Time", r"^day_of", "Load date -> day of week, day of month, day of year."),
    ("Market", r"^market_index$|^quote_signal$|^market_distance", "Market index and quote signal as given; market index x log distance."),
    ("Geography", r"lat|lon|haversine", "City coordinates -> raw coordinates, latitude/longitude differences (signed and absolute), great-circle (haversine) miles, and road distance vs haversine ratio / gap."),
    ("Weight", r"^weight", "Weight -> log weight (negatives clipped to 0), absolute weight, weight/distance, weight x log distance."),
    ("Distance", r"^(distance|log_distance)$", "Road distance as given, and log(1 + distance)."),
]
groups = {g[0]: [] for g in GROUPS}
groups["Other"] = []
for f in meta["features"]:
    groups[next((g[0] for g in GROUPS if re.search(g[1], f)), "Other")].append(f)
groups_out = [dict(name=g[0], how=g[2], features=groups[g[0]]) for g in GROUPS] + ([dict(name="Other", how="Not matched by a group rule.", features=groups["Other"])] if groups["Other"] else [])

# ---- models -------------------------------------------------------------
models = [dict(model="Ridge, alpha=0.01, log target (final)", MAE=fin.MAE, RMSE=fin.RMSE, R2=fin.R2, src="ridge_tuning_results.csv", final=True)]
models += [dict(model=r["Model"], MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"], src="temporal_validation_results.csv") for r in recs(tv)]
models += [dict(model=r["Model"] + (" (experiments.py settings)" if "Rate Per" not in r["Model"] and "Median" not in r["Model"] else ""), MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"], src="model_experiment_results.csv") for r in recs(ex)]
models.sort(key=lambda m: m["MAE"])

# ---- December scenario --------------------------------------------------
dec_info = dict(mi_fallback=float(tr.market_index.median()), qs_fallback=float(tr.quote_signal.median()),
                mi_nd=float(va.market_index.mean()), mi_train=float(tr.market_index.mean()), qs_nd=float(va.quote_signal.mean()))

# ---- submission checks (verified here, not assumed) ------------------------
checks = []
def chk(name, ok, detail):
    checks.append(dict(name=name, status="pass" if ok else "fail", detail=detail))

exp_ids = [f"TE-{i:06d}" for i in range(1, 12001)]
chk("validation_predictions.csv exists with load_id,predicted_rate", vp is not None and list(vp.columns) == ["load_id", "predicted_rate"], "Column names and order checked")
chk("12,000 validation rows", vp is not None and len(vp) == 12000, f"{0 if vp is None else len(vp):,} rows")
chk("Load IDs exact (TE-000001..TE-012000) and match data/validation.csv", vp is not None and vp.load_id.tolist() == exp_ids and vp.load_id.tolist() == va.load_id.tolist(), "Order and values compared")
pr = vp.predicted_rate if vp is not None else pd.Series([0.0])
chk("All predictions finite and positive", bool(np.isfinite(pr).all() and (pr > 0).all()), f"min ${pr.min():,.2f}, max ${pr.max():,.2f}")
chk("December predictions file", dec is not None, "outputs/predictions/december_predictions.csv")
chk("31 December rows, one per day", dec is not None and len(dec) == 31 and pd.to_datetime(dec.date).nunique() == 31, "2025-12-01 to 2025-12-31")
chk("December chart (scorer output)", P("scorer_results", "candidate_december.png").is_file(), "scorer_results/candidate_december.png")
chk("Report document", any(P("reports").glob("*.docx")), ", ".join(p.name for p in P("reports").glob("*.docx")) or "none")
chk("requirements.txt", P("requirements.txt").is_file(), f"{len(P('requirements.txt').read_text().split())} packages listed")
ntests = len(re.findall(r"^def test_", P("tests", "test_pipeline.py").read_text(), re.M))
chk("Tests present", ntests > 0, f"{ntests} test functions in tests/test_pipeline.py (this dashboard does not execute them)")
with tempfile.TemporaryDirectory() as t:  # never writes to scorer_results/
    r = subprocess.run([sys.executable, str(P("score.py")), "--predictions", str(P("validation_predictions.csv")),
                        "--december-predictions", str(P("outputs", "predictions", "december_predictions.csv")), "--output-dir", t],
                       capture_output=True, text=True, cwd=ROOT)
scorer = dict(code=r.returncode, out=(r.stdout + r.stderr).strip().replace(t, "<tmp>"))

# ---- EDA gallery ----------------------------------------------------------
rpm = tr.posted_rate / tr.distance
bk = pd.cut(tr.distance, [0, 200, 500, 1000, 1500, 2000, np.inf], labels=["<=200", "201-500", "501-1000", "1001-1500", "1501-2000", ">2000"])
bm = rpm.groupby(bk, observed=True).median()
eqm, eqr = tr.groupby("equipment").posted_rate.median(), rpm.groupby(tr.equipment).median()
mm, dw = tr.groupby(tr.date.dt.month).posted_rate.median(), tr.groupby(tr.date.dt.dayofweek).posted_rate.median()
S = lambda a, b: sp.loc[a, b]
T, Tg, Dn, Eq, Mk, Tm, TV = "Target analysis", "Target analysis", "Distance analysis", "Equipment analysis", "Market analysis", "Temporal analysis", "Train vs validation"
eda = [
    ("01_posted_rate_distribution", T, "Posted rate distribution", "How often each posted rate occurs.", f"Mean ${es.target_mean:,.0f} is above the median ${es.target_median:,.0f}; the maximum is ${es.target_max:,.0f}, so there is a long right tail."),
    ("02_log_posted_rate_distribution", T, "Log posted rate distribution", "The same distribution after log1p, the target scale the final model learns.", "Training on log1p(posted_rate) compresses the long tail."),
    ("03_rate_per_mile_distribution", T, "Rate per mile distribution", "Posted rate divided by distance.", f"Mean ${es.rate_per_mile_mean:.2f}, median ${es.rate_per_mile_median:.2f}, range ${es.rate_per_mile_min:.2f} to ${es.rate_per_mile_max:.2f} per mile."),
    ("08_posted_rate_vs_weight", T, "Posted rate vs weight", "Each load's weight against its posted rate.", f"Weak relationship: Spearman correlation {S('weight','posted_rate'):.3f}."),
    ("04_posted_rate_vs_distance", Dn, "Posted rate vs distance", "Each load's distance against its posted rate.", f"Strongest single signal: Spearman correlation {S('distance','posted_rate'):.3f}."),
    ("05_rate_per_mile_vs_distance", Dn, "Rate per mile vs distance", "How price per mile changes with trip length.", f"Spearman {S('distance','rate_per_mile'):.3f}: longer trips cost less per mile."),
    ("12_rate_per_mile_by_distance_bucket", Dn, "Rate per mile by distance bucket", "Rate per mile in distance bands.", f"Median rate per mile falls from ${bm.iloc[0]:.2f} (<=200 mi) to ${bm.iloc[-1]:.2f} (>2000 mi)."),
    ("06_median_rate_by_equipment", Eq, "Median rate by equipment", "Median posted rate for each equipment type.", "Median posted rate: " + ", ".join(f"{k} ${v:,.0f}" for k, v in eqm.items()) + "."),
    ("07_rate_per_mile_by_equipment", Eq, "Rate per mile by equipment", "Median rate per mile for each equipment type.", "Median rate per mile: " + ", ".join(f"{k} ${v:.2f}" for k, v in eqr.items()) + "."),
    ("09_market_index_over_time", Mk, "Market index over time", "Daily market index, Jan-Oct 2025.", "Weekly cycles on top of a seasonal rise and fall; monthly means range from " + f"{mi[mi.s=='train'].market_index.min():.2f} to {mi[mi.s=='train'].market_index.max():.2f}."),
    ("10_quote_signal_distribution", Mk, "Quote signal distribution", "Spread of the quote signal feature.", f"Mean {es.quote_signal_mean:.2f}; weak raw correlation with posted rate ({S('quote_signal','posted_rate'):.3f})."),
    ("14_posted_rate_vs_market_index", Mk, "Posted rate vs market index", "Market index against posted rate.", f"Weak raw correlation with posted rate ({S('market_index','posted_rate'):.3f}) but {S('market_index','rate_per_mile'):.3f} with rate per mile."),
    ("15_posted_rate_vs_quote_signal", Mk, "Posted rate vs quote signal", "Quote signal against posted rate.", f"Weak raw correlation ({S('quote_signal','posted_rate'):.3f})."),
    ("11_monthly_median_rate", Tm, "Monthly median rate", "Median posted rate per month.", f"Monthly medians range from ${mm.min():,.0f} to ${mm.max():,.0f}."),
    ("13_rate_by_day_of_week", Tm, "Rate by day of week", "Posted rate by weekday.", f"Weekday medians range from ${dw.min():,.0f} to ${dw.max():,.0f}."),
    ("16_train_vs_validation_distance", TV, "Distance: train vs validation", "Distance distribution in the development and assessment sets.", f"Mean distance {tr.distance.mean():,.0f} mi (development) vs {va.distance.mean():,.0f} mi (assessment)."),
    ("17_train_vs_validation_weight", TV, "Weight: train vs validation", "Weight distribution in the development and assessment sets.", f"Mean weight {tr.weight.mean():,.0f} lb vs {va.weight.mean():,.0f} lb (missing values excluded)."),
]
def img(n):
    p = P("outputs", "eda", n + ".png")
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.is_file() else None
eda = [dict(id=e[0], cat=e[1], title=e[2], what=e[3], take=e[4], img=img(e[0])) for e in eda]

# ---- technical details ------------------------------------------------------
tree = {d: sorted(p.name for p in P(d).iterdir() if p.is_file()) for d in ["data", "src", "tests", "artifacts", "reports", "scorer_results"]}
tree.update({f"outputs/{s.name}": sorted(p.name for p in s.iterdir() if p.is_file()) for s in sorted(P("outputs").iterdir()) if s.is_dir()})
order = ["data_audit", "eda", "data_cleaning", "features", "validation", "experiments", "ridge_tuning", "error_analysis", "final_pipeline", "predict_validation", "predict_december", "create_report"]
tech = dict(tree=tree, reqs=P("requirements.txt").read_text().split(), cmds=[f"python src/{s}.py" for s in order if P("src", s + ".py").is_file()] +
            ["python score.py --predictions validation_predictions.csv --december-predictions outputs/predictions/december_predictions.csv"], meta={k: v for k, v in meta.items() if k != "features"})


# ---- production ML (additive; reads JSON/JSONL only, never unpickles a model) -------------------
sys.path.append(str(ROOT / "src"))
try:
    from production.config import get_settings
    from production.model_registry import list_models, read_metadata, read_production_record
    from production.monitoring import summarize_predictions
    from production.retraining import read_status

    def _jload(path):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except Exception:
            return None

    _s = get_settings()
    _rec = read_production_record(_s.registry_path)
    _meta = read_metadata(_s.registry_path, _rec["production_version"]) if _rec else None
    _skip = ("feature_names", "reference_profile")
    prod = dict(
        registry=bool(_rec),
        record={k: v for k, v in (_rec or {}).items() if k != "history"},
        history=(_rec or {}).get("history", []),
        model=None if _meta is None else {k: v for k, v in _meta.items() if k not in _skip},
        versions=list_models(_s.registry_path),
        predictions=summarize_predictions(_s.prediction_log_path),
        drift=_jload(_s.drift_dir / "drift_report.json"),
        performance=_jload(_s.performance_dir / "performance_report.json"),
        retraining=read_status(_s.retraining_dir),
        tuning=recs(rd("outputs", "experiments", "production_tuning.csv")),
    )
except Exception as exc:  # production layer absent or broken: the assessment dashboard must still build
    prod = dict(registry=False, error=str(exc))

pt = dec.copy() if dec is not None else None
D = dict(
    k=dict(dev=len(tr), pre=len(pre), holdout=int(err.rows), assess=len(va), feats=meta["feature_count"], raw_cols=len(tr.columns), mae=err.MAE, rmse=err.RMSE, r2=fin.R2, mape=err.MAPE_percent,
           consistent=bool(abs(err.MAE - fin.MAE) < 1e-6), alpha=meta["alpha"], ho_start="2025-09-01", ho_end=str(tr.date.max().date())),
    prof=dict(train=prof(tr), val=prof(va)), hist=dict(counts=cnt.tolist(), edges=edges.round(0).tolist(), over=int((tr.posted_rate > q99).sum()), q99=q99),
    mi=dict(labels=mi.date.tolist(), vals=mi.market_index.round(3).tolist(), split=mi.s.tolist()),
    newc=dict(cities=newc, rows=int(new_mask.sum()), pct=float(new_mask.mean() * 100)),
    clean=recs(rd("outputs", "data_audit", "cleaning_diagnostic.csv")), groups=groups_out, models=recs(pd.DataFrame(models)), tune=recs(tune),
    err={k: recs(rd("outputs", "experiments", f"error_{k}.csv")) for k in ["by_distance", "by_equipment", "by_weight", "by_month", "new_city"]},
    pred=dict(cols=["id", "pickup", "delivery", "equipment", "distance", "weight", "date", "rate"],
              rows=json.loads(pd.DataFrame({"id": va.load_id, "p": va.pickup, "d": va.delivery, "e": va.equipment, "dist": va.distance, "w": va.weight,
                                            "dt": va.date.dt.strftime("%Y-%m-%d"), "r": (vp.predicted_rate.round(2) if vp is not None else np.nan)}).to_json(orient="values"))),
    dec=dict(rows=recs(pt), **dec_info), prod=prod, checks=checks, scorer=scorer, tech=tech, eda=eda, missing=missing,
    pre_dates=dict(start=str(pre.date.min().date()), end=str(pre.date.max().date())))

tpl = (HERE / "template.html").read_text(encoding="utf-8")
out = tpl.replace("/*__DATA__*/", "const D=" + json.dumps(D, allow_nan=False).replace("</", "<\\/") + ";")
(HERE / "dist").mkdir(exist_ok=True)
(HERE / "dist" / "index.html").write_text(out, encoding="utf-8")
print(f"Built dashboard/dist/index.html ({len(out)/1e6:.1f} MB); scorer exit={scorer['code']}; missing inputs={missing or 'none'}")
