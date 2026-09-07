# 🛵 QSR Delivery-Time Prediction — Production MLOps

**A leakage-audited delivery-ETA regressor for quick-service restaurants, served from EKS with a tagged fallback contract, an offline-gated test suite, and three-point train/serve parity.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Model](https://img.shields.io/badge/Model-XGBoost%202.1.4-orange)
![MAPE](https://img.shields.io/badge/Holdout%20MAPE-13.36%25-success)
![Gate](https://img.shields.io/badge/gate-%3C18%25%20—%20passed-success)
![Features](https://img.shields.io/badge/features-23-informational)
![Tests](https://img.shields.io/badge/tests-16%20passing-success)
![Serving](https://img.shields.io/badge/Serving-FastAPI-009688)
![Infra](https://img.shields.io/badge/EKS-t3.small%20ap--south--2-orange?logo=amazonaws)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)

---

## What this repository actually is

Most ETA-prediction repos stop at a notebook and an R². This one carries the same model
through to a running endpoint, and the interesting engineering is in the gap between those
two things — the failure modes there are silent, not loud.

| Phase | Scope | Status |
|---|---|---|
| **1 — Experimentation** | EDA, feature engineering, model selection | signed off ([companion repo](https://github.com/BishalRanjanBadu/Delivery-Time-Prediction-QSR)) |
| **2 — Deployment** | `src/` package, 16 tests, container, EKS, CI/CD, three-point parity | **complete — signed off 2026-09-02** |
| **3 — Operate & Monitor** | drift detection, label collection, automated retraining | not started — prerequisites listed below |

Each phase is gated by a recorded sign-off listing what was reviewed, what was accepted, and
which deviations were taken knowingly. See `PHASE2_SIGNOFF.md`.

---

## ✅ What was independently verified, not asserted

The claims below are backed by executable checks in `tests/`, or by commands run against the
live system — not by prose in this file.

**Leakage boundary, enforced mechanically.** `test_train_splits_before_fit` spies on the real
call site in `train.py` and asserts `split_raw()` is invoked before any `.fit()`. Its
companion, `test_presplit_pattern_is_detectably_different`, asserts that fitting on the full
frame produces *different* scaler statistics than fitting on train — so the regression cannot
be invisible even if someone re-orders the calls.

**Algebraic target derivative dropped.** `delivery_speed` is a direct function of the target
(`Time_taken / distance`). It is removed in `LEAKAGE_COLS` before any statistic is computed.
Identifier columns (`ID`, `Delivery_person_ID`) are dropped in the same step.

**Three-point parity, to the decimal.** The same payload was sent to local `uvicorn`
(Python 3.12), `docker run` (Python 3.11), and through the EKS LoadBalancer:

| Runtime | Python | `predicted_time_min` |
|---|---|---|
| local uvicorn | 3.12 | **29.0** |
| `docker run` | 3.11 | **29.0** |
| EKS LoadBalancer | 3.11 | **29.0** |

A cross-version unpickle can corrupt fitted state — scaler means, feature order — while still
returning HTTP 200. This check is what rules that out, and it is not optional.

**The test suite runs with no cloud credentials at all.** Verified with
`env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u S3_BUCKET -u RAW_KEY pytest tests/ -v`
→ 16 passed. Data comes from a committed fixture, which removes the entire class of
"CI failed on NoSuchKey".

**Every external CI reference verified upstream, in one pass.** Action tags checked with
`git ls-remote --tags`, not recalled from memory — see [the Trivy note](#the-trivy-tag-finding).

---

## Overview

Predicts **delivery duration in minutes** (`Time_taken (min)`) for an order **before
dispatch**, so it can drive rider allocation and the ETA quoted to the customer.

| | |
|---|---|
| Target | `Time_taken (min)`, continuous |
| Problem type | Supervised regression |
| Grain | One row per order |
| Split | Random 80/20, `random_state=42` |
| Tuning | `RandomizedSearchCV`, **5-fold `KFold`**, scored on `neg_mean_absolute_error` |
| Holdout | 20% test split, scored once for reporting |
| Estimator | XGBoost 2.1.4 |

**Why a random split, not a temporal one.** Inference here is per-order, not forward-looking
over a horizon — the model scores an order against conditions known at dispatch time, with no
dependence on order sequence. If the deployment context changes to forecasting future
periods, this must become a chronological split with `TimeSeriesSplit`; it is recorded as a
declared assumption, not an oversight.

## Business problem

Quick-commerce and food-delivery platforms lose customer trust when quoted delivery times are
wrong, and lose margin when riders are allocated against a bad estimate. The usual failure
mode is a backtest that looks excellent because a post-outcome field leaked into training.
The second failure mode — less discussed — is a model that deploys cleanly and then silently
degrades because the training and serving code disagree about a dtype, a category level, or a
NaN. Both are addressed explicitly here.

## Dataset

| Property | Value |
|---|---|
| Rows | **38,964** orders |
| Columns | 22 |
| Base metrics missing | present in `multiple_deliveries`, imputed by train-fit mode |
| Sentinel values | literal `"NaN"` strings in `Road_traffic_density`, `Festival`, `Weather_conditions` — normalised, then imputed |
| Leakage columns | `delivery_speed` (target derivative) — dropped |
| Identifier columns | `ID`, `Delivery_person_ID` — dropped |

Inputs cover rider attributes (age, rating), vehicle type and condition, weather, road-traffic
density, order type, city type, festival flag, order and pickup timestamps, restaurant and
delivery coordinates, and `multiple_deliveries`.

> **The object at `raw/` was not raw.** It carried `distance_km` and `delivery_speed` already
> engineered and sentinels already stripped — it is the Phase-1 cleaned output stored under a
> `raw/` key. This mattered: it made two P0 preprocessing defects **latent on this file** and
> live only at inference or on a genuine retrain. Verify a raw object's header before trusting
> its key name.

## Methodology

| # | Stage | Does |
|---|---|---|
| 01 | Data Loading & First Look | Load from S3; sanity checks; target-leakage scan; stamp the train/test split |
| 02 | Data Cleaning | Drop identifiers and leakage columns; normalise sentinels; deterministic per-row feature engineering |
| 03 | Missing Values & Outliers | Median/mode imputation and IQR caps, **fit on train rows only** |
| 04 | Statistics & EDA | Train-rows-only analysis of distributions and segment behaviour |
| 05 | Hypothesis Testing | Group-difference tests; outcome feeds the feature list, not production code |
| 06 | Correlation, VIF & Encoding | One-hot, ordinal and binary encodings; VIF-based reduction; leakage self-check |
| 07 | Model Building & Evaluation | Baselines → tree ensembles; `RandomizedSearchCV`; SHAP; artifacts to S3 |

Stages 02, 03 and 06 are productionised into a single `DeliveryPreprocessor` in
`src/preprocess.py`. There is **one transform implementation**, used at training and at
serving — not a second copy that can drift.

## Feature engineering

Deterministic per-row work only. Nothing aggregates across rows, so nothing can leak.

**23 features** survive VIF reduction, in this exact order — the contract the serving
container reindexes every request to:

| # | Type | Features |
|---|---|---|
| 1–2 | Rider | `Delivery_person_Age`, `Delivery_person_Ratings` |
| 3 | Ordinal | `Road_traffic_density` → `{Low:0, Medium:1, High:2, Jam:3}` |
| 4–5 | Order/vehicle | `Vehicle_condition`, `multiple_deliveries` |
| 6 | Binary | `Festival` → `{No:0, Yes:1}` |
| 7 | **Derived distance** | `distance_km` — haversine from the coordinate pairs when absent |
| 8–11 | Temporal | `order_hour`, `day_of_week`, `is_weekend`, `prep_time_min` (order → pickup, midnight-wrap corrected) |
| 12–16 | One-hot weather | `Fog`, `Sandstorms`, `Stormy`, `Sunny`, `Windy` |
| 17–19 | One-hot order type | `Drinks`, `Meal`, `Snack` |
| 20–21 | One-hot vehicle | `motorcycle`, `scooter` |
| 22–23 | One-hot city | `Semi-Urban`, `Urban` |
| — | Dropped after derivation | the four raw coordinate columns |

One-hot encoding uses `drop='first'`, so `Cloudy` weather, `Buffet` order type,
`electric_scooter` and `Metropolitian` city are the reference levels. Unseen categories at
inference are absorbed by an explicit `Unknown` level rather than becoming NaN.

### Why `distance_km` is derived, not assumed

`OUTLIER_COLS` referenced `distance_km`, but the engineering step never created it — the code
silently depended on the raw file happening to carry it. Against a true source-schema file it
raised `KeyError` at `fit()`.

It is now computed by haversine from the coordinate pairs, with junk rows (either endpoint
near 0° — a known artifact of this dataset) masked so the imputer handles them rather than a
20,000 km distance entering the scaler. The raw coordinates are then dropped in favour of the
interpretable derived feature — a judgment call over mechanical VIF elimination, since the
coordinates are redundant once distance exists.

`test_distance_km_derived_when_absent` drops the column and asserts the pipeline still fits
and produces a non-null feature.

### The NaN that reached the model

Ordinal and binary maps originally ran **after** imputation. `.map(DICT)` returns `NaN` for
every unmapped value — and this dataset ships literal `"NaN "` strings (trailing space) plus
column-name prefixes like `"conditions Sunny"`. Nothing filled them.

Measured on a representative frame: **343 NaN cells in the encoded training matrix, 86 in
test.** `StandardScaler` computes its statistics ignoring NaN and then propagates it;
XGBoost accepts NaN as a missing-branch signal. Metrics still looked reasonable.

The order is now **normalise sentinels → apply fixed maps → impute**, missing categorical
levels become an explicit `Unknown` one-hot level, and `transform()` **raises** if any NaN
survives, naming the offending columns. A silent accuracy leak became a build failure.

## Models & results

**Holdout: 20% random split, scored once.**

| Metric | Value |
|---|---|
| **MAPE** | **13.36 %** |
| MAE | **3.11 min** |
| RMSE | 3.87 min |
| R² | 0.824 |
| 5-fold CV MAE | 3.12 |

The train/CV gap of ≈0.01 min indicates no meaningful overfit despite ensemble complexity.

**Promoted hyperparameters** (`RandomizedSearchCV`, 5-fold, `neg_mean_absolute_error`):

```
n_estimators 500 · max_depth 7 · learning_rate 0.03 · subsample 0.8 · colsample_bytree 0.8
```

### Quality gate

| Gate | Threshold | Actual | |
|---|---|---|---|
| Holdout MAPE | < 18 % | **13.36 %** | ✅ |
| Beats incumbent | required | first promotion | ✅ |

`train.py` exits non-zero when the gate fails, and writes artifacts to S3 **only** when the
model both passes the gate and beats the incumbent in `model_metrics.json`. A failing retrain
leaves production untouched.

> **The 18% threshold is provisional and uncosted.** It has no stated derivation — no cost of
> a one-minute error, no measured baseline from the current dispatcher process. It is labelled
> as such rather than presented as analysis, with a review trigger before the first automated
> promotion in Phase 3. A round number adopted by agreement is not a derivation.

> **On MAPE as the gate metric.** It is defensible here because the target is strictly
> positive (delivery minutes) and relative error is what operations feels. It carries a known
> asymmetry: it penalises over-prediction more heavily than under-prediction of equal
> magnitude — which, for a *quoted* ETA, arguably cuts the right way. Recorded so the choice
> is visible rather than inherited.

### When the incumbent is not a baseline

The promotion gate compares against `models/model_metrics.json`. During this build the
incumbent had been produced by the **buggy preprocessor** — the one feeding 343 NaN cells to
the scaler. A corrected pipeline that scored marginally worse on paper would have been
blocked, leaving production on the bad artifact indefinitely.

The old metrics were archived and the live file deleted so the quality gate alone governed
that promotion. **An incumbent produced by code you have since fixed is not a legitimate
baseline.**

## Key predictors (SHAP)

Mean |contribution| in **minutes**, computed with `TreeExplainer` over a 600-order sample of
the real data. Units are the target's units, so these read directly as "this feature moves the
ETA by this much on average".

| # | Feature | Mean \|SHAP\| | Direction |
|---|---|---|---|
| 1 | `Road_traffic_density` | **2.65 min** | higher → slower (r = +0.92) |
| 2 | `Delivery_person_Age` | **2.46 min** | higher → slower (r = +0.83) |
| 3 | `distance_km` | **2.43 min** | higher → slower (r = +0.78) |
| 4 | `Delivery_person_Ratings` | **2.03 min** | higher → **faster** (r = −0.82) |
| 5 | `Vehicle_condition` | **1.98 min** | higher → **faster** (r = −0.85) |
| 6 | `Weather_conditions_Sunny` | 1.28 min | sunny → **faster** (r = −0.87) |
| 7 | `multiple_deliveries` | 0.68 min | higher → slower (r = +0.78) |
| 8 | `Weather_conditions_Windy` | 0.54 min | windy → faster (r = −0.62) |

**Every direction is domain-correct.** Jam traffic slower than Low, longer distance slower,
better-rated riders and better-maintained vehicles faster, sunny weather faster. This is the
check that matters more than the ranking: a "higher → faster" on traffic or distance would be
a defect surfacing, not an insight. Two of these relationships are additionally pinned as
build-failing tests (`test_jam_traffic_slower_than_low`, `test_longer_distance_not_faster`).

### Gain and SHAP disagree, and SHAP is the one to read

| Feature | Gain rank | SHAP rank |
|---|---|---|
| `Weather_conditions_Sunny` | **1** (0.160) | 6 |
| `distance_km` | **13** (0.023) | **3** |
| `Delivery_person_Age` | 12 (0.030) | **2** |

Gain measures average improvement per split, which rewards a binary flag that cleanly
separates a subgroup and penalises a continuous feature used in many small splits.
`distance_km` sits 13th by gain and 3rd by actual contribution. Quoting gain would have
understated the single most physically obvious driver of delivery time by ten places.

## Key insights

- **Preparation time contributes almost nothing.** `prep_time_min` does not appear in the top
  15 by either measure — it ranks below `day_of_week` at under 0.08 min. This **contradicts
  the Phase-1 analysis**, which described it as the strongest predictor. The likely cause is
  that `Time_Orderd` → `Time_Order_picked` in this dataset is near-constant (a small set of
  discrete 5/10/15-minute gaps), so the column carries little variance to exploit. Recorded as
  a finding to investigate, not smoothed over.
- **Rider attributes rival physical constraints.** Age, rating and vehicle condition together
  contribute more than traffic and distance combined. For a model whose output drives rider
  allocation, that is a fairness question as much as a modelling one — see
  [Known limitations](#known-limitations).
- **Order type and vehicle type are effectively ignored.** All five `Type_of_order_*` and
  `Type_of_vehicle_*` one-hot columns fall below 0.08 min. They are retained because removing
  them changes the feature contract and invalidates the artifact, but a future retrain could
  drop them for a smaller, faster model.
- **Weather is bimodal, not graded.** `Sunny` carries 1.28 min while `Fog` carries 0.22 min —
  the model has learned "clear vs not clear" rather than a severity ordering.
- **The null-category path was never exercised by training data.** Every training row has a
  traffic value after imputation, so the branch handling a client that sends
  `"Road_traffic_density": null` only ever runs in production. It now has a dedicated test.

---

## Serving

| Route | Purpose |
|---|---|
| `GET /live` | process liveness, **no I/O** — safe for `livenessProbe` |
| `GET /health` | readiness: model, preprocessor and feature order actually load from S3 |
| `GET /model_info` | MAPE / MAE / R² / training date / model version of the promoted artifact |
| `POST /predict` | batch prediction with the full business layer |
| `GET /docs` | OpenAPI / Swagger UI |

**Liveness and readiness are deliberately separate.** Both originally pointed at `/health`,
which loads artifacts from S3 — so a transient S3 blip would restart a *healthy* pod. `/live`
is now a pure process check, and a `startupProbe` (180 s budget) covers the cold model load so
a slow first start is not a `CrashLoopBackOff`.

**Response contract:**

```json
{"model_version": "v1.1", "results": [{
  "predicted_time_min": 29.0,
  "eta_band": "Average",
  "delivery_window_min": [25.9, 32.1],
  "source": "model"
}]}
```

| Field | Meaning |
|---|---|
| `eta_band` | `Fast` (<20) / `Average` (20–34) / `Slow` (≥34) — for rider allocation and customer messaging |
| `delivery_window_min` | prediction ± the model's held-out MAE — a calibrated range, not false point-precision |
| `source` | `model` or `fallback` |

**The fallback is always tagged.** If artifacts or the model path fail, `/predict` degrades to
a documented rule-based estimate (base + distance + traffic + festival) marked
`"source": "fallback"` rather than returning an undefined 5xx. An untagged fallback is
indistinguishable from a real prediction, which is worse than an error — alerting keys off
this field.

**Send `null`, not `NaN`.** Pandas NaN is not valid JSON; FastAPI rejects it with *"Out of
range float values are not JSON compliant"*. Missing categoricals are imputed and encoded as
`Unknown`. Verified live: a payload with four null categoricals returned 17.2 min with
`"source": "model"` on all three runtimes.

## Train/serve parity

`src/preprocess.py` holds **one** `transform()`, used by `train.py` and by `predict.py`.
There is no second implementation to drift.

Artifacts are loaded once at startup and cached: `models/best_model.pkl`,
`models/preprocessor.pkl`, `models/feature_names.pkl`, `models/model_metrics.json`.
Incoming records are reindexed to the saved training feature order, missing columns filled,
extras dropped — an order or count mismatch silently corrupts predictions rather than raising.

Parity is then proven at three points on an identical payload including nulls. See
[the table above](#-what-was-independently-verified-not-asserted).

## Tests

**16 tests, all passing with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET` and
`RAW_KEY` unset.** Data resolves as local extract → committed fixture → S3 last, so the hard
gate is the one CI job with zero infrastructure dependency.

| Category | What it pins |
|---|---|
| Call-order guard | `split_raw()` precedes any `.fit()` at the real call site |
| Leakage regression | fitted statistics are unchanged when test rows are appended |
| Pre-split detectability | fitting on the full frame is provably different from fitting on train |
| Transform purity | `transform()` refits nothing |
| Derived features | `distance_km` is produced when the raw frame lacks it; raw coordinates are dropped |
| Sentinel handling | `"NaN "`, `"conditions NaN"` and unseen levels never survive as NaN |
| NaN contract | no NaN reaches the scaler or the model |
| Fallback | model failure degrades to a tagged fallback, never a 5xx |
| Business invariant | Jam traffic never predicts faster than Low; longer distance never predicts shorter |
| Sanity | metrics are not suspiciously perfect (a near-zero error means hunt for a leak) |
| Quality gate | holdout MAPE stays under the threshold |
| Serving contract | full API round-trip on mocked S3, asserting `source == "model"` |

## Architecture

```
                    ┌──────────────────────────────────────────┐
   git push ───────►│  GitHub Actions — mlops_pipeline.yml     │
                    │  test (no creds) → build → Trivy → ECR   │
                    │  → render+verify → IRSA preflight → EKS  │
                    └────────────────┬─────────────────────────┘
                                     ▼
   ┌─────────────────────────────────────────────────────────┐
   │  EKS  qsr-eta-prediction  ·  t3.small (x86_64)  ·  k8s 1.34│
   │  ┌───────────────────────────────────────────────────┐  │
   │  │ Pod × 1 — API only (ROLE=api, ENABLE_SHAP=0)      │  │
   │  │   uvicorn :8000                                    │  │
   │  │   startupProbe /health · liveness /live            │  │
   │  │   requests 150m / 384Mi   limits 500m / 768Mi      │  │
   │  └───────────────────────────────────────────────────┘  │
   │  Service (LoadBalancer) → :80 → :8000                   │
   └────────────────────────────┬────────────────────────────┘
                                │ IRSA, read-only
                                ▼
              s3://qsr-eta-prediction
                raw/zomato_raw.csv
                models/  best_model.pkl · preprocessor.pkl
                         feature_names.pkl · model_metrics.json
```

Node headroom at deploy: system pods (coredns, aws-node, kube-proxy, metrics-server) hold
**540 Mi of ~1460 Mi allocatable**, leaving ~920 Mi against a 384 Mi request. Verified with
`kubectl describe node` *before* deploying, not discovered as a `Pending` pod afterwards.

**CI does not bootstrap infrastructure.** The cluster, the OIDC provider and the IRSA
ServiceAccounts are one-time human steps that the pipeline *verifies* and never creates.

## CI/CD

| Job | Trigger | Does |
|---|---|---|
| `test` | every push + PR | pytest, 16 tests, **no AWS credentials** |
| `build` | push | docker build → Trivy scan → push to ECR (`git-SHA` + `latest`) |
| `deploy-uat` | `develop` | render+verify → IRSA preflight → apply → rollout to `uat` |
| `deploy-prod` | `main` | same, to `prod` |

**Two secrets, zero repo Variables.** Only `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
are configured by hand. Region, account, bucket, cluster, ECR repo and raw key are baked in as
`${{ vars.X || 'default' }}`, so a fresh clone runs without creating a single Variable. An
earlier revision needed six hand-created Variables and failed on each in turn.

**Render before apply.** `envsubst | kubectl apply` hides an unset variable —
`image: registry/repo:` is valid YAML and fails minutes later as `ImagePullBackOff`.
`render_and_verify.sh` renders to files, greps for unresolved `${VAR}` on **non-comment lines
only**, asserts the tag is non-empty, parses the YAML, prints the resolved image reference,
then allows the apply. Both paths are proven: exit 0 with a tag, exit 1 without.

### The Trivy tag finding

The scan gate was correctly reconfigured to be non-blocking — and pinned to
`aquasecurity/trivy-action@0.28.0`. That tag does not exist; the repo's tags are
**v-prefixed** (`v0.28.0` … `v0.36.0`). The job failed at *"Set up job"*, before a single
step ran, which looks nothing like a code problem.

The behavioural fix had been made without verifying the **reference**. Every external
reference is now checked against upstream in one pass:

```bash
git ls-remote --tags https://github.com/<owner>/<repo>.git \
  | sed 's|.*refs/tags/||' | grep -v '\^{}' | sort -V | tail
```

The scan needs **three** guards, not one: `ignore-unfixed: true` so unpatchable base-image
CVEs cannot block, `exit-code: "0"` so findings report, and `continue-on-error: true` because
Trivy pulls its DB from ghcr.io, which rate-limits in CI — without the third, an
infrastructure failure wedges the build even with the first two set.

## Tech stack

`Python 3.11` · `XGBoost 2.1.4` · `scikit-learn 1.5.2` · `statsmodels 0.14.4` · `SHAP 0.46.0`
· `pandas 2.2.3` · `numpy 1.26.4` · `FastAPI 0.115.6` · `uvicorn` · `pytest` · `moto` ·
`Docker` · `AWS S3 · ECR · EKS · IAM/IRSA` · `GitHub Actions`

Serving pins match the artifact's training environment, and every one was verified published
on PyPI before being written. `uvicorn` is pinned **without** `[standard]` — the extras pull
`uvloop`, which has no Windows wheel and made the lockfile uninstallable on the development
machine.

## Repository structure

```
├── src/
│   ├── preprocess.py       THE transform — fit (train only) / transform (everywhere)
│   ├── train.py            split → fit → tune → gate → compare incumbent → promote
│   ├── predict.py          transform-only inference, business layer, tagged fallback
│   ├── api.py              FastAPI — /live /health /model_info /predict
│   ├── app.py              Streamlit UI (not installed in the serving image)
│   └── s3_io.py            default credential chain only — never explicit keys
├── tests/
│   ├── test_model.py       16 tests — the CI hard gate
│   └── fixtures/
│       ├── sample_raw.csv  COMMITTED — why CI needs no credentials
│       └── _make_fixture.py regenerate from the real raw object
├── k8s/
│   ├── deployment.yml      prod — slim single-track, 1 replica
│   ├── deployment-uat.yml  UAT — 1 replica
│   ├── config.yml          namespaces + ConfigMaps
│   ├── canary_rollout.sh   preserved for paid tier — see Known limitations
│   └── bake_monitor.py     canary guards (error rate / latency / prediction distribution)
├── .github/
│   ├── workflows/mlops_pipeline.yml
│   └── scripts/render_and_verify.sh
├── Dockerfile · entrypoint.sh · requirements.txt · requirements.lock
├── .env.example · .pre-commit-config.yaml
└── architecture.html · HOW_TO_RUN.md
```

## How to run

**Tests — no cloud credentials required:**

```bash
git clone https://github.com/BishalRanjanBadu/Delivery-Time-Prediction-QSR-.git
cd Delivery-Time-Prediction-QSR-
python -m venv venv && source venv/Scripts/activate    # venv/bin/activate on Linux/macOS
pip install -r requirements.txt

env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u S3_BUCKET -u RAW_KEY pytest tests/ -v
```

Install from `requirements.txt`, **not** `requirements.lock` — the lock is a Linux image
artifact and legitimately contains packages with no Windows wheels.

**Serve locally** (requires AWS credentials with read access to the model bucket):

```bash
aws configure                                          # keys land in ~/.aws/, never in the repo
export S3_BUCKET=qsr-eta-prediction AWS_REGION=ap-south-2 ENABLE_SHAP=0
ROLE=api ./entrypoint.sh                               # http://127.0.0.1:8000/docs
```

`.env` carries configuration only — **never credentials**. Environment variables take
precedence over `~/.aws/credentials`, so a stale key in `.env` silently overrides a freshly
configured one.

**Predict:**

```bash
curl -sS -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "records": [{
    "Delivery_person_Age": 30, "Delivery_person_Ratings": 4.7,
    "Restaurant_latitude": 22.745, "Restaurant_longitude": 75.892,
    "Delivery_location_latitude": 22.765, "Delivery_location_longitude": 75.912,
    "Order_Date": "15-03-2022", "Time_Orderd": "19:00", "Time_Order_picked": "19:10",
    "Weather_conditions": "Fog", "Road_traffic_density": "Jam", "Vehicle_condition": 1,
    "Type_of_order": "Meal", "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1, "Festival": "No", "City": "Urban", "distance_km": 9.0
  }]}'
```

**Retrain and promote:**

```bash
python src/train.py
```

Writes artifacts to S3 **only** if the gate passes and the incumbent is beaten. A failing
retrain leaves production untouched and exits non-zero.

**Deploy:** the operator runbook covers the full sequence — local setup → tests → train →
serve → Docker → ECR → EKS (with the free-tier instance-type pre-check) → namespaces →
IRSA → deploy → verify live → CI/CD → teardown, each step with a verification command.

## Known limitations

- **Canary deferred.** A replica-ratio canary needs ≥2 pods across tracks, and one free-tier
  `t3.small` has ~920 Mi free after system pods. Production runs slim single-track at 1
  replica. `canary_rollout.sh` and the canary manifest are preserved. **On real infrastructure
  the canary is not optional.**
- **The bake monitor has no input.** `bake_monitor.py` reads `monitoring/prediction_log.csv`,
  which nothing writes — so its prediction-distribution guard would fail closed on every
  deploy. Restoring the canary requires prediction logging first.
- **No inference logging, and the IRSA policy is read-only** (`s3:GetObject` on `models/*`
  and `raw/*`). Phase 3 must enable both **in the same change** — one without the other yields
  `AccessDenied` per request.
- **No drift baseline.** `train.py` writes four artifacts; `reference/training_reference.*` is
  not one of them. Drift has nothing to compare against.
- **No `prediction_id`**, so predictions cannot be joined to ground truth. Only input drift is
  observable; performance decay is not.
- **No versioned data contract.** Schema validation is implicit in the Pydantic request model
  rather than a `contracts/schema_v1.json` shared by training, serving and the drift job.
- **`Delivery_person_Age` is used as a feature, and it is the #2 driver** (2.46 min, older →
  slower). No protected-attribute decision was recorded in Phase 1, no proxy analysis was run,
  and no fairness metric is measured by group. Where model output influences rider allocation
  or pay, age is a protected characteristic in most employment regimes and its use needs a
  stated legal basis — or removal. **This is the largest open governance item**, and it is a
  compliance decision, not a modelling one.
- **Static IAM keys in CI**, not GitHub OIDC. OIDC is the better pattern and the intended
  upgrade; stated explicitly rather than claimed otherwise.
- **SHAP disabled in-cluster.** `shap` + `numba` + `llvmlite` import ≈250 Mi RSS against a
  768 Mi limit. Available locally and on larger nodes.
- **Interchange format is CSV, not Parquet.** Recorded as a deviation to fix, not a preference.
- **Model artifacts are mutable.** `models/best_model.pkl` is overwritten on promotion, so
  there is no versioned rollback path — an image rollback would reload the same artifact. An
  immutable `models/v_<utc>_<sha>/` registry with a `CURRENT.json` alias is the Phase-3 fix.

## Costs

| Resource | Cost |
|---|---|
| EKS control plane | ~$0.10/hour (**~$73/month — never free tier**) |
| `t3.small` node | free-tier dependent |
| Classic ELB (one per `type: LoadBalancer` Service) | ~$16/month |
| S3 + ECR | negligible at this volume |

Roughly ₹600–800/day all-in while running. **Delete the `type: LoadBalancer` Service before
the cluster** or the ELB is orphaned and keeps billing:

```bash
kubectl -n prod delete svc qsr-eta-api
eksctl delete cluster --name qsr-eta-prediction --region ap-south-2
aws eks list-clusters --region ap-south-2               # verify empty
```

UAT is deliberately dormant behind its `develop` branch gate — two namespaces would mean two
ELBs billing simultaneously and memory contention on one node.

## Author

**Bishal Ranjan Badu**
Data Science · Machine Learning · MLOps
