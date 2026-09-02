# QSR Delivery-Time Prediction — Production MLOps

**A leakage-safe delivery-ETA regressor, served from EKS and redeployed on every push.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost%202.1.4-orange)
![Serving](https://img.shields.io/badge/Serving-FastAPI%20%C2%B7%20Docker%20%C2%B7%20EKS-009688?logo=fastapi&logoColor=white)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)
![Phase](https://img.shields.io/badge/Phase%202-deployed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Scope.** This repository is the **production system**: preprocessing, training,
> serving, container, Kubernetes manifests, and the CI/CD pipeline. The
> exploratory analysis lives in the companion notebook repository.
> Drift detection, automated retraining, and SLA alarms are **Phase 3** and are
> deliberately not implemented here — see [Roadmap](#roadmap).

---

## Table of Contents

- [What this does](#what-this-does)
- [Architecture](#architecture)
- [Results](#results)
- [The leakage boundary](#the-leakage-boundary)
- [Correctness guarantees](#correctness-guarantees)
- [API](#api)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Design decisions](#design-decisions)
- [Cost](#cost)
- [Roadmap](#roadmap)
- [Author](#author)

---

## What this does

Predicts `Time_taken (min)` for a food-delivery order before dispatch, and
returns the number alongside the business layer operations actually consumes:

| Output | Purpose |
|---|---|
| `predicted_time_min` | The point estimate |
| `eta_band` | `Fast` / `Average` / `Slow` bucket for rider allocation and customer messaging |
| `delivery_window_min` | Prediction ± the model's held-out MAE — a calibrated range, not false point-precision |
| `source` | `model` or `fallback` — makes degraded mode visible to callers and alerting |
| `shap_explanation` | Top per-prediction feature contributions (optional; off on memory-constrained nodes) |

Trained on ~39,000 historical orders covering rider attributes, vehicle
condition, weather, traffic density, order type, city type, festival flag, and
order/pickup timestamps.

---

## Architecture

Two independent automation loops. **Loop 1 is live; Loop 2 is Phase 3.**

```
LOOP 1 — CODE  (implemented)
  git push ──▶ GitHub Actions
                 ├─ test    pytest, 16 tests, no AWS credentials required
                 ├─ build   docker build → Trivy scan (report-only) → push to ECR
                 └─ deploy  render+verify manifests → IRSA preflight → kubectl apply → rollout

LOOP 2 — DATA  (Phase 3, not implemented)
  EventBridge (daily) ──▶ Lambda ──▶ detect_drift ──▶ SageMaker retrain
                                                   └─▶ compare + promote ──▶ Loop 1
```

**Runtime path.** One Docker image → ECR → EKS behind a LoadBalancer. Pods
obtain S3 credentials via **IRSA** — no long-lived key exists anywhere in the
cluster. Model, preprocessor, feature order, and metrics are all loaded from S3
at startup and cached.

**S3 is the single source of truth.** No local filesystem data layer.

```
s3://<bucket>/
  raw/zomato_raw.csv          training input
  models/best_model.pkl       promoted model
  models/preprocessor.pkl     fitted on the train split only
  models/feature_names.pkl    exact training feature order
  models/model_metrics.json   metrics + training date (governs promotion)
```

> **CI does not bootstrap infrastructure.** The EKS cluster, the OIDC provider,
> and the IRSA ServiceAccounts are one-time manual steps that the pipeline
> *verifies* but does not create. See `RUNBOOK.md`.

---

## Results

Held-out test set, 20% split, XGBoost tuned via `RandomizedSearchCV` with 5-fold CV.

| Metric | Value |
|---|---|
| MAPE | **13.36 %** |
| MAE | **3.11 min** |
| RMSE | 3.87 min |
| R² | 0.824 |
| 5-fold CV MAE | 3.12 (train/CV gap ≈ 0.01 → no meaningful overfit) |

**Quality gate:** MAPE < 0.18. Training exits non-zero if the gate fails, and
artifacts are written to S3 **only** when the model both passes the gate and
beats the incumbent in `model_metrics.json`. A failing retrain leaves production
untouched.

### Parity verification

The same payload was sent to three runtimes and compared to the decimal:

| Runtime | Python | `predicted_time_min` |
|---|---|---|
| local uvicorn | 3.12 | 29.0 |
| `docker run` | 3.11 | 29.0 |
| via EKS LoadBalancer | 3.11 | 29.0 |

Identical across all three is the evidence that the artifact, the runtime, and
the serving path agree. A cross-version unpickle can silently corrupt fitted
state (scaler means, feature order) while still returning HTTP 200 — this check
is what rules that out.

---

## The leakage boundary

Every statistic that transforms data — imputation medians and modes, IQR outlier
bounds, one-hot categories, VIF selection, the scaler — is **fit on the training
split only** and then applied to test, inference, and every retrain.
`transform()` fits nothing.

A fit/transform signature alone does not guarantee this; the **call site** has to
split first. `train.py` does:

```python
train_df, test_df = split_raw(raw)   # split FIRST
pre.fit_transform(train_df)          # learns on train only
pre.transform(test_df)               # applies to test
```

Two tests enforce it and fail the build if the order is ever inverted:

- `test_train_splits_before_fit` — spies on the real call site and asserts
  `split()` is invoked before `fit()`
- `test_presplit_pattern_is_detectably_different` — asserts that fitting on the
  full frame produces *different* scaler means than fitting on train, so the
  regression cannot be invisible

`delivery_speed` (an algebraic function of the target) is dropped as a leakage
column before any statistic is computed.

---

## Correctness guarantees

Sixteen tests gate every build. Beyond the usual technical checks, these encode
failure modes that are silent rather than loud — the model stays "up" and keeps
returning plausible numbers while being wrong.

| Guarantee | Test |
|---|---|
| Statistics fit on train rows only | `test_leakage_stats_from_train_only` |
| Pre-split fitting is detectable | `test_presplit_pattern_is_detectably_different` |
| `transform()` never refits | `test_transform_does_not_refit` |
| Split precedes fit at the call site | `test_train_splits_before_fit` |
| `distance_km` is derived when the raw file lacks it | `test_distance_km_derived_when_absent` |
| Sentinel categories (`"NaN "`, unseen levels) never survive as NaN | `test_sentinel_categories_do_not_become_nan` |
| No NaN reaches the scaler or model | `test_transform_no_nulls` |
| Model failure degrades to a tagged fallback, never a 5xx | `test_predict_records_falls_back_instead_of_raising` |
| Jam traffic never predicts faster than Low | `test_jam_traffic_slower_than_low` |
| Longer distance never predicts shorter time | `test_longer_distance_not_faster` |
| Metrics are not suspiciously perfect | `test_metrics_not_suspiciously_perfect` |
| MAPE stays under the gate | `test_mape_gate` |
| Full API round-trip on mocked S3 | `test_api_end_to_end` |

**The test job runs with no cloud credentials.** Data resolves as local extract →
committed fixture (`tests/fixtures/sample_raw.csv`) → S3 as a last resort. This
makes the hard gate the one job with zero infrastructure dependency. Verify
locally under CI conditions:

```bash
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u S3_BUCKET -u RAW_KEY pytest tests/ -v
```

### Preprocessing order

Three rules that are not leakage but corrupt predictions just as quietly:

1. **Sentinels are normalised before mapping.** The source export writes literal
   `"NaN"` strings and column-name prefixes (`"conditions Sunny"`). These become
   real NaN first.
2. **Fixed maps run before imputation.** `.map(DICT)` returns NaN for any
   unmapped value; if mapping ran after imputation nothing would fill them, and
   `StandardScaler` computes statistics ignoring NaN and then propagates it.
3. **`transform()` asserts NaN-free**, naming the offending columns. A silent
   accuracy leak becomes a build failure.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/live` | GET | Process liveness. No S3, no model load. Used by the k8s `livenessProbe`. |
| `/health` | GET | Readiness. Confirms artifacts actually load. Used by `readinessProbe` and `startupProbe`. |
| `/model_info` | GET | Current MAPE / MAE / R² / training date / model version |
| `/predict` | POST | Batch prediction with the full business layer |
| `/docs` | GET | OpenAPI / Swagger UI |

Liveness and readiness are deliberately **separate**: pointing liveness at an
endpoint that touches S3 restarts healthy pods on a transient blip.

```bash
curl -s -X POST "http://$ENDPOINT/predict" -H "Content-Type: application/json" -d '{
  "records": [{
    "Delivery_person_Age": 30, "Delivery_person_Ratings": 4.7,
    "Restaurant_latitude": 22.745, "Restaurant_longitude": 75.892,
    "Delivery_location_latitude": 22.765, "Delivery_location_longitude": 75.912,
    "Order_Date": "15-03-2022", "Time_Orderd": "19:00", "Time_Order_picked": "19:10",
    "Weather_conditions": "Fog", "Road_traffic_density": "Jam", "Vehicle_condition": 1,
    "Type_of_order": "Meal", "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1, "Festival": "No", "City": "Urban", "distance_km": 9.0
  }]
}'
```

```json
{"model_version":"v1.1","results":[{
  "predicted_time_min": 29.0, "eta_band": "Average",
  "delivery_window_min": [25.9, 32.1], "source": "model"}]}
```

**Send `null`, not `NaN`.** Pandas NaN is not valid JSON; FastAPI rejects it with
*"Out of range float values are not JSON compliant"*. Missing categoricals are
imputed and encoded as an explicit `Unknown` level.

**Fallback contract.** If artifacts or the model path fail, `/predict` degrades
to a documented rule-based estimate tagged `"source": "fallback"` rather than
returning an undefined 5xx. An untagged fallback would be indistinguishable from
a healthy prediction — alerting keys off this field.

---

## Repository layout

```
src/
  preprocess.py         stateful preprocessor: fit (train only) / transform (everywhere)
  train.py              split → fit → tune → gate → compare incumbent → promote to S3
  predict.py            transform-only inference, business layer, fallback contract
  api.py                FastAPI: /live /health /model_info /predict
  app.py                Streamlit UI (not installed in the serving image)
  s3_io.py              shared S3 layer over boto3
tests/
  test_model.py         16 tests — the CI hard gate
  fixtures/             committed sample + regeneration script (offline CI)
k8s/
  deployment.yml        prod, slim single-track
  deployment-uat.yml    UAT, 1 replica
  config.yml            namespaces + ConfigMaps
  canary_rollout.sh     preserved for paid tier — see Design decisions
  bake_monitor.py       canary guards (error rate / latency / prediction distribution)
.github/
  workflows/mlops_pipeline.yml
  scripts/render_and_verify.sh
Dockerfile  entrypoint.sh  requirements.txt  requirements.lock
.env.example  .pre-commit-config.yaml  architecture.html  HOW_TO_RUN.md  RUNBOOK.md
```

---

## Quickstart

```bash
git clone https://github.com/BishalRanjanBadu/Delivery-Time-Prediction-QSR-.git
cd Delivery-Time-Prediction-QSR-

python -m venv venv && source venv/bin/activate     # Git Bash: venv/Scripts/activate
pip install -r requirements.txt
pytest tests/ -v                                     # 16 passing, no AWS needed
```

Install from `requirements.txt`, **not** `requirements.lock`. The lock is a Linux
image artifact and legitimately contains packages with no Windows wheels
(`nvidia-nccl-cu12` via the xgboost Linux wheel).

Train and serve against S3:

```bash
aws configure                                        # credentials live in ~/.aws/, never in the repo
export S3_BUCKET=<bucket> AWS_REGION=<region> RAW_KEY=raw/zomato_raw.csv
python src/train.py
ROLE=api ./entrypoint.sh                             # http://localhost:8000/docs
```

`.env` carries configuration only — **never credentials**. Environment variables
take precedence over `~/.aws/credentials`, so a stale key in `.env` silently
overrides a freshly configured one.

---

## Deployment

Full copy-paste sequence with verification after every step is in **`RUNBOOK.md`**:
local setup → tests → train → serve → Docker → ECR → EKS → IRSA → deploy →
verify live → CI/CD → teardown.

Two guards worth calling out:

**Render before apply.** `envsubst | kubectl apply` hides an unset variable —
`image: registry/repo:` is valid YAML and fails minutes later as
`ImagePullBackOff`. `render_and_verify.sh` renders to files, greps for unresolved
`${VAR}` on non-comment lines, asserts the image tag is non-empty, parses the
YAML, prints the resolved image reference, and only then allows the apply.

**Instance-type pre-check.** A Free-Tier-restricted account can only launch
free-tier-eligible types; others fail with no error and no instances **in every
region**, so region-hopping never helps. Check `free-tier-eligible` *and*
`instance-type-offerings` for the target AZs before creating the cluster — and
check architecture, since `t4g.*` is eligible but ARM64.

---

## CI/CD

| Job | Trigger | Does |
|---|---|---|
| `test` | every push + PR | pytest, 16 tests, **no AWS credentials** |
| `build` | push | docker build → Trivy scan → push to ECR (`git-SHA` + `latest`) |
| `deploy-uat` | `develop` | render+verify → IRSA preflight → apply → rollout to `uat` |
| `deploy-prod` | `main` | same, to `prod` |

**Zero-config.** Only two GitHub **Secrets** are required —
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Every non-secret identifier
(region, account, bucket, cluster, ECR repo, raw key) is baked in as
`${{ vars.X || 'default' }}`, so a fresh clone runs without creating a single
repo Variable.

**The security scan reports, it does not block.** Three guards prevent it wedging
the pipeline: `ignore-unfixed: true` (unpatchable base-image CVEs can't block),
`exit-code: "0"` (findings report), and `continue-on-error: true` (Trivy pulls
its DB from ghcr.io, which rate-limits in CI). Flip the first two once the base
image is on a maintained patch cadence.

**Deploy jobs pre-flight their dependencies** — required CLI tools, and the
existence of the IRSA ServiceAccount. CI verifies the SA; it does not create one.
A missing SA otherwise surfaces as a pod stuck in `ContainerCreating` with no CI
signal.

---

## Design decisions

**Canary is deferred, not omitted.** A replica-ratio canary needs ≥2 pods across
stable and canary tracks, which does not fit a single free-tier `t3.small`
(~920 Mi free after system pods). Production runs a slim single-track deployment
at 1 replica. `canary_rollout.sh` and the canary manifest are preserved for a
paid-tier node pool. Restoring canary also requires closing a known gap: the bake
monitor's prediction-distribution guard reads a prediction log that nothing
currently writes, so it would fail closed on every deploy.

**API-only serving image.** Streamlit is not installed — it costs ~150 MB the
node does not have. `entrypoint.sh` fails fast with an actionable message on
`ROLE=app`/`both` rather than crashing on `command not found`.

**SHAP is off in the cluster** (`ENABLE_SHAP=0`). The `shap` + `numba` +
`llvmlite` import is ~250 MB RSS against a 768 Mi limit. It remains available
locally and on larger nodes.

**Static keys, not OIDC.** GitHub OIDC is the better pattern and the intended
upgrade; this deployment uses static keys in encrypted Actions secrets. Stated
explicitly rather than claimed otherwise.

**UAT is dormant.** Each `type: LoadBalancer` Service bills separately. UAT stays
behind its `develop` branch gate rather than running concurrently on a single node.

**`distance_km` is derived, not assumed.** When the raw frame lacks it, it is
computed by haversine from the coordinate columns, with junk rows (either
endpoint near 0°) masked for the imputer. The raw coordinates are then dropped in
favour of the interpretable derived feature — a judgment call over mechanical
VIF-based elimination.

---

## Cost

A managed EKS cluster bills continuously — control plane plus node plus one
LoadBalancer per Service, roughly ₹600–800/day regardless of traffic. Tear down
when idle:

```bash
kubectl -n prod delete svc qsr-eta-api      # release the ELB FIRST, or it is orphaned and keeps billing
eksctl delete cluster --name <cluster> --region <region>
aws eks list-clusters --region <region>     # verify empty
```

S3 and ECR cost pennies and are the resume point.

---

## Roadmap

Phase 3, gated on Phase 2 sign-off:

- [ ] Drift detection — EventBridge → Lambda → Evidently, with KS / chi-square fallback
- [ ] SNS alerting on **every** run, drift or not (silence ≠ healthy; it may mean the job died)
- [ ] SageMaker retraining pipeline: Processing → Training → compare → promote
- [ ] MLflow experiment tracking, so any deployed model is traceable to its run
- [ ] CloudWatch SLA alarms against stated p99 latency and uptime targets
- [ ] Prediction logging — prerequisite for restoring the canary bake monitor
- [ ] Model card, bias/fairness re-check, DR scope, load test

Known open items carried forward: the serving IRSA policy is read-only, so
retraining needs a scoped `s3:PutObject` or a separate execution role.

---

## Author

**Bishal Ranjan Badu** — Data Science · Machine Learning · MLOps

Companion notebook repository: exploratory analysis, feature engineering, and
model selection.
