# Zomato Delivery-Time — HOW TO RUN (Phase 2: Deployment)

Serving path + code-push automation. Monitoring / drift / retraining are Phase 3.

## 0. Prereqs
- Python 3.11, Docker, `kubectl` + EKS access, an ECR repo `zomato-api`, an S3 bucket.
- Copy `.env.example` → `.env`, fill it in. **Never commit `.env`.** AWS auth: credential chain locally, IRSA in-cluster. No keys in code.

## 1. S3 layout used by Phase 2
```
raw/zomato_raw.csv           raw input (upload once)
models/preprocessor.pkl      fitted on train split only
models/best_model.pkl        promoted model
models/feature_names.pkl     exact training feature order
models/model_metrics.json    MAPE/MAE/R2 + training date  (promotion compares against this)
```

## 2. Local setup
```bash
pip install -r requirements.txt        # or: pip install -r requirements.lock  (exact)
export $(grep -v '^#' .env | xargs)
```

## 3. Train + promote
```bash
python src/train.py
```
Order: read raw from S3 → `split_raw()` → `preprocessor.fit_transform(train)` → `transform(test)` → 5-fold tuned XGBoost → **MAPE gate (<0.18)** → compare vs incumbent `model_metrics.json` → promote to S3 **only if gate-passing and better**. Exit code non-zero if the gate fails (CI gates on it).

## 4. Serve
```bash
# both services from one image locally:
ROLE=both ./entrypoint.sh
#   FastAPI  http://localhost:8000/docs   (/health, /model_info, /predict)
#   Streamlit http://localhost:8501
```
Serving is transform-only (loads the fitted preprocessor from S3) → no train/serve skew.

`POST /predict` body:
```json
{"records":[{"Delivery_person_Age":30,"Delivery_person_Ratings":4.7,"Road_traffic_density":"Jam","distance_km":9.0,"Time_Orderd":"19:00","Time_Order_picked":"19:10","Order_Date":"15-03-2022","Weather_conditions":"Fog","City":"Urban","Type_of_order":"Meal","Type_of_vehicle":"motorcycle","multiple_deliveries":1,"Festival":"No","Vehicle_condition":1}]}
```
Response per record: `predicted_time_min`, `eta_band` (Fast/Average/Slow), `delivery_window_min`, `shap_explanation`.

## 5. Tests (the CI gate)
```bash
pytest tests/ -v
```
Technical + **business-invariant** (Jam ≥ Low traffic; farther ≥ nearer) + **leakage regression** + **split-before-fit call-order guard** + **MAPE gate** + API end-to-end on mocked S3.

## 6. Docker
```bash
docker build -t zomato-api:local .
docker run -p 8000:8000 -p 8501:8501 --env-file .env zomato-api:local
```
No credentials in the image; AWS auth injected at runtime (env locally, IRSA in EKS).

## 7. CI/CD (`.github/workflows/mlops_pipeline.yml`)
`git push` → test → build + Trivy scan → push ECR → deploy. `develop` → UAT (rolling), `main` → prod (canary). AWS via OIDC.
Required GitHub **vars**: `AWS_REGION`, `EKS_CLUSTER`, `ECR_REGISTRY`, `S3_BUCKET`, `IRSA_ROLE_ARN`, `CURRENT_STABLE_TAG`, `PROM_URL`. Required **secret**: `AWS_CICD_ROLE_ARN`.

## 8. Prod deploy + canary
Manifests apply with canary at 0 replicas, then:
```bash
bash k8s/canary_rollout.sh
```
Ramps 10% → 50% → 100%, baking each step via `k8s/bake_monitor.py` (error rate / p95 latency / **prediction distribution** vs stable). Any breached guard → auto-rollback to 0 canary replicas. Metrics unavailable = fail closed. Fallback on model error/timeout: previous stable variant → (Phase-3) rule-based default → graceful failure, never an undefined 5xx.

## Not in Phase 2 (deferred to Phase 3)
Drift detection, SNS alerts, SageMaker retraining pipeline, MLflow tracking, CloudWatch SLA alarms, model card, bias re-check, DR, load test, cost-teardown runbook.
