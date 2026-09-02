# Zomato Delivery-Time — Production (Phase 2: Deployment)

Predicts `Time_taken (min)` for delivery orders (regression). Data + model
artifacts live in S3. This phase covers the **serving path and the code-push
automation loop**; monitoring, drift, and the SageMaker retraining pipeline are
Phase 3.

## Stack (the "best of both" hybrid)
- **Serving:** FastAPI (`/predict`, `/health`, `/model_info`) + optional Streamlit UI, one Docker image → ECR → EKS (LoadBalancer, ≥2 replicas).
- **Promotion:** MLflow (Phase 3) + `models/model_metrics.json` comparison — a new model is promoted only if it passes the MAPE gate **and** beats the incumbent.
- **Deploy safety:** canary rollout with a bake monitor (error rate / latency / prediction distribution) and auto-rollback — not a plain RollingUpdate.

## Automation — Trigger 1 (code)
`git push` → `mlops_pipeline.yml`: **test → build + Trivy scan → push ECR → deploy EKS**.
`develop` → UAT namespace (straight rollout); `main` → prod (canary). AWS auth via GitHub OIDC — no static keys in the repo.

## Leakage boundary
`src/preprocess.py` fits every statistic (imputation, encoding, VIF, scaler) on the **train split only**; `.transform()` fits nothing. `train.py` splits before any fit, and `tests/test_model.py` fails the build if that order is ever violated (`test_train_splits_before_fit`, `test_presplit_pattern_is_detectably_different`).

## Layout
```
src/   preprocess.py  train.py  predict.py  api.py  app.py  s3_io.py
tests/ test_model.py  (technical + business-invariant + leakage + call-order + MAPE gate)
k8s/   deployment.yml (prod canary)  deployment-uat.yml  config.yml  canary_rollout.sh  bake_monitor.py
.github/workflows/ mlops_pipeline.yml
Dockerfile  entrypoint.sh  requirements.txt  requirements.lock  .env.example  .pre-commit-config.yaml
architecture.html  HOW_TO_RUN.md
```
See `HOW_TO_RUN.md` for local run, tests, Docker, and deploy steps.
