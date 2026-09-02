#!/usr/bin/env bash
# Canary rollout with bake monitoring and automatic rollback.
#
# Not a straight `kubectl set image` cutover. We bring the new version up on a
# small traffic slice (replica ratio), bake for a defined window while comparing
# the canary's error rate / latency / prediction distribution against stable,
# then ramp to 100% only if healthy — otherwise roll back to 0 canary replicas.
#
# Traffic weight ≈ canary_replicas / (canary_replicas + stable_replicas), since
# both tracks sit behind one Service. Ramp schedule: 10% -> 50% -> 100%.
#
# Requires: kubectl context set to the prod cluster; CANARY_TAG built & pushed.
set -euo pipefail

NS=prod
STABLE=qsr-eta-api-stable
CANARY=qsr-eta-api-canary
BAKE_SECONDS="${BAKE_SECONDS:-300}"          # per-step bake window
ERROR_RATE_MAX="${ERROR_RATE_MAX:-0.02}"     # 2% 5xx ceiling on canary
LATENCY_P95_MAX_MS="${LATENCY_P95_MAX_MS:-800}"
# Prediction-distribution guard: canary mean prediction must stay within this
# fraction of stable's mean (catches a model that's "up" but predicting badly).
PRED_MEAN_TOLERANCE="${PRED_MEAN_TOLERANCE:-0.15}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

rollback() {
  log "ROLLBACK: scaling canary to 0, restoring stable to full."
  kubectl -n "$NS" scale deploy "$CANARY" --replicas=0
  kubectl -n "$NS" scale deploy "$STABLE" --replicas=9
  exit 1
}
trap 'rollback' ERR

# --- health gate: canary pods must pass readiness before any traffic weight ---
promote_step() {
  local canary_replicas=$1 stable_replicas=$2 pct=$3
  log "Shifting ~${pct}% traffic: canary=${canary_replicas} stable=${stable_replicas}"
  kubectl -n "$NS" scale deploy "$CANARY" --replicas="$canary_replicas"
  kubectl -n "$NS" scale deploy "$STABLE" --replicas="$stable_replicas"
  kubectl -n "$NS" rollout status deploy "$CANARY" --timeout=120s

  log "Baking ${BAKE_SECONDS}s and evaluating canary health vs stable..."
  sleep "$BAKE_SECONDS"

  # bake_monitor.py pulls canary vs stable metrics (from Prometheus/CloudWatch and
  # the prediction log in S3) and exits non-zero if any guard is breached. The
  # ERR trap then triggers rollback.
  python3 k8s/bake_monitor.py \
    --namespace "$NS" \
    --error-rate-max "$ERROR_RATE_MAX" \
    --latency-p95-max-ms "$LATENCY_P95_MAX_MS" \
    --pred-mean-tolerance "$PRED_MEAN_TOLERANCE"
  log "Step ${pct}% healthy."
}

log "Starting canary rollout for tag ${CANARY_TAG:-<unset>}"
promote_step 1 9  10
promote_step 5 5  50
promote_step 10 0 100

# Canary is now serving 100%. Promote it to be the new stable image and reset.
NEW_IMAGE=$(kubectl -n "$NS" get deploy "$CANARY" -o jsonpath='{.spec.template.spec.containers[0].image}')
log "Canary healthy at 100%. Promoting ${NEW_IMAGE} to stable."
kubectl -n "$NS" set image deploy "$STABLE" api="$NEW_IMAGE"
kubectl -n "$NS" scale deploy "$STABLE" --replicas=9
kubectl -n "$NS" rollout status deploy "$STABLE" --timeout=180s
kubectl -n "$NS" scale deploy "$CANARY" --replicas=0
log "Rollout complete. Stable now on ${NEW_IMAGE}."
