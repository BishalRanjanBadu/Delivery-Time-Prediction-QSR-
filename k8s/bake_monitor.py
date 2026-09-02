"""
Canary bake monitor.

Called by canary_rollout.sh after each traffic-shift step. Compares the canary
track against the stable track on three guards and exits non-zero (which trips
the rollout script's rollback) if any is breached:

  1. error rate  — canary 5xx share must stay under --error-rate-max
  2. latency     — canary p95 must stay under --latency-p95-max-ms
  3. prediction distribution — canary mean prediction must stay within
     --pred-mean-tolerance of stable's mean. This is the guard a pod-level
     health check cannot provide: a model that is "up" (200s, low latency) but
     predicting badly is caught here, not in production metrics a day later.

Metrics sources are pluggable. In this cluster:
  * error rate + latency come from Prometheus (via PROM_URL), and
  * per-track prediction stats come from the rolling prediction log the serving
    layer writes to S3 (tagged with model_version == track tag).
If a metrics source is unreachable, that is treated as a FAILED guard (fail
closed) rather than a silent pass — an unobservable canary is not a healthy one.
"""
from __future__ import annotations

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[bake] %(message)s")
log = logging.getLogger(__name__)


def _prometheus_query(expr: str) -> float | None:
    url = os.environ.get("PROM_URL")
    if not url:
        return None
    try:
        import urllib.parse
        import urllib.request
        import json

        q = urllib.parse.urlencode({"query": expr})
        with urllib.request.urlopen(f"{url}/api/v1/query?{q}", timeout=10) as r:
            data = json.load(r)
        result = data["data"]["result"]
        return float(result[0]["value"][1]) if result else 0.0
    except Exception as e:  # noqa: BLE001
        log.error("Prometheus query failed (%s): %s", expr, e)
        return None


def check_error_rate(ns: str, max_rate: float) -> bool:
    rate = _prometheus_query(
        f'sum(rate(http_requests_total{{namespace="{ns}",track="canary",status=~"5.."}}[5m]))'
        f' / sum(rate(http_requests_total{{namespace="{ns}",track="canary"}}[5m]))'
    )
    if rate is None:
        log.error("error-rate metric unavailable -> FAIL CLOSED")
        return False
    log.info("canary error rate=%.4f (max %.4f)", rate, max_rate)
    return rate <= max_rate


def check_latency(ns: str, max_ms: float) -> bool:
    p95 = _prometheus_query(
        f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket'
        f'{{namespace="{ns}",track="canary"}}[5m])) by (le))'
    )
    if p95 is None:
        log.error("latency metric unavailable -> FAIL CLOSED")
        return False
    p95_ms = p95 * 1000.0
    log.info("canary p95=%.0fms (max %.0fms)", p95_ms, max_ms)
    return p95_ms <= max_ms


def check_prediction_distribution(tolerance: float) -> bool:
    """Compare canary vs stable mean prediction from the S3 prediction log."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import s3_io
        import pandas as pd

        key = os.environ.get("PREDICTION_LOG_KEY", "monitoring/prediction_log.csv")
        df = s3_io.read_csv_s3(key)
        stable_tag = os.environ.get("STABLE_TAG", "")
        canary_tag = os.environ.get("CANARY_TAG", "")
        s_mean = df.loc[df["model_version"] == stable_tag, "prediction"].mean()
        c_mean = df.loc[df["model_version"] == canary_tag, "prediction"].mean()
        if pd.isna(s_mean) or pd.isna(c_mean):
            log.error("insufficient prediction log data -> FAIL CLOSED")
            return False
        rel = abs(c_mean - s_mean) / max(abs(s_mean), 1e-6)
        log.info("stable mean=%.2f canary mean=%.2f rel_diff=%.3f (tol %.3f)",
                 s_mean, c_mean, rel, tolerance)
        return rel <= tolerance
    except Exception as e:  # noqa: BLE001
        log.error("prediction-distribution check failed (%s) -> FAIL CLOSED", e)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="prod")
    ap.add_argument("--error-rate-max", type=float, default=0.02)
    ap.add_argument("--latency-p95-max-ms", type=float, default=800)
    ap.add_argument("--pred-mean-tolerance", type=float, default=0.15)
    args = ap.parse_args()

    guards = {
        "error_rate": check_error_rate(args.namespace, args.error_rate_max),
        "latency": check_latency(args.namespace, args.latency_p95_max_ms),
        "prediction_distribution": check_prediction_distribution(args.pred_mean_tolerance),
    }
    for name, ok in guards.items():
        log.info("guard %-24s %s", name, "PASS" if ok else "FAIL")

    if all(guards.values()):
        log.info("all guards passed")
        return 0
    log.error("one or more guards failed -> signalling rollback")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
