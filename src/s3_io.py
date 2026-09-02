"""
S3 I/O layer for the Zomato delivery-time production pipeline.

All persistence goes through S3 via boto3 — no local filesystem as the source of
truth. Credentials come from the environment / AWS credential chain, never from
code. Populate them via .env (see .env.example) or the container's IAM role.
"""
import io
import os
import json
import logging

import boto3
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# Resolved from environment. In EKS/Lambda these come from the pod/function IAM
# role and BUCKET/REGION env vars; locally they come from .env.
BUCKET = os.environ.get("S3_BUCKET", "")
REGION = os.environ.get("AWS_REGION", "ap-south-1")


def get_client():
    """Build an S3 client. Relies on the standard AWS credential chain
    (env vars, shared config, or instance/pod role). No keys in code."""
    return boto3.client("s3", region_name=REGION)


def _bucket(bucket: str | None) -> str:
    # Resolve at call time so env vars set after import (tests, late config)
    # are honoured; fall back to the import-time value.
    b = bucket or os.environ.get("S3_BUCKET", "") or BUCKET
    if not b:
        raise RuntimeError(
            "S3 bucket not set. Provide S3_BUCKET in the environment "
            "(see .env.example) or pass bucket= explicitly."
        )
    return b


def read_csv_s3(key: str, bucket: str | None = None) -> pd.DataFrame:
    b = _bucket(bucket)
    obj = get_client().get_object(Bucket=b, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info("Loaded s3://%s/%s shape=%s", b, key, df.shape)
    return df


def save_csv_s3(df: pd.DataFrame, key: str, bucket: str | None = None) -> None:
    b = _bucket(bucket)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    get_client().put_object(Bucket=b, Key=key, Body=buf.getvalue())
    logger.info("Saved s3://%s/%s shape=%s", b, key, df.shape)


def save_pickle_s3(obj, key: str, bucket: str | None = None) -> None:
    """Persist any Python object (fitted preprocessor, model, metrics) to S3."""
    b = _bucket(bucket)
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    get_client().put_object(Bucket=b, Key=key, Body=buf.getvalue())
    logger.info("Saved s3://%s/%s", b, key)


def load_pickle_s3(key: str, bucket: str | None = None):
    b = _bucket(bucket)
    obj = get_client().get_object(Bucket=b, Key=key)
    return joblib.load(io.BytesIO(obj["Body"].read()))


def save_json_s3(payload: dict, key: str, bucket: str | None = None) -> None:
    b = _bucket(bucket)
    get_client().put_object(
        Bucket=b, Key=key, Body=json.dumps(payload, indent=2, default=str)
    )
    logger.info("Saved s3://%s/%s", b, key)


def load_json_s3(key: str, bucket: str | None = None) -> dict:
    b = _bucket(bucket)
    obj = get_client().get_object(Bucket=b, Key=key)
    return json.loads(obj["Body"].read())


def save_bytes_s3(data: bytes, key: str, content_type: str = "application/octet-stream",
                  bucket: str | None = None) -> None:
    """Persist raw bytes (e.g. an HTML drift report) to S3."""
    b = _bucket(bucket)
    get_client().put_object(Bucket=b, Key=key, Body=data, ContentType=content_type)
    logger.info("Saved s3://%s/%s", b, key)
