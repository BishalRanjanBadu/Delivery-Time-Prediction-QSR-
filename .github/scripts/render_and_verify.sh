#!/usr/bin/env bash
# Render k8s manifests with envsubst and REFUSE TO APPLY if any ${VAR} was unset.
# An unset var silently yields `image: <registry>/<repo>:` (bare trailing colon),
# which k8s accepts and then fails at ImagePullBackOff minutes later.
# Usage: render_and_verify.sh k8s/deployment.yml
set -euo pipefail
WORKLOAD="${1:?usage: render_and_verify.sh <workload-manifest>}"

: "${ECR_REGISTRY:?ECR_REGISTRY is unset}"
: "${IMAGE_TAG:?IMAGE_TAG is unset}"
: "${S3_BUCKET:?S3_BUCKET is unset}"
: "${AWS_REGION:?AWS_REGION is unset}"

envsubst < k8s/config.yml  > rendered-config.yml
envsubst < "$WORKLOAD"     > rendered-workload.yml

fail=0
for f in rendered-config.yml rendered-workload.yml; do
  if grep -nE '\$\{[A-Za-z_]' "$f"; then
    echo "ERROR: unresolved \${VAR} above in $f" >&2; fail=1
  fi
  if grep -nE '^\s*image:\s*\S*:\s*$' "$f"; then
    echo "ERROR: image tag resolved to empty in $f" >&2; fail=1
  fi
done

echo "--- rendered image references ---"
grep -nE '^\s*image:' rendered-workload.yml || { echo "ERROR: no image: line found" >&2; fail=1; }

PY=$(command -v python3 || command -v python); "$PY" -c "import sys,yaml;[list(yaml.safe_load_all(open(f))) for f in sys.argv[1:]]" \
  rendered-config.yml rendered-workload.yml \
  || { echo "ERROR: rendered YAML does not parse" >&2; fail=1; }

exit "$fail"
