#!/usr/bin/env bash
# Render k8s manifests with envsubst and REFUSE TO APPLY if anything is unresolved.
# An unset var silently yields `image: <registry>/<repo>:` (bare trailing colon),
# which k8s accepts and then fails at ImagePullBackOff minutes later.
#
# Portable across the GitHub runner (python3) and Git Bash on Windows (python).
# The YAML parse check is skipped with a warning if PyYAML is unavailable rather
# than failing the deploy on a missing dev dependency.
#
# Usage: render_and_verify.sh k8s/deployment.yml
set -euo pipefail
WORKLOAD="${1:?usage: render_and_verify.sh <workload-manifest>}"

: "${ECR_REGISTRY:?ECR_REGISTRY is unset}"
: "${IMAGE_TAG:?IMAGE_TAG is unset}"
: "${S3_BUCKET:?S3_BUCKET is unset}"
: "${AWS_REGION:?AWS_REGION is unset}"

envsubst < k8s/config.yml > rendered-config.yml
envsubst < "$WORKLOAD"    > rendered-workload.yml

fail=0
for f in rendered-config.yml rendered-workload.yml; do
  # Only flag ${VAR}-shaped tokens on non-comment lines. A literal "${...}" in a
  # comment is documentation, not an unresolved substitution.
  if grep -nE '^[^#]*\$\{[A-Za-z_][A-Za-z0-9_]*\}' "$f"; then
    echo "ERROR: unresolved \${VAR} above in $f" >&2; fail=1
  fi
  if grep -nE '^[[:space:]]*image:[[:space:]]*\S*:[[:space:]]*$' "$f"; then
    echo "ERROR: image tag resolved to empty in $f" >&2; fail=1
  fi
done

echo "--- rendered image references ---"
grep -nE '^[[:space:]]*image:' rendered-workload.yml \
  || { echo "ERROR: no image: line found" >&2; fail=1; }

PY="$(command -v python3 || command -v python || true)"
if [ -n "$PY" ] && "$PY" -c "import yaml" >/dev/null 2>&1; then
  "$PY" -c "import sys,yaml;[list(yaml.safe_load_all(open(f))) for f in sys.argv[1:]]" \
    rendered-config.yml rendered-workload.yml \
    || { echo "ERROR: rendered YAML does not parse" >&2; fail=1; }
  echo "YAML parse OK"
else
  echo "WARN: PyYAML unavailable - skipping parse check (kubectl will still validate)" >&2
fi

exit "$fail"