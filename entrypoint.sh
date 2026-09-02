#!/usr/bin/env bash
# Start FastAPI and/or Streamlit from one image based on $ROLE.
#
# The serving image is API-ONLY: streamlit is deliberately NOT in
# requirements.txt (it costs ~150 MB and the free-tier node has no headroom).
# ROLE=app / ROLE=both therefore fail fast with an actionable message instead of
# crashing on `streamlit: command not found` — this was a real prod break when a
# .env carrying ROLE=both was passed to `docker run`.
set -euo pipefail

ROLE="${ROLE:-api}"          # default is api, NOT both

need_streamlit() {
  if ! command -v streamlit >/dev/null 2>&1; then
    echo "FATAL: ROLE=${ROLE} requires streamlit, which is not installed in this image." >&2
    echo "       This image is API-only. Either set ROLE=api, or add 'streamlit' to" >&2
    echo "       requirements.txt and rebuild. Refusing to start." >&2
    exit 78   # EX_CONFIG
  fi
}

case "$ROLE" in
  api)
    exec uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 ;;
  app)
    need_streamlit
    exec streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 ;;
  both)
    need_streamlit
    uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 &
    exec streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 ;;
  *)
    echo "FATAL: unknown ROLE='${ROLE}' (expected api|app|both)." >&2
    exit 78 ;;
esac
