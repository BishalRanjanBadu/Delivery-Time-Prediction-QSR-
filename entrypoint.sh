#!/usr/bin/env bash
# Start FastAPI and/or Streamlit from one image based on $ROLE.
set -e
case "${ROLE:-both}" in
  api)
    exec uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 ;;
  app)
    exec streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 ;;
  both|*)
    uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000 &
    exec streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0 ;;
esac
