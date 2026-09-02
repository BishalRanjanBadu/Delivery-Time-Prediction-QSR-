# One image serves both FastAPI (8000) and Streamlit (8501), per the v3 stack.
# No credentials baked in — AWS auth comes from the pod IAM role (IRSA) at runtime.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels
COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000 8501
# ROLE=api | app | both (default). K8s sets ROLE per container/probe needs.
ENV ROLE=both
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
ENTRYPOINT ["./entrypoint.sh"]
