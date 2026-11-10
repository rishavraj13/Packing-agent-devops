# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
LABEL org.opencontainers.image.title="Smart Packing List Agent" \
    org.opencontainers.image.description="CLI packing-list agent with optional observability endpoints" \
    org.opencontainers.image.source="https://github.com/AbhinavKumar2124/Packing-List-Agent"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data
WORKDIR /app
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app --home /app app && mkdir /data && chown app:app /data
COPY --from=builder /install /usr/local
COPY app ./app
COPY cli.py ./
USER app
VOLUME ["/data"]
ENTRYPOINT ["python", "cli.py"]
