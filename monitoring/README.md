# Observability

This folder runs a lightweight Prometheus + Grafana stack for the local LMS MLOps demo.

## Run

Start the API on the host so Prometheus can scrape `/metrics` from inside Docker:

```bash
PYTHONPATH=src ~/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start Prometheus and Grafana:

```bash
docker compose -f docker-compose.observability.yml up
```

Open:

- Prometheus: http://localhost:9090
- Prometheus targets: http://localhost:9090/targets
- Grafana: http://localhost:3000

Grafana default login:

```text
admin / admin
```

The dashboard is provisioned under:

```text
LMS MLOps / LMS MLOps Overview
```

## Notes

- Prometheus scrapes `host.docker.internal:8000/metrics`.
- Grafana uses the Docker Compose service name `http://prometheus:9090`.
- Runtime outputs such as `outputs/`, `storage/`, and `mlruns/` should stay out of git.
