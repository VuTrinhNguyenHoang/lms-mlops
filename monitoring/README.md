# Monitoring

Prometheus and Grafana are part of the main Docker Compose stack.

```bash
docker compose up --build
```

Open:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Grafana login: `admin / admin`

The provisioned dashboard is:

```text
LMS MLOps / LMS MLOps Overview
```

Prometheus scrapes the API through the Docker network target `api:8000`.
Runtime outputs stay in ignored local folders: `outputs/`, `storage/`, and Docker volumes.
