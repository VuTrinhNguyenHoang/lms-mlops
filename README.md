# LMS MLOps

Ứng dụng MLOps cho bài toán quản lý dữ liệu học tập LMS nhằm dự đoán nguy cơ rớt môn.

Repo này được thiết kế để demo trọn vòng đời:

```text
upload prediction CSV
-> predict bằng MLflow champion model
-> lưu input/output artifact
-> tạo data/prediction drift report
-> hiển thị metric trên Prometheus/Grafana
-> upload truth CSV
-> evaluate performance drift
-> quyết định retrain
-> train candidate model
-> chỉ promote nếu candidate tốt hơn champion
```

## Trả Lời Nhanh: Khi Khởi Chạy Đã Có Champion Chưa?

Nếu chạy trên **fresh Docker volume** thì **chưa có model champion**.

Lý do: `docker compose up` chỉ khởi động hạ tầng gồm API, Prefect, MLflow, MinIO, Prometheus, Grafana. Hệ thống chưa tự train model lúc startup để tránh mỗi lần bật compose lại tạo thêm model version mới ngoài ý muốn.

Vì vậy ở lần chạy đầu tiên, cần bootstrap champion bằng:

```bash
curl -X POST http://localhost:8000/models/champion/train
```

Sau khi flow train hoàn tất, MLflow sẽ có registered model `lms-dropout-risk-model` với alias `champion`. Từ lúc đó prediction flow mới load được model.

Nếu Docker volume `mlflow-data` đã tồn tại từ lần chạy trước và đã có alias `champion`, thì không cần train lại. Nếu chạy `docker compose down -v`, volume bị xóa và phải bootstrap lại.

## Kiến Trúc

```text
FastAPI
  -> nhận upload, validate file, lưu raw artifact, trigger Prefect

Prefect flows
  -> train, predict, evaluate truth, retrain

ML core
  -> data contract, sklearn Pipeline, MLflow Registry, Evidently, rules

Storage
  -> local path cho flow đọc/ghi ổn định
  -> MinIO mirror raw files, predictions, reports, merged datasets, decisions

Observability
  -> FastAPI /metrics
  -> Prometheus scrape
  -> Grafana dashboard
```

Dependency direction được giữ dạng DAG:

```text
api -> flows/storage/core
flows -> data/features/models/drift/rules/storage
ML logic không import API layer
```

## Data Contract

Prediction CSV:

```text
ID_COLUMNS + FEATURE_COLUMNS
```

Truth CSV:

```text
ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMN
```

Prediction output:

```text
id, batch_id, risk_score, predicted_label, risk_level, model_name, model_version
```

Trong repo này:

- ID column: `id`
- Target column: `nograd`
- Reference data: `data/reference/simulated_data.csv`
- Prediction demo batch: `data/demo/prediction_batch.csv`
- Truth demo batch: `data/demo/truth_batch.csv`
- Drifted demo batch: `data/demo/drifted_prediction_batch.csv`, `data/demo/drifted_truth_batch.csv`

## Chạy Bằng Docker Compose

Yêu cầu:

- Docker
- Docker Compose plugin

Start toàn bộ stack:

```bash
docker compose up -d --build
```

Kiểm tra container:

```bash
docker compose ps
```

Kỳ vọng các service chính đều `Up`:

- `api`
- `prefect-server`
- `prefect-flows`
- `mlflow`
- `minio`
- `prometheus`
- `grafana`

Kiểm tra API health:

```bash
curl http://localhost:8000/health
```

Kỳ vọng:

```json
{"status":"ok"}
```

## Các Màn Hình Cần Mở Khi Demo

API Swagger:

```text
http://localhost:8000/docs
```

Dùng để xem và test endpoint upload/train/status.

Prefect UI:

```text
http://localhost:4200
```

Cần check:

- Deployments đã xuất hiện: `train-initial-champion`, `predict-batch`, `evaluate-truth`, `evaluate-and-maybe-retrain`, `retrain-model`
- Flow run chuyển sang `Completed` sau mỗi thao tác
- Nếu lỗi, mở flow run để xem task log

MLflow UI:

```text
http://localhost:5000
```

Cần check:

- Experiment `lms-dropout-risk`
- Registered model `lms-dropout-risk-model`
- Model version có alias `champion`
- Metrics của các model candidate

MinIO Console:

```text
http://localhost:9001
```

Login:

```text
minioadmin / minioadmin
```

Cần check bucket `lms-mlops` có các prefix:

- `raw/prediction/`
- `raw/truth/`
- `processed/predictions/`
- `processed/evaluations/`
- `reports/evidently/data_drift/`
- `training/merged/`
- `decisions/retrain/`

Prometheus:

```text
http://localhost:9090
```

Cần check:

- `Status -> Targets`
- Target `api:8000` ở trạng thái `UP`
- Sau khi chạy prediction/truth, query thử metric: `lms_data_drift_share`, `lms_retrain_decision`, `lms_model_champion_version`

Grafana:

```text
http://localhost:3000
```

Login:

```text
admin / admin
```

Cần mở dashboard:

```text
LMS MLOps / LMS MLOps Overview
```

Cần check:

- Data drift share
- Drifted feature count
- Prediction score drift
- Classification recall/F1/precision
- False negative count
- Retrain decision
- Promotion decision
- Champion model version

## Luồng Test 1: Bootstrap Champion

Chạy:

```bash
curl -X POST http://localhost:8000/models/champion/train
```

Kết quả API trả về dạng:

```json
{
  "status": "accepted",
  "flow_run_id": "...",
  "training_path": "/app/data/reference/simulated_data.csv"
}
```

Sau đó kiểm tra:

1. Prefect UI: flow `train-initial-champion` completed.
2. MLflow UI: model `lms-dropout-risk-model` có alias `champion`.
3. Sau bước prediction đầu tiên, `/metrics` sẽ có `lms_model_champion_version` vì metric này ghi nhận champion version mà prediction flow đã dùng.

Nếu chưa bootstrap champion mà upload prediction ngay, prediction flow sẽ lỗi vì chưa có `models:/lms-dropout-risk-model@champion`.

## Luồng Test 2: Batch Bình Thường

Upload prediction batch:

```bash
curl -X POST http://localhost:8000/batches/prediction \
  -F "batch_id=demo-001" \
  -F "file=@data/demo/prediction_batch.csv"
```

Kiểm tra flow:

```bash
curl http://localhost:8000/batches/demo-001
```

Kỳ vọng sau khi Prefect flow completed:

- `raw_prediction.exists = true`
- `prediction_output.exists = true`
- `data_drift_report.exists = true`

Tải prediction output:

```bash
curl http://localhost:8000/batches/demo-001/predictions
```

Mở drift report HTML:

```text
http://localhost:8000/batches/demo-001/drift-report
```

Upload truth batch:

```bash
curl -X POST http://localhost:8000/batches/truth \
  -F "batch_id=demo-001" \
  -F "file=@data/demo/truth_batch.csv"
```

Kiểm tra evaluation:

```bash
curl http://localhost:8000/batches/demo-001/evaluation
```

Kỳ vọng với batch thường:

- Có metrics như `accuracy`, `precision_risk`, `recall_risk`, `f1_risk`
- `retrain_decision` thường là `false`
- Grafana cập nhật performance metrics

## Luồng Test 3: Batch Có Drift

Upload prediction batch drifted:

```bash
curl -X POST http://localhost:8000/batches/prediction \
  -F "batch_id=drift-001" \
  -F "file=@data/demo/drifted_prediction_batch.csv"
```

Upload truth batch drifted:

```bash
curl -X POST http://localhost:8000/batches/truth \
  -F "batch_id=drift-001" \
  -F "file=@data/demo/drifted_truth_batch.csv"
```

Kiểm tra batch status:

```bash
curl http://localhost:8000/batches/drift-001
```

Kiểm tra evaluation:

```bash
curl http://localhost:8000/batches/drift-001/evaluation
```

Kiểm tra retrain decision:

```bash
curl http://localhost:8000/batches/drift-001/retrain
```

Kỳ vọng với batch drift:

- `data_drift_detected = true`
- `retrain_decision = true`
- Retrain flow tạo merged training dataset
- Candidate model được train
- `promotion_decision` chỉ `true` nếu candidate tốt hơn champion theo rule

Trên Grafana cần thấy các panel drift/retrain thay đổi rõ hơn so với batch thường.

## Endpoint Chính

```text
GET  /health
GET  /metrics
POST /models/champion/train
POST /batches/prediction
POST /batches/truth
POST /batches/{batch_id}/retrain
GET  /batches/{batch_id}
GET  /batches/{batch_id}/predictions
GET  /batches/{batch_id}/drift-report
GET  /batches/{batch_id}/evaluation
GET  /batches/{batch_id}/retrain
```

## Runtime Artifact

Local runtime folders:

```text
storage/
outputs/
mlruns/
```

Các folder này được ignore vì là dữ liệu sinh ra khi chạy demo.

Trong Docker Compose:

- `storage/` giữ raw upload local cho flow
- `outputs/` giữ prediction/evaluation/report local
- MinIO giữ object mirror
- Docker volumes giữ state của MLflow, Prefect, Prometheus, Grafana, MinIO

## Reset Demo

Dừng stack nhưng giữ volume:

```bash
docker compose down
```

Dừng stack và xóa toàn bộ volume:

```bash
docker compose down -v
```

Sau khi dùng `down -v`, champion model trong MLflow cũng mất. Cần chạy lại:

```bash
curl -X POST http://localhost:8000/models/champion/train
```

Muốn xóa output local:

```bash
rm -rf outputs storage mlruns
```

## Local Development

Tạo virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Tạo lại demo data nếu cần:

```bash
python scripts/generate_demo_data.py
```

Run training flow local:

```bash
PYTHONPATH=src python -m flows.train_flow
```

Run API local:

```bash
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Với demo đầy đủ, ưu tiên Docker Compose vì service DNS, MLflow, Prefect, MinIO, Prometheus và Grafana đã được nối sẵn.

## Repo Map

```text
src/core          shared config and contracts
src/data          CSV loading, validation, split, retrain dataset builder
src/features      sklearn feature transformer
src/models        model factory, train, predict, evaluate, MLflow registry
src/drift         Evidently data drift and performance drift
src/rules         retrain and promotion rules
src/storage       local paths and MinIO artifact mirror
src/flows         Prefect flows
src/api           FastAPI upload/status layer
src/monitoring    Prometheus metrics rendering
monitoring/       Prometheus and Grafana config
data/             deterministic demo/reference CSVs
scripts/          repo utility scripts
```
