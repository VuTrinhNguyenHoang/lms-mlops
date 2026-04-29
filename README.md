# LMS MLOps

Hệ thống MLOps cho bài toán dự đoán nguy cơ rớt môn từ dữ liệu học tập LMS.

Mục tiêu chính:

- Upload CSV `ID + FEATURES` để dự đoán risk score/risk label.
- Lưu input/output artifacts.
- Theo dõi data drift, prediction drift và performance metrics trên Grafana.
- Upload truth CSV `ID + FEATURES + TARGET` để đánh giá mô hình.
- Tự quyết định retrain khi có drift đủ điều kiện.
- Chỉ promote model mới khi candidate tốt hơn champion theo rule đã chọn.

## Tổng Quan Luồng

```text
prediction CSV
-> FastAPI nhận file và lưu raw artifact
-> Prefect chạy prediction flow
-> MLflow load champion model
-> sklearn Pipeline predict_proba
-> lưu prediction output
-> Evidently tính drift
-> Prometheus/Grafana hiển thị metric

truth CSV
-> FastAPI nhận file truth
-> Prefect chạy evaluation flow
-> join truth với prediction theo ID + batch_id
-> tính performance metrics
-> quyết định retrain
-> train candidate model
-> promote nếu candidate vượt champion
```

## Data Contract

Prediction input:

```text
ID_COLUMNS + FEATURE_COLUMNS
```

Truth input:

```text
ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMN
```

Prediction output:

```text
id, batch_id, risk_score, predicted_label, risk_level, model_name, model_version
```

Trong hệ thống hiện tại:

- ID column: `id`
- Target column: `nograd`
- Reference training data: `data/reference/simulated_data.csv`
- Prediction batch thường: `data/demo/prediction_batch.csv`
- Truth batch thường: `data/demo/truth_batch.csv`
- Prediction batch có drift: `data/demo/drifted_prediction_batch.csv`
- Truth batch có drift: `data/demo/drifted_truth_batch.csv`

## Cấu Trúc Thư Mục

```text
.
├── data/
│   ├── demo/                  # CSV mẫu cho prediction/truth/drift
│   └── reference/             # dữ liệu reference để train champion ban đầu
├── monitoring/
│   ├── grafana/               # dashboard + provisioning datasource
│   └── prometheus/            # Prometheus scrape config
├── scripts/
│   └── generate_demo_data.py  # tạo lại bộ CSV mẫu
├── src/
│   ├── api/                   # FastAPI endpoint layer
│   ├── core/                  # config, schema, data contract
│   ├── data/                  # load/validate/split/build retrain dataset
│   ├── drift/                 # data drift và performance drift
│   ├── features/              # sklearn feature transformer
│   ├── flows/                 # Prefect flows
│   ├── models/                # train, predict, evaluate, MLflow registry
│   ├── monitoring/            # Prometheus metric rendering
│   ├── rules/                 # retrain và promotion rules
│   └── storage/               # local/MinIO artifact storage
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Runtime folders sinh ra khi chạy:

```text
outputs/    # prediction, evaluation, drift reports, retrain outputs
storage/    # raw uploaded CSV files
mlruns/     # MLflow local artifacts nếu chạy ngoài Docker
```

Các folder runtime này được ignore khỏi git.

## Framework Và Service

| Thành phần | Vai trò |
|---|---|
| FastAPI | HTTP API nhận upload CSV, validate file, lưu raw artifact, trigger Prefect deployment, trả batch/flow metadata. |
| Prefect | Orchestration cho train, predict, evaluate truth, retrain. Heavy work không nằm trong request handler. |
| MLflow | Tracking experiment, log candidate model, lưu model artifact, Model Registry, alias `champion`. |
| scikit-learn | Training/inference bằng `Pipeline` gồm `LMSFeatureBuilder` và estimator có `predict_proba`. |
| Evidently | Tính data drift/prediction drift và xuất HTML/JSON report. |
| MinIO | Object storage S3-compatible để mirror raw files, prediction outputs, reports, merged datasets, decisions. |
| Prometheus | Scrape `/metrics` từ FastAPI. |
| Grafana | Dashboard quan sát drift, model quality, retrain/promotion và champion version. |
| Docker Compose | Chạy toàn bộ stack local bằng container. |

## Kiến Trúc Dependency

Project giữ dependency theo DAG:

```text
api -> core, storage, Prefect trigger
flows -> core, storage, data, models, drift, rules, monitoring
models -> core, data, features
drift -> core
rules -> core
storage -> core
monitoring -> core
core -> standard library / pydantic
```

Nguyên tắc:

- API layer mỏng, không train/predict/drift trực tiếp.
- Prefect flow nhận path/object key, tự load dữ liệu bên trong flow.
- Model serving dùng MLflow alias `champion`.
- Model phải hỗ trợ `predict_proba`.
- Retrain rule luôn trả decision và reasons.

## Yêu Cầu Cài Đặt

- Docker
- Docker Compose plugin
- `curl`

Kiểm tra nhanh:

```bash
docker --version
docker compose version
```

## Khởi Chạy Hệ Thống

### 1. Reset môi trường

Nếu muốn chạy sạch:

```bash
docker compose down -v --remove-orphans
sudo rm -rf outputs storage mlruns
mkdir -p outputs storage
```

Lệnh này xóa Docker volumes của MLflow, Prefect, MinIO, Prometheus và Grafana.

### 2. Start toàn bộ stack

```bash
docker compose up -d --build
```

Kiểm tra container:

```bash
docker compose ps
```

Các service chính cần ở trạng thái `Up`:

- `api`
- `prefect-server`
- `prefect-flows`
- `mlflow`
- `minio`
- `prometheus`
- `grafana`

Kiểm tra API:

```bash
curl http://localhost:8000/health
```

Kết quả mong đợi:

```json
{"status":"ok"}
```

### 3. Bootstrap champion model

Fresh volume chưa có model champion. Train init bằng:

```bash
curl -X POST http://localhost:8000/models/champion/train
```

Kết quả trả về có dạng:

```json
{
  "status": "accepted",
  "flow_run_id": "...",
  "training_path": "/app/data/reference/simulated_data.csv"
}
```

Kiểm tra:

- Prefect UI có flow run `train-initial-champion` ở trạng thái `Completed`.
- MLflow có experiment `lms-dropout-risk`.
- MLflow Model Registry có model `lms-dropout-risk-model` với alias `champion`.

Lưu ý: `docker compose up -d --build` chỉ khởi động các service hạ tầng. Ở lần khởi tạo đầu tiên, hoặc sau khi chạy `docker compose down -v`, cần bootstrap champion model bằng endpoint trên trước khi chạy prediction.

### 4. Chạy batch prediction bình thường

```bash
curl -X POST http://localhost:8000/batches/prediction \
  -F "batch_id=demo-001" \
  -F "file=@data/demo/prediction_batch.csv"
```

Kiểm tra status:

```bash
curl http://localhost:8000/batches/demo-001
```

Kỳ vọng:

- `raw_prediction.exists = true`
- `prediction_output.exists = true`
- `data_drift_report.exists = true`

Xem prediction output:

```bash
curl http://localhost:8000/batches/demo-001/predictions
```

Mở drift report:

```text
http://localhost:8000/batches/demo-001/drift-report
```

### 5. Upload truth batch bình thường

```bash
curl -X POST http://localhost:8000/batches/truth \
  -F "batch_id=demo-001" \
  -F "file=@data/demo/truth_batch.csv"
```

Xem evaluation:

```bash
curl http://localhost:8000/batches/demo-001/evaluation
```

Kỳ vọng:

- Có `accuracy`, `precision_risk`, `recall_risk`, `f1_risk`.
- `retrain_decision = false` với batch bình thường.
- Grafana bắt đầu có performance metrics.

### 6. Chạy batch drift

Prediction drifted batch:

```bash
curl -X POST http://localhost:8000/batches/prediction \
  -F "batch_id=drift-001" \
  -F "file=@data/demo/drifted_prediction_batch.csv"
```

Truth drifted batch:

```bash
curl -X POST http://localhost:8000/batches/truth \
  -F "batch_id=drift-001" \
  -F "file=@data/demo/drifted_truth_batch.csv"
```

Kiểm tra evaluation:

```bash
curl http://localhost:8000/batches/drift-001/evaluation
```

Kiểm tra retrain:

```bash
curl http://localhost:8000/batches/drift-001/retrain
```

Kỳ vọng:

- `data_drift_detected = true`
- `retrain_decision = true`
- Retrain flow tạo merged training dataset.
- Candidate model được train.
- `promotion_decision` chỉ `true` nếu candidate tốt hơn champion.

## Giao Diện Cần Kiểm Tra

| UI | URL | Ghi chú |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | Test endpoint trực tiếp. |
| Prefect | http://localhost:4200 | Xem deployments và flow runs. |
| MLflow | http://localhost:5000 | Xem experiments, runs, model registry. |
| MinIO | http://localhost:9001 | Login `minioadmin / minioadmin`. |
| Prometheus | http://localhost:9090 | Kiểm tra targets và query metrics. |
| Grafana | http://localhost:3000 | Login `admin / admin`. |

Grafana dashboard:

```text
LMS MLOps / LMS MLOps Overview
```

Nếu Prefect UI báo không kết nối được `prefect-server:4200/api`, refresh cứng browser bằng `Ctrl + F5`. Compose đã cấu hình UI API URL là `http://localhost:4200/api`.

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

## Checklist

- [x] Định nghĩa data contract cho prediction/truth.
- [x] Tạo CSV mẫu/reference cố định.
- [x] Xây dựng sklearn Pipeline có `feature_builder`.
- [x] Train và register champion model qua MLflow.
- [x] Load model bằng MLflow alias `champion` khi inference.
- [x] FastAPI endpoint cho train, prediction upload, truth upload, status, output, report.
- [x] Prefect flows cho train, predict, evaluate truth, retrain.
- [x] Data drift report bằng Evidently.
- [x] Performance metrics khi upload truth.
- [x] Retrain decision rule có lý do.
- [x] Promotion rule: chỉ promote khi candidate tốt hơn champion.
- [x] Local artifact paths cho raw/prediction/evaluation/report/retrain.
- [x] MinIO mirror cho raw files, predictions, reports, merged datasets, decisions.
- [x] Prometheus metrics endpoint `/metrics`.
- [x] Grafana dashboard provisioning.
- [x] Docker Compose stack cho API, Prefect, MLflow, MinIO, Prometheus, Grafana.
- [ ] Test suite tự động cho API, flows, rules và data validation.
- [ ] CI pipeline.
- [ ] Batch/job metadata database.
- [ ] Query trạng thái flow run trực tiếp từ Prefect API trong endpoint status.
- [ ] Baseline performance drift theo champion history thay vì threshold tĩnh.
- [ ] Retrain từ 2-3 truth batches gần nhất.
- [ ] Authentication/authorization cho API và dashboards.
- [ ] Data validation sâu hơn cho dtype/range/missing values.
- [ ] Alerting rules cho Grafana/Prometheus.
