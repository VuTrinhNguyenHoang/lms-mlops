from typing import Any

from pydantic import BaseModel


class RiskDistribution(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0


class BatchListItem(BaseModel):
    batch_id: str
    status: str
    total_students: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    truth_uploaded: bool
    evaluated: bool
    created_at: str | None = None
    updated_at: str | None = None


class BatchListResponse(BaseModel):
    items: list[BatchListItem]


class BatchSummary(BaseModel):
    batch_id: str
    status: str
    total_students: int
    risk_distribution: RiskDistribution
    truth_uploaded: bool
    evaluated: bool


class PredictionPreviewItem(BaseModel):
    id: Any
    risk_score: float
    predicted_label: int
    risk_level: str


class PredictionPreviewResponse(BaseModel):
    batch_id: str
    page: int
    page_size: int
    total: int
    items: list[PredictionPreviewItem]


class EvaluationSummary(BaseModel):
    batch_id: str
    truth_rows: int
    matched_rows: int
    matched_ratio: float
    metrics: dict[str, Any]


class UserTruthUploadResponse(BaseModel):
    status: str
    batch_id: str
    message: str
