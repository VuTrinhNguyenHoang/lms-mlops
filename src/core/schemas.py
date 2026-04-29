from pydantic import BaseModel


class ArtifactStatus(BaseModel):
    path: str
    exists: bool


class BatchStatus(BaseModel):
    batch_id: str
    status: str
    artifacts: dict[str, ArtifactStatus]
    retrain_decision: bool | None = None
    promotion_decision: bool | None = None
