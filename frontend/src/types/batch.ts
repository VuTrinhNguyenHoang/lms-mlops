export type BatchStatus =
  | "processing_prediction"
  | "predicted"
  | "evaluating"
  | "evaluated"
  | "not_found";

export type RiskLevel = "high" | "medium" | "low";

export interface RiskDistribution {
  high: number;
  medium: number;
  low: number;
}

export interface BatchListItem {
  batch_id: string;
  status: BatchStatus;
  total_students: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  truth_uploaded: boolean;
  evaluated: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface BatchListResponse {
  items: BatchListItem[];
}

export interface BatchSummary {
  batch_id: string;
  status: BatchStatus;
  total_students: number;
  risk_distribution: RiskDistribution;
  truth_uploaded: boolean;
  evaluated: boolean;
}

export interface PredictionPreviewItem {
  id: string | number | null;
  risk_score: number;
  predicted_label: number;
  risk_level: RiskLevel;
}

export interface PredictionPreviewResponse {
  batch_id: string;
  page: number;
  page_size: number;
  total: number;
  items: PredictionPreviewItem[];
}

export interface EvaluationSummary {
  batch_id: string;
  truth_rows: number;
  matched_rows: number;
  matched_ratio: number;
  metrics: {
    accuracy?: number;
    precision_risk?: number;
    recall_risk?: number;
    f1_risk?: number;
    false_negative_count?: number;
  };
}

export interface PredictionUploadResponse {
  status: "accepted";
  batch_id: string;
}

export interface TruthUploadResponse {
  status: "accepted";
  batch_id: string;
  message: string;
}
