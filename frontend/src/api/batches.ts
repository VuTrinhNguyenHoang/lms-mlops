import { apiClient, API_BASE_URL } from "./client";
import type {
  BatchListResponse,
  BatchSummary,
  EvaluationSummary,
  PredictionPreviewResponse,
  PredictionUploadResponse,
  RiskLevel,
  TruthUploadResponse,
} from "../types/batch";

export async function fetchBatches(): Promise<BatchListResponse> {
  const response = await apiClient.get<BatchListResponse>("/batches");
  return response.data;
}

export async function fetchBatchSummary(batchId: string): Promise<BatchSummary> {
  const response = await apiClient.get<BatchSummary>(`/batches/${batchId}/summary`);
  return response.data;
}

export async function fetchPredictionPreview(params: {
  batchId: string;
  page: number;
  pageSize: number;
  riskLevel?: RiskLevel | "all";
  query?: string;
}): Promise<PredictionPreviewResponse> {
  const response = await apiClient.get<PredictionPreviewResponse>(
    `/batches/${params.batchId}/predictions/preview`,
    {
      params: {
        page: params.page,
        page_size: params.pageSize,
        risk_level: params.riskLevel === "all" ? undefined : params.riskLevel,
        q: params.query || undefined,
      },
    },
  );
  return response.data;
}

export async function fetchEvaluationSummary(
  batchId: string,
): Promise<EvaluationSummary> {
  const response = await apiClient.get<EvaluationSummary>(
    `/batches/${batchId}/evaluation/summary`,
  );
  return response.data;
}

export async function uploadPredictionBatch(params: {
  file: File;
  batchId?: string;
}): Promise<PredictionUploadResponse> {
  const formData = new FormData();
  formData.append("file", params.file);

  if (params.batchId?.trim()) {
    formData.append("batch_id", params.batchId.trim());
  }

  const response = await apiClient.post<PredictionUploadResponse>(
    "/batches/prediction",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    },
  );
  return response.data;
}

export async function uploadTruthBatch(params: {
  batchId: string;
  file: File;
}): Promise<TruthUploadResponse> {
  const formData = new FormData();
  formData.append("file", params.file);

  const response = await apiClient.post<TruthUploadResponse>(
    `/batches/${params.batchId}/truth`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    },
  );
  return response.data;
}

export function predictionDownloadUrl(batchId: string): string {
  return `${API_BASE_URL}/batches/${batchId}/predictions`;
}
