import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Download,
  ExternalLink,
  RefreshCw,
  Search,
} from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  fetchBatchSummary,
  fetchBatches,
  fetchEvaluationSummary,
  fetchPredictionPreview,
  predictionDownloadUrl,
  uploadPredictionBatch,
} from "../api/batches";
import { getErrorMessage } from "../api/client";
import EmptyState from "../components/EmptyState";
import { formatDate, formatDecimal } from "../components/format";
import LoadingBlock from "../components/LoadingBlock";
import RiskBadge from "../components/RiskBadge";
import StatusBadge from "../components/StatusBadge";
import UploadDropzone from "../components/UploadDropzone";
import type { BatchListItem, BatchStatus, EvaluationSummary } from "../types/batch";

const STATUS_OPTIONS: Array<{ value: "all" | BatchStatus; label: string }> = [
  { value: "all", label: "All statuses" },
  { value: "processing_prediction", label: "Processing" },
  { value: "predicted", label: "Predicted" },
  { value: "evaluating", label: "Evaluating" },
  { value: "evaluated", label: "Evaluated" },
];
const HISTORY_PAGE_SIZE_OPTIONS = [2, 3];
const DEFAULT_HISTORY_PAGE_SIZE = 3;
const DASHBOARD_PREVIEW_PAGE_SIZE = 6;

export default function BatchHistoryPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<"all" | BatchStatus>("all");
  const [selectedBatchId, setSelectedBatchId] = useState<string | null>(null);
  const [batchId, setBatchId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [overwrite, setOverwrite] = useState(false);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyPageSize, setHistoryPageSize] = useState(DEFAULT_HISTORY_PAGE_SIZE);
  const [previewPage, setPreviewPage] = useState(1);
  const [previewRiskLevel, setPreviewRiskLevel] = useState<"all" | "high" | "medium" | "low">("all");
  const [previewPredictedLabel, setPreviewPredictedLabel] = useState<"all" | "risk" | "safe">("all");
  const [previewSearch, setPreviewSearch] = useState("");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["batches"],
    queryFn: fetchBatches,
    refetchInterval: 4000,
  });

  const filteredItems = useMemo(() => {
    const items = data?.items ?? [];
    const normalizedQuery = query.trim().toLowerCase();

    return items.filter((item) => {
      const matchesQuery =
        !normalizedQuery || item.batch_id.toLowerCase().includes(normalizedQuery);
      const matchesStatus = status === "all" || item.status === status;
      return matchesQuery && matchesStatus;
    });
  }, [data?.items, query, status]);

  const totalHistoryPages = Math.max(1, Math.ceil(filteredItems.length / historyPageSize));
  const pagedItems = useMemo(() => {
    const startIndex = (historyPage - 1) * historyPageSize;
    return filteredItems.slice(startIndex, startIndex + historyPageSize);
  }, [filteredItems, historyPage, historyPageSize]);

  useEffect(() => {
    if (historyPage > totalHistoryPages) {
      setHistoryPage(totalHistoryPages);
    }
  }, [historyPage, totalHistoryPages]);

  useEffect(() => {
    setHistoryPage(1);
  }, [query, status, historyPageSize]);

  useEffect(() => {
    if (pagedItems.length === 0) {
      setSelectedBatchId(null);
      return;
    }

    const selectedStillVisible = pagedItems.some(
      (item) => item.batch_id === selectedBatchId,
    );

    if (!selectedBatchId || !selectedStillVisible) {
      setSelectedBatchId(pagedItems[0].batch_id);
    }
  }, [pagedItems, selectedBatchId]);

  const selectedBatch = useMemo(
    () => data?.items.find((item) => item.batch_id === selectedBatchId) ?? null,
    [data?.items, selectedBatchId],
  );

  const selectedBatchHasPrediction =
    selectedBatch?.status === "predicted" ||
    selectedBatch?.status === "evaluating" ||
    selectedBatch?.status === "evaluated";

  const summaryQuery = useQuery({
    queryKey: ["dashboard-batch-summary", selectedBatchId],
    queryFn: () => fetchBatchSummary(selectedBatchId ?? ""),
    enabled: Boolean(selectedBatchId),
    refetchInterval: (queryResult) => {
      const currentStatus = queryResult.state.data?.status;
      return currentStatus === "processing_prediction" || currentStatus === "evaluating"
        ? 3000
        : false;
    },
  });

  const previewQuery = useQuery({
    queryKey: [
      "dashboard-prediction-preview",
      selectedBatchId,
      previewPage,
      previewRiskLevel,
      previewSearch,
    ],
    queryFn: () =>
      fetchPredictionPreview({
        batchId: selectedBatchId ?? "",
        page: previewPage,
        pageSize: DASHBOARD_PREVIEW_PAGE_SIZE,
        riskLevel: previewRiskLevel,
        query: previewSearch,
      }),
    enabled: Boolean(selectedBatchId) && selectedBatchHasPrediction,
  });

  const evaluationQuery = useQuery({
    queryKey: ["dashboard-evaluation-summary", selectedBatchId],
    queryFn: () => fetchEvaluationSummary(selectedBatchId ?? ""),
    enabled: Boolean(selectedBatchId) && Boolean(selectedBatch?.evaluated),
  });

  const uploadMutation = useMutation({
    mutationFn: uploadPredictionBatch,
    onSuccess: (response) => {
      setUploadMessage("Prediction file accepted. Processing has started.");
      setFile(null);
      queryClient.invalidateQueries({ queryKey: ["batches"] });
      navigate(`/batches/${response.batch_id}`, {
        state: {
          message: "File accepted. Prediction is running.",
        },
      });
    },
    onError: (error) => {
      setUploadMessage(getErrorMessage(error));
    },
  });

  const dateRange = getDateRangeLabel(data?.items ?? []);
  const summary = summaryQuery.data;
  const distribution = summary?.risk_distribution;
  const evaluation = evaluationQuery.data;
  const matrix = buildConfusionMatrix(evaluation);
  const previewRows = useMemo(() => {
    const rows = previewQuery.data?.items ?? [];
    if (previewPredictedLabel === "risk") {
      return rows.filter((item) => item.predicted_label === 1);
    }
    if (previewPredictedLabel === "safe") {
      return rows.filter((item) => item.predicted_label === 0);
    }
    return rows;
  }, [previewPredictedLabel, previewQuery.data?.items]);
  const totalPreviewPages = Math.max(
    1,
    Math.ceil((previewQuery.data?.total ?? 0) / DASHBOARD_PREVIEW_PAGE_SIZE),
  );
  const previewStart = previewQuery.data?.total
    ? (previewPage - 1) * DASHBOARD_PREVIEW_PAGE_SIZE + 1
    : 0;
  const previewEnd = Math.min(
    previewPage * DASHBOARD_PREVIEW_PAGE_SIZE,
    previewQuery.data?.total ?? 0,
  );
  const historyStart = filteredItems.length === 0
    ? 0
    : (historyPage - 1) * historyPageSize + 1;
  const historyEnd = Math.min(historyPage * historyPageSize, filteredItems.length);

  function handleUploadSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploadMessage(null);

    if (!file) {
      setUploadMessage("Choose a prediction CSV before uploading.");
      return;
    }

    const normalizedBatchId = batchId.trim();
    const batchExists = Boolean(
      normalizedBatchId && data?.items.some((item) => item.batch_id === normalizedBatchId),
    );

    if (batchExists && !overwrite) {
      setUploadMessage("This batch ID already exists. Enable overwrite to replace it.");
      return;
    }

    uploadMutation.mutate({
      file,
      batchId: normalizedBatchId || undefined,
    });
  }

  function refreshDashboard() {
    queryClient.invalidateQueries({ queryKey: ["batches"] });
    if (selectedBatchId) {
      queryClient.invalidateQueries({
        queryKey: ["dashboard-batch-summary", selectedBatchId],
      });
      queryClient.invalidateQueries({
        queryKey: ["dashboard-prediction-preview", selectedBatchId],
      });
      queryClient.invalidateQueries({
        queryKey: ["dashboard-evaluation-summary", selectedBatchId],
      });
    }
  }

  function useLastUploadSettings() {
    const latestBatch = data?.items[0];
    if (latestBatch) {
      setBatchId(`${latestBatch.batch_id}-copy`);
      setOverwrite(false);
      setUploadMessage("Last batch settings copied. Choose a CSV file to upload.");
    }
  }

  function handlePreviewSearch(value: string) {
    setPreviewSearch(value.replace(/^S-/i, ""));
    setPreviewPage(1);
  }

  function handlePreviewRiskChange(value: "all" | "high" | "medium" | "low") {
    setPreviewRiskLevel(value);
    setPreviewPage(1);
  }

  function handlePreviewLabelChange(value: "all" | "risk" | "safe") {
    setPreviewPredictedLabel(value);
    setPreviewPage(1);
  }

  function clearPreviewFilters() {
    setPreviewSearch("");
    setPreviewRiskLevel("all");
    setPreviewPredictedLabel("all");
    setPreviewPage(1);
  }

  return (
    <section className="dashboard-page">
      <div className="dashboard-toolbar">
        <label className="search-field dashboard-search">
          <Search size={17} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by Batch ID..."
          />
        </label>

        <div className="date-range-control">
          <span>{dateRange.start}</span>
          <span className="range-arrow">→</span>
          <span>{dateRange.end}</span>
          <Calendar size={16} />
        </div>

        <label className="filter-control">
          <span>Status</span>
          <select
            value={status}
            onChange={(event) => setStatus(event.target.value as "all" | BatchStatus)}
          >
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label.replace("All statuses", "All")}
              </option>
            ))}
          </select>
        </label>

        <button
          className="icon-button refresh-button"
          type="button"
          onClick={refreshDashboard}
          aria-label="Refresh dashboard"
        >
          <RefreshCw size={18} />
        </button>
      </div>

      <div className="panel table-panel history-panel">
        {isLoading ? <LoadingBlock label="Loading batches" /> : null}
        {isError ? (
          <EmptyState
            title="Batch history is unavailable"
            description="Check that the FastAPI service is running on port 8000."
          />
        ) : null}
        {!isLoading && !isError && filteredItems.length === 0 ? (
          <EmptyState
            title="No batches found"
            description="Upload a prediction CSV to create the first batch."
          />
        ) : null}

        {!isLoading && !isError && filteredItems.length > 0 ? (
          <div className="table-scroll">
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Status</th>
                  <th>Total students</th>
                  <th>High risk</th>
                  <th>Medium risk</th>
                  <th>Low risk</th>
                  <th>Truth</th>
                  <th>Created at</th>
                  <th>Updated at</th>
                  <th className="action-cell">Actions</th>
                </tr>
              </thead>
              <tbody>
                {pagedItems.map((item) => (
                  <tr
                    key={item.batch_id}
                    className={item.batch_id === selectedBatchId ? "selected-row" : ""}
                  >
                    <td className="mono-cell">{item.batch_id}</td>
                    <td>
                      <StatusBadge status={item.status} />
                    </td>
                    <td>{item.total_students}</td>
                    <td>
                      <span className="risk-number high">{item.high_risk_count}</span>{" "}
                      <span className="risk-share">
                        {formatCountShare(item.high_risk_count, item.total_students)}
                      </span>
                    </td>
                    <td>
                      <span className="risk-number medium">{item.medium_risk_count}</span>{" "}
                      <span className="risk-share">
                        {formatCountShare(item.medium_risk_count, item.total_students)}
                      </span>
                    </td>
                    <td>
                      <span className="risk-number low">{item.low_risk_count}</span>{" "}
                      <span className="risk-share">
                        {formatCountShare(item.low_risk_count, item.total_students)}
                      </span>
                    </td>
                    <td>
                      <span className={item.truth_uploaded ? "truth-ok" : "truth-pending"}>
                        {item.truth_uploaded ? "Uploaded" : "—"}
                      </span>
                    </td>
                    <td>{formatDate(item.created_at)}</td>
                    <td>{formatDate(item.updated_at)}</td>
                    <td className="action-cell">
                      <button
                        className="table-action"
                        type="button"
                        onClick={() => setSelectedBatchId(item.batch_id)}
                      >
                        View
                      </button>
                      <Link
                        className="table-action table-action-icon"
                        to={`/batches/${item.batch_id}`}
                        aria-label={`Open ${item.batch_id}`}
                      >
                        <ChevronDown size={14} />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {!isLoading && !isError && filteredItems.length > 0 ? (
          <div className="dashboard-table-footer">
            <span>
              Showing {historyStart} to {historyEnd} of {filteredItems.length} results
            </span>
            <div className="footer-pagination">
              <span>Rows per page:</span>
              <select
                value={historyPageSize}
                onChange={(event) => setHistoryPageSize(Number(event.target.value))}
                aria-label="Rows per page"
              >
                {HISTORY_PAGE_SIZE_OPTIONS.map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
              <button
                type="button"
                aria-label="First page"
                disabled={historyPage === 1}
                onClick={() => setHistoryPage(1)}
              >
                <ChevronsLeft size={16} />
              </button>
              <button
                type="button"
                aria-label="Previous page"
                disabled={historyPage === 1}
                onClick={() => setHistoryPage((current) => Math.max(1, current - 1))}
              >
                <ChevronLeft size={16} />
              </button>
              <span className="page-number active">{historyPage}</span>
              <button
                type="button"
                aria-label="Next page"
                disabled={historyPage >= totalHistoryPages}
                onClick={() => setHistoryPage((current) => Math.min(totalHistoryPages, current + 1))}
              >
                <ChevronRight size={16} />
              </button>
              <button
                type="button"
                aria-label="Last page"
                disabled={historyPage >= totalHistoryPages}
                onClick={() => setHistoryPage(totalHistoryPages)}
              >
                <ChevronsRight size={16} />
              </button>
            </div>
          </div>
        ) : null}
      </div>

      <div className="dashboard-lower-grid">
        <form className="panel upload-card" onSubmit={handleUploadSubmit}>
          <div className="card-title-row">
            <h2>Upload Prediction</h2>
          </div>
          <p className="card-subtitle">Upload a CSV file containing model predictions.</p>
          <a className="guide-link" href="/docs/data-contract.md" target="_blank" rel="noreferrer">
            CSV format guide <ExternalLink size={13} />
          </a>

          <UploadDropzone label="Drag and drop CSV file here" file={file} onChange={setFile} />

          <div className="or-divider">
            <span />
            <strong>or</strong>
            <span />
          </div>

          <button
            className="button button-secondary wide-button"
            type="button"
            onClick={useLastUploadSettings}
          >
            Use last upload settings
          </button>

          <div className="field-group compact-field">
            <label htmlFor="dashboard-batch-id">Batch ID (optional)</label>
            <input
              id="dashboard-batch-id"
              className="text-control"
              value={batchId}
              onChange={(event) => setBatchId(event.target.value)}
              placeholder="e.g. BATCH-2024-06-01-001"
            />
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(event) => setOverwrite(event.target.checked)}
            />
            <span>Overwrite if batch ID already exists</span>
          </label>

          {uploadMessage ? <div className="alert alert-info">{uploadMessage}</div> : null}

          <button className="button button-primary upload-submit" disabled={uploadMutation.isPending}>
            {uploadMutation.isPending ? "Uploading" : "Upload prediction"}
          </button>
        </form>

        <div className="panel dashboard-detail-card">
          <div className="card-title-row">
            <div>
              <h2>
                Batch Detail:{" "}
                <span className="mono-heading">{selectedBatchId ?? "No batch selected"}</span>
              </h2>
            </div>
            <div className="title-actions">
              {summary ? <StatusBadge status={summary.status} /> : null}
              {selectedBatchId && selectedBatchHasPrediction ? (
                <a className="button button-primary compact" href={predictionDownloadUrl(selectedBatchId)}>
                  <Download size={15} />
                  Download CSV
                </a>
              ) : null}
            </div>
          </div>

          {summary && distribution ? (
            <>
              <div className="metric-strip">
                <MiniMetric label="Total students" value={summary.total_students} />
                <MiniMetric
                  label="High risk"
                  value={
                    <>
                      {distribution.high}
                      <small>{formatCountShare(distribution.high, summary.total_students)}</small>
                    </>
                  }
                  tone="high"
                />
                <MiniMetric
                  label="Medium risk"
                  value={
                    <>
                      {distribution.medium}
                      <small>{formatCountShare(distribution.medium, summary.total_students)}</small>
                    </>
                  }
                  tone="medium"
                />
                <MiniMetric
                  label="Low risk"
                  value={
                    <>
                      {distribution.low}
                      <small>{formatCountShare(distribution.low, summary.total_students)}</small>
                    </>
                  }
                  tone="low"
                />
                <MiniMetric
                  label="Truth"
                  value={summary.truth_uploaded ? "Uploaded" : "—"}
                  tone={summary.truth_uploaded ? "low" : "default"}
                />
              </div>

              <div className="detail-filter-row">
                <label className="search-field compact-search">
                  <Search size={15} />
                  <input
                    value={previewSearch}
                    onChange={(event) => handlePreviewSearch(event.target.value)}
                    placeholder="Search student ID..."
                  />
                </label>
                <label>
                  Risk level
                  <select
                    className="select-control"
                    value={previewRiskLevel}
                    onChange={(event) =>
                      handlePreviewRiskChange(event.target.value as "all" | "high" | "medium" | "low")
                    }
                  >
                    <option value="all">All</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </label>
                <label>
                  Predicted label
                  <select
                    className="select-control"
                    value={previewPredictedLabel}
                    onChange={(event) =>
                      handlePreviewLabelChange(event.target.value as "all" | "risk" | "safe")
                    }
                  >
                    <option value="all">All</option>
                    <option value="risk">At Risk</option>
                    <option value="safe">Not At Risk</option>
                  </select>
                </label>
                <button className="button button-ghost compact" type="button" onClick={clearPreviewFilters}>
                  Clear filters
                </button>
              </div>

              <div className="table-scroll detail-table-scroll">
                <table className="dashboard-table compact-table">
                  <thead>
                    <tr>
                      <th>Student ID</th>
                      <th>Risk score</th>
                      <th>Risk level</th>
                      <th>Predicted label</th>
                      <th>Truth</th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((item) => (
                      <tr key={`${item.id}-${item.risk_score}`}>
                        <td className="mono-cell">S-{item.id}</td>
                        <td>{formatDecimal(item.risk_score, 2)}</td>
                        <td>
                          <RiskBadge level={item.risk_level} />
                        </td>
                        <td className={item.predicted_label === 1 ? "label-risk" : "label-safe"}>
                          {item.predicted_label === 1 ? "At Risk" : "Not At Risk"}
                        </td>
                        <td>{summary.truth_uploaded ? "Uploaded" : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="dashboard-table-footer detail-footer">
                <span>
                  Showing {previewStart} to {previewEnd} of {previewQuery.data?.total ?? 0} results
                </span>
                <div className="footer-pagination">
                  <span>Rows per page:</span>
                  <span className="page-size-label">{DASHBOARD_PREVIEW_PAGE_SIZE}</span>
                  <button
                    type="button"
                    aria-label="Previous prediction page"
                    disabled={previewPage === 1}
                    onClick={() => setPreviewPage((current) => Math.max(1, current - 1))}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <span className="page-number active">{previewPage}</span>
                  <button
                    type="button"
                    aria-label="Next prediction page"
                    disabled={previewPage >= totalPreviewPages}
                    onClick={() => setPreviewPage((current) => Math.min(totalPreviewPages, current + 1))}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            </>
          ) : (
            <EmptyState
              title="No batch selected"
              description="Upload or select a prediction batch to view details."
            />
          )}
        </div>

        <div className="panel evaluation-card">
          <div className="card-title-row">
            <div>
              <h2>Evaluation Summary</h2>
              <p className="card-subtitle">
                {selectedBatch?.created_at
                  ? `Computed on ${formatDate(selectedBatch.created_at)}`
                  : "Available after truth upload"}
              </p>
            </div>
            {evaluation ? <StatusBadge status="evaluated" /> : null}
          </div>

          {evaluation ? (
            <>
              <div className="evaluation-list">
                <MetricRow label="Accuracy" value={formatDecimal(evaluation.metrics.accuracy, 4)} />
                <MetricRow
                  label="Precision risk"
                  value={formatDecimal(evaluation.metrics.precision_risk, 4)}
                />
                <MetricRow
                  label="Recall risk"
                  value={formatDecimal(evaluation.metrics.recall_risk, 4)}
                />
                <MetricRow label="F1 risk" value={formatDecimal(evaluation.metrics.f1_risk, 4)} />
                <MetricRow label="Matched ratio" value={formatDecimal(evaluation.matched_ratio, 4)} />
              </div>

              <div className="confusion-card">
                <h3>Confusion Matrix (Risk vs Truth)</h3>
                <table className="confusion-table">
                  <thead>
                    <tr>
                      <th />
                      <th>Truth<br />At Risk</th>
                      <th>Truth<br />Not At Risk</th>
                      <th>Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Pred<br />At Risk</th>
                      <td>{matrix.tp}</td>
                      <td>{matrix.fp}</td>
                      <td>{matrix.tp + matrix.fp}</td>
                    </tr>
                    <tr>
                      <th>Pred<br />Not At Risk</th>
                      <td>{matrix.fn}</td>
                      <td>{matrix.tn}</td>
                      <td>{matrix.fn + matrix.tn}</td>
                    </tr>
                    <tr>
                      <th>Total</th>
                      <td>{matrix.tp + matrix.fn}</td>
                      <td>{matrix.fp + matrix.tn}</td>
                      <td>{evaluation.matched_rows}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <Link className="evaluation-link" to={`/batches/${evaluation.batch_id}/evaluation`}>
                View evaluation details <ChevronRight size={16} />
              </Link>
            </>
          ) : (
            <EmptyState
              title="Evaluation not available"
              description="Upload truth for the selected batch to generate quality metrics."
            />
          )}
        </div>
      </div>
    </section>
  );
}

function MiniMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  tone?: "default" | "high" | "medium" | "low";
}) {
  return (
    <div className={`mini-metric mini-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatCountShare(count: number, total: number): string {
  if (!total) {
    return "";
  }

  return `${((count / total) * 100).toFixed(1)}%`;
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-row compact-metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function getDateRangeLabel(items: BatchListItem[]): { start: string; end: string } {
  const dates = items
    .map((item) => item.created_at)
    .filter((value): value is string => Boolean(value))
    .map((value) => new Date(value))
    .filter((date) => !Number.isNaN(date.getTime()))
    .sort((a, b) => a.getTime() - b.getTime());

  if (dates.length === 0) {
    return { start: "No data", end: "No data" };
  }

  return {
    start: formatDateInput(dates[0]),
    end: formatDateInput(dates[dates.length - 1]),
  };
}

function formatDateInput(date: Date): string {
  return new Intl.DateTimeFormat("en", {
    month: "2-digit",
    day: "2-digit",
    year: "numeric",
  }).format(date);
}

function buildConfusionMatrix(evaluation: EvaluationSummary | undefined) {
  if (!evaluation) {
    return { tp: 0, fp: 0, fn: 0, tn: 0 };
  }

  const total = evaluation.matched_rows;
  const accuracy = evaluation.metrics.accuracy ?? 0;
  const precision = evaluation.metrics.precision_risk ?? 0;
  const recall = evaluation.metrics.recall_risk ?? 0;
  const fn = evaluation.metrics.false_negative_count ?? 0;

  let tp = recall < 1 ? Math.round((fn * recall) / Math.max(1 - recall, 0.001)) : 0;
  if (!Number.isFinite(tp) || tp < 0) {
    tp = 0;
  }

  let fp = precision > 0 ? Math.round(tp * (1 / precision - 1)) : 0;
  if (!Number.isFinite(fp) || fp < 0) {
    fp = 0;
  }

  let tn = Math.round(accuracy * total - tp);
  if (!Number.isFinite(tn) || tn < 0) {
    tn = Math.max(total - tp - fp - fn, 0);
  }

  const currentTotal = tp + fp + fn + tn;
  if (currentTotal !== total) {
    tn = Math.max(tn + total - currentTotal, 0);
  }

  return { tp, fp, fn, tn };
}
