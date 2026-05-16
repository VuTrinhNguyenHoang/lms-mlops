import { ChangeEvent, useMemo, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { Download, Search } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  fetchBatchSummary,
  fetchPredictionPreview,
  predictionDownloadUrl,
  uploadTruthBatch,
} from "../api/batches";
import { getErrorMessage } from "../api/client";
import EmptyState from "../components/EmptyState";
import { formatDecimal } from "../components/format";
import LoadingBlock from "../components/LoadingBlock";
import PageHeader from "../components/PageHeader";
import RiskBadge from "../components/RiskBadge";
import StatusBadge from "../components/StatusBadge";
import SummaryTile from "../components/SummaryTile";
import UploadDropzone from "../components/UploadDropzone";
import type { RiskLevel } from "../types/batch";

const PAGE_SIZE = 20;

interface LocationState {
  message?: string;
}

export default function BatchDetailPage() {
  const { batchId = "" } = useParams();
  const location = useLocation();
  const state = location.state as LocationState | null;
  const queryClient = useQueryClient();
  const [riskLevel, setRiskLevel] = useState<RiskLevel | "all">("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [truthFile, setTruthFile] = useState<File | null>(null);
  const [truthMessage, setTruthMessage] = useState<string | null>(null);

  const summaryQuery = useQuery({
    queryKey: ["batch-summary", batchId],
    queryFn: () => fetchBatchSummary(batchId),
    enabled: Boolean(batchId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "processing_prediction" || status === "evaluating"
        ? 3000
        : false;
    },
  });

  const canLoadPreview =
    summaryQuery.data?.status === "predicted" ||
    summaryQuery.data?.status === "evaluating" ||
    summaryQuery.data?.status === "evaluated";

  const previewQuery = useQuery({
    queryKey: ["prediction-preview", batchId, page, riskLevel, search],
    queryFn: () =>
      fetchPredictionPreview({
        batchId,
        page,
        pageSize: PAGE_SIZE,
        riskLevel,
        query: search,
      }),
    enabled: Boolean(batchId) && canLoadPreview,
  });

  const truthMutation = useMutation({
    mutationFn: uploadTruthBatch,
    onSuccess: (response) => {
      setTruthMessage(response.message);
      setTruthFile(null);
      queryClient.invalidateQueries({ queryKey: ["batch-summary", batchId] });
    },
    onError: (error) => {
      setTruthMessage(getErrorMessage(error));
    },
  });

  const totalPages = useMemo(() => {
    const total = previewQuery.data?.total ?? 0;
    return Math.max(1, Math.ceil(total / PAGE_SIZE));
  }, [previewQuery.data?.total]);

  function handleRiskChange(event: ChangeEvent<HTMLSelectElement>) {
    setRiskLevel(event.target.value as RiskLevel | "all");
    setPage(1);
  }

  function handleSearchChange(event: ChangeEvent<HTMLInputElement>) {
    setSearch(event.target.value);
    setPage(1);
  }

  function handleTruthUpload() {
    setTruthMessage(null);
    if (!truthFile) {
      setTruthMessage("Select a truth CSV before uploading.");
      return;
    }

    truthMutation.mutate({
      batchId,
      file: truthFile,
    });
  }

  if (summaryQuery.isLoading) {
    return <LoadingBlock label="Loading batch detail" />;
  }

  if (summaryQuery.isError || !summaryQuery.data) {
    return (
      <EmptyState
        title="Batch not found"
        description="Return to history and select an existing batch."
      />
    );
  }

  const summary = summaryQuery.data;
  const distribution = summary.risk_distribution;

  return (
    <section className="page-stack">
      <PageHeader
        title={`Batch ${batchId}`}
        description="Prediction result and truth upload for this batch."
        action={
          <div className="header-actions">
            {summary.evaluated ? (
              <Link className="button button-secondary" to={`/batches/${batchId}/evaluation`}>
                Evaluation
              </Link>
            ) : null}
            <a className="button button-ghost" href={predictionDownloadUrl(batchId)}>
              <Download size={17} />
              Download CSV
            </a>
          </div>
        }
      />

      {state?.message ? <div className="alert alert-info">{state.message}</div> : null}

      <div className="detail-status-row">
        <StatusBadge status={summary.status} />
        {summary.truth_uploaded ? <span className="muted-text">Truth uploaded</span> : null}
      </div>

      <div className="summary-grid">
        <SummaryTile label="Total students" value={summary.total_students} />
        <SummaryTile label="High risk" value={distribution.high} tone="high" />
        <SummaryTile label="Medium risk" value={distribution.medium} tone="medium" />
        <SummaryTile label="Low risk" value={distribution.low} tone="low" />
      </div>

      <div className="panel truth-panel">
        <div>
          <h2>Upload truth</h2>
          <p>Attach the ID + FEATURES + TARGET CSV for this batch.</p>
        </div>
        <div className="truth-upload">
          <UploadDropzone
            label="Choose truth CSV"
            file={truthFile}
            onChange={setTruthFile}
          />
          <button
            className="button button-secondary"
            type="button"
            onClick={handleTruthUpload}
            disabled={truthMutation.isPending}
          >
            {truthMutation.isPending ? "Uploading" : "Upload truth"}
          </button>
        </div>
        {truthMessage ? <div className="alert alert-info">{truthMessage}</div> : null}
      </div>

      <div className="panel table-panel">
        <div className="panel-heading">
          <div>
            <h2>Prediction table</h2>
            <p>Sorted by risk score in descending order.</p>
          </div>
          <div className="toolbar inline">
            <label className="search-field compact-search">
              <Search size={16} />
              <input
                value={search}
                onChange={handleSearchChange}
                placeholder="Search ID"
              />
            </label>
            <select
              className="select-control"
              value={riskLevel}
              onChange={handleRiskChange}
            >
              <option value="all">All risk levels</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>

        {!canLoadPreview ? (
          <EmptyState
            title="Prediction is still running"
            description="This page will refresh when the prediction output is available."
          />
        ) : null}

        {previewQuery.isLoading && canLoadPreview ? (
          <LoadingBlock label="Loading predictions" />
        ) : null}

        {previewQuery.data && previewQuery.data.items.length > 0 ? (
          <>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Risk score</th>
                    <th>Risk level</th>
                    <th>Predicted label</th>
                  </tr>
                </thead>
                <tbody>
                  {previewQuery.data.items.map((item) => (
                    <tr key={`${item.id}-${item.risk_score}`}>
                      <td className="mono-cell">{item.id}</td>
                      <td>{formatDecimal(item.risk_score, 3)}</td>
                      <td>
                        <RiskBadge level={item.risk_level} />
                      </td>
                      <td>{item.predicted_label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination-row">
              <span>
                Page {page} of {totalPages}
              </span>
              <div className="pagination-actions">
                <button
                  className="button button-ghost compact"
                  disabled={page === 1}
                  onClick={() => setPage((current) => Math.max(1, current - 1))}
                >
                  Previous
                </button>
                <button
                  className="button button-ghost compact"
                  disabled={page >= totalPages}
                  onClick={() => setPage((current) => current + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        ) : null}

        {previewQuery.data && previewQuery.data.items.length === 0 ? (
          <EmptyState
            title="No prediction rows"
            description="Adjust the search or risk-level filter."
          />
        ) : null}
      </div>
    </section>
  );
}
