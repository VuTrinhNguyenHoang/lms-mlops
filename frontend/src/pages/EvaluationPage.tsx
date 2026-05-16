import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { fetchEvaluationSummary } from "../api/batches";
import EmptyState from "../components/EmptyState";
import { formatPercent } from "../components/format";
import LoadingBlock from "../components/LoadingBlock";
import PageHeader from "../components/PageHeader";
import SummaryTile from "../components/SummaryTile";

export default function EvaluationPage() {
  const { batchId = "" } = useParams();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["evaluation-summary", batchId],
    queryFn: () => fetchEvaluationSummary(batchId),
    enabled: Boolean(batchId),
  });

  if (isLoading) {
    return <LoadingBlock label="Loading evaluation" />;
  }

  if (isError || !data) {
    return (
      <section className="page-stack">
        <PageHeader
          title={`Evaluation ${batchId}`}
          description="Evaluation is available after a truth CSV is processed."
          action={
            <Link className="button button-ghost" to={`/batches/${batchId}`}>
              Back to batch
            </Link>
          }
        />
        <EmptyState
          title="Evaluation is not available"
          description="Upload truth for this batch and wait until evaluation finishes."
        />
      </section>
    );
  }

  const metrics = data.metrics;

  return (
    <section className="page-stack">
      <PageHeader
        title={`Evaluation ${batchId}`}
        description="User-facing model quality metrics for the uploaded truth batch."
        action={
          <Link className="button button-ghost" to={`/batches/${batchId}`}>
            Back to batch
          </Link>
        }
      />

      <div className="summary-grid evaluation-grid">
        <SummaryTile label="Accuracy" value={formatPercent(metrics.accuracy)} />
        <SummaryTile
          label="Precision risk"
          value={formatPercent(metrics.precision_risk)}
        />
        <SummaryTile label="Recall risk" value={formatPercent(metrics.recall_risk)} />
        <SummaryTile label="F1 risk" value={formatPercent(metrics.f1_risk)} />
        <SummaryTile label="Matched ratio" value={formatPercent(data.matched_ratio)} />
      </div>

      <div className="panel metric-panel">
        <h2>Evaluation summary</h2>
        <div className="metric-list">
          <MetricRow label="Truth rows" value={data.truth_rows} />
          <MetricRow label="Matched rows" value={data.matched_rows} />
          <MetricRow label="Matched ratio" value={formatPercent(data.matched_ratio)} />
          <MetricRow
            label="False negative count"
            value={metrics.false_negative_count ?? "-"}
          />
        </div>
      </div>
    </section>
  );
}

function MetricRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
