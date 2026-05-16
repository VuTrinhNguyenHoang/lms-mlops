import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { fetchBatches } from "../api/batches";
import EmptyState from "../components/EmptyState";
import { formatDate } from "../components/format";
import LoadingBlock from "../components/LoadingBlock";
import PageHeader from "../components/PageHeader";
import StatusBadge from "../components/StatusBadge";

export default function EvaluationIndexPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["batches"],
    queryFn: fetchBatches,
    refetchInterval: 4000,
  });

  const evaluatedItems = (data?.items ?? []).filter((item) => item.evaluated);

  return (
    <section className="page-stack">
      <PageHeader
        title="Evaluation"
        description="Open evaluated batches and review user-facing quality metrics."
      />

      <div className="panel table-panel">
        {isLoading ? <LoadingBlock label="Loading evaluations" /> : null}
        {isError ? (
          <EmptyState
            title="Evaluation list is unavailable"
            description="Check that the FastAPI service is running on port 8000."
          />
        ) : null}
        {!isLoading && !isError && evaluatedItems.length === 0 ? (
          <EmptyState
            title="No evaluated batches"
            description="Upload truth for a predicted batch to generate evaluation metrics."
          />
        ) : null}

        {!isLoading && !isError && evaluatedItems.length > 0 ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Status</th>
                  <th>Total students</th>
                  <th>Created</th>
                  <th className="action-cell">Action</th>
                </tr>
              </thead>
              <tbody>
                {evaluatedItems.map((item) => (
                  <tr key={item.batch_id}>
                    <td className="mono-cell">{item.batch_id}</td>
                    <td>
                      <StatusBadge status={item.status} />
                    </td>
                    <td>{item.total_students}</td>
                    <td>{formatDate(item.created_at)}</td>
                    <td className="action-cell">
                      <Link
                        className="button button-ghost compact"
                        to={`/batches/${item.batch_id}/evaluation`}
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </section>
  );
}
