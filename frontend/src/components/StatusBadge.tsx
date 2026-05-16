import type { BatchStatus } from "../types/batch";

const STATUS_LABELS: Record<BatchStatus, string> = {
  processing_prediction: "Processing",
  predicted: "Predicted",
  evaluating: "Evaluating",
  evaluated: "Evaluated",
  not_found: "Not found",
};

interface StatusBadgeProps {
  status: BatchStatus;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span className={`status-badge status-${status}`}>
      {STATUS_LABELS[status] ?? status}
    </span>
  );
}
