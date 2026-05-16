interface SummaryTileProps {
  label: string;
  value: string | number;
  tone?: "default" | "high" | "medium" | "low";
}

export default function SummaryTile({
  label,
  value,
  tone = "default",
}: SummaryTileProps) {
  return (
    <div className={`summary-tile tile-${tone}`}>
      <div className="summary-label">{label}</div>
      <div className="summary-value">{value}</div>
    </div>
  );
}
