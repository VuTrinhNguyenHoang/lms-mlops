import type { RiskLevel } from "../types/batch";

interface RiskBadgeProps {
  level: RiskLevel;
}

export default function RiskBadge({ level }: RiskBadgeProps) {
  return <span className={`risk-badge risk-${level}`}>{level}</span>;
}
