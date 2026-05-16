export function formatDate(value: string | null): string {
  if (!value) {
    return "-";
  }

  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatPercent(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${Math.round(value * 100)}%`;
}

export function formatDecimal(value: number | undefined, digits = 2): string {
  if (value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return value.toFixed(digits);
}
