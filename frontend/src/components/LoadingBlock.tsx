interface LoadingBlockProps {
  label?: string;
}

export default function LoadingBlock({ label = "Loading data" }: LoadingBlockProps) {
  return (
    <div className="loading-block" role="status">
      <span className="spinner" />
      <span>{label}</span>
    </div>
  );
}
