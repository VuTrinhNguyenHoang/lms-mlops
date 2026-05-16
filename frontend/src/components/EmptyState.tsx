interface EmptyStateProps {
  title: string;
  description: string;
}

export default function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <div className="empty-title">{title}</div>
      <div className="empty-description">{description}</div>
    </div>
  );
}
