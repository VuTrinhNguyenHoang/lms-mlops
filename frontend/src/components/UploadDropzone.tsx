import { UploadCloud } from "lucide-react";

interface UploadDropzoneProps {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
}

export default function UploadDropzone({
  label,
  file,
  onChange,
}: UploadDropzoneProps) {
  return (
    <label className="dropzone">
      <input
        type="file"
        accept=".csv,text/csv"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      <span className="dropzone-icon">
        <UploadCloud size={22} />
      </span>
      <span className="dropzone-title">{file ? file.name : label}</span>
      <span className="dropzone-meta">CSV only</span>
    </label>
  );
}
