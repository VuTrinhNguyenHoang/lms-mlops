import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";

import { getErrorMessage } from "../api/client";
import { uploadPredictionBatch } from "../api/batches";
import PageHeader from "../components/PageHeader";
import UploadDropzone from "../components/UploadDropzone";

export default function UploadPredictionPage() {
  const navigate = useNavigate();
  const [batchId, setBatchId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: uploadPredictionBatch,
    onSuccess: (response) => {
      navigate(`/batches/${response.batch_id}`, {
        state: {
          message: "File accepted. Prediction is running.",
        },
      });
    },
    onError: (error) => {
      setErrorMessage(getErrorMessage(error));
    },
  });

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage(null);

    if (!file) {
      setErrorMessage("Select a CSV file before uploading.");
      return;
    }

    mutation.mutate({
      file,
      batchId: batchId || undefined,
    });
  }

  return (
    <section className="page-stack narrow-page">
      <PageHeader
        title="Upload Prediction"
        description="Upload an ID + FEATURES CSV. The prediction flow will run in the background."
      />

      <form className="panel form-panel" onSubmit={handleSubmit}>
        <div className="field-group">
          <label htmlFor="batch-id">Batch ID</label>
          <input
            id="batch-id"
            className="text-control"
            value={batchId}
            onChange={(event) => setBatchId(event.target.value)}
            placeholder="Optional, for example demo-001"
          />
          <p className="help-text">
            Leave this empty to let the API generate a batch identifier.
          </p>
        </div>

        <UploadDropzone
          label="Choose prediction CSV"
          file={file}
          onChange={setFile}
        />

        {errorMessage ? <div className="alert alert-error">{errorMessage}</div> : null}

        <div className="form-actions">
          <button className="button button-primary" disabled={mutation.isPending}>
            {mutation.isPending ? "Uploading" : "Upload prediction"}
          </button>
        </div>
      </form>
    </section>
  );
}
