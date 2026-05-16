import { Navigate, Route, Routes } from "react-router-dom";

import Layout from "./components/Layout";
import BatchDetailPage from "./pages/BatchDetailPage";
import BatchHistoryPage from "./pages/BatchHistoryPage";
import EvaluationIndexPage from "./pages/EvaluationIndexPage";
import EvaluationPage from "./pages/EvaluationPage";
import UploadPredictionPage from "./pages/UploadPredictionPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<BatchHistoryPage />} />
        <Route path="/batches/new" element={<UploadPredictionPage />} />
        <Route path="/batches/:batchId" element={<BatchDetailPage />} />
        <Route path="/batches/:batchId/evaluation" element={<EvaluationPage />} />
        <Route path="/evaluation" element={<EvaluationIndexPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
