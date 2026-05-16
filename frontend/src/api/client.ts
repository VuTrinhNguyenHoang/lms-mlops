import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") {
      return detail;
    }
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Unexpected error";
}
