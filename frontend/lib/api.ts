import type { AblationResponse, ExperimentsResponse, HistoryItem, MetricsResponse, PredictionResponse } from "@/lib/types";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(`No se pudo conectar con la API en ${API_BASE_URL}.`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail ?? `La API devolvió HTTP ${response.status}.`, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  predict: (text: string) => request<PredictionResponse>("/predict", { method: "POST", body: JSON.stringify({ text }) }),
  history: (limit = 50) => request<{ items: HistoryItem[] }>(`/history?limit=${limit}`),
  metrics: () => request<MetricsResponse>("/metrics"),
  experiments: () => request<ExperimentsResponse>("/experiments"),
  ablation: () => request<AblationResponse>("/ablation"),
  health: () => request<{ status: string }>("/health"),
};
