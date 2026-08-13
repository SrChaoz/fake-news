export type PredictionLabel = "REAL" | "FAKE";

export interface Entity {
  text: string;
  normalized?: string;
  label?: string;
  source?: string;
  start_char?: number;
  end_char?: number;
}

export interface OntologyConflict {
  subject?: string;
  expected_relation?: string;
  reason?: string;
}

export interface PredictionResponse {
  extracted_claim: string;
  category: string;
  detected_entities: Entity[];
  ontological_conflicts: OntologyConflict[];
  prediction: PredictionLabel;
  confidence_score: number;
  explanation: string;
  probability_real: number;
  decision_threshold_real: number;
  model_prediction: PredictionLabel;
  decision_source: string;
  verification_status: "SUPPORTED" | "REFUTED" | "INSUFFICIENT_EVIDENCE" | "ONTOLOGY_CONFLICT" | "ONTOLOGY_SUPPORT";
  evidence: Array<{
    claim: string;
    label: PredictionLabel;
    evidence_text: string;
    source_dataset: string;
    source_url: string | null;
    similarity: number;
  }>;
}

export interface HistoryItem {
  id: number;
  input_text: string;
  extracted_claim: string | null;
  category: string | null;
  prediction: PredictionLabel;
  confidence_score: number;
  explanation: string | null;
  created_at: string;
}

export interface MetricSet {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  macro_f1?: number;
  confusion_matrix?: number[][];
}

export interface MetricsResponse {
  model: string;
  metrics: MetricSet;
  cross_validation: unknown;
  dataset: Record<string, unknown>;
}

export interface ExperimentsResponse {
  hybrid_model: MetricSet;
  transformer_baseline: { name?: string; metrics?: MetricSet };
  bert: { status: string; configuration?: Record<string, unknown> };
  roberta: { status: string; detail?: string };
  ontology_system: { status: string; detail?: string };
  saved_experiments: Array<Record<string, unknown>>;
}

export type AblationResponse = Record<string, MetricSet>;

export interface HoldoutEvaluationReport {
  benchmark: string;
  model_directory: string;
  records: number;
  class_counts: Record<PredictionLabel, number>;
  metrics: {
    accuracy: number;
    balanced_accuracy: number;
    macro_f1: number;
    fake: Required<Pick<MetricSet, "precision" | "recall" | "f1_score">>;
    real: Required<Pick<MetricSet, "precision" | "recall" | "f1_score">>;
    confusion_matrix: number[][];
  };
  error_count: number;
  warning?: string;
}

export interface HoldoutEvaluationResponse {
  evaluation: {
    status: "idle" | "running" | "completed" | "failed";
    started_at: string | null;
    finished_at: string | null;
    error: string | null;
  };
  report: HoldoutEvaluationReport | null;
}
