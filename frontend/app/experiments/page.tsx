"use client";

import { AlertCircle, BarChart3, Download, Play, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { LoadingSpinner } from "@/components/loading-spinner";
import { useToast } from "@/components/toast";
import { api, ApiError } from "@/lib/api";
import type { AblationResponse, ExperimentsResponse, HoldoutEvaluationResponse, MetricSet, MetricsResponse } from "@/lib/types";

const ablationLabels: Record<string, string> = {
  full_hybrid: "Completo",
  without_ontology: "Sin ontologías",
  without_claim_extraction: "Sin claim",
  without_ontology_and_claim: "Sin ambos",
};

type DashboardData = {
  metrics: MetricsResponse;
  experiments: ExperimentsResponse;
  ablation: AblationResponse;
  holdout: HoldoutEvaluationResponse;
};

function percent(value: number | undefined | null): string {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "N/D";
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function ExperimentsPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [startingEvaluation, setStartingEvaluation] = useState(false);
  const { notify } = useToast();

  const load = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const [metrics, experiments, ablation, holdout] = await Promise.all([
        api.metrics(), api.experiments(), api.ablation(), api.holdoutEvaluation(),
      ]);
      setData({ metrics, experiments, ablation, holdout });
    } catch (error) {
      notify(error instanceof ApiError ? error.message : "No se pudieron cargar las métricas.", "error");
    } finally {
      if (showLoading) setLoading(false);
    }
  }, [notify]);

  useEffect(() => { void load(true); }, [load]);
  useEffect(() => {
    if (data?.holdout.evaluation.status !== "running") return;
    const timeout = window.setTimeout(() => void load(), 2500);
    return () => window.clearTimeout(timeout);
  }, [data?.holdout.evaluation.status, load]);

  const runEvaluation = async () => {
    setStartingEvaluation(true);
    try {
      await api.startHoldoutEvaluation();
      notify("Evaluación del modelo promovido iniciada. La página se actualizará al finalizar.");
      await load();
    } catch (error) {
      notify(error instanceof ApiError ? error.message : "No se pudo iniciar la evaluación.", "error");
    } finally {
      setStartingEvaluation(false);
    }
  };

  const exportMetrics = () => {
    if (!data) return;
    downloadJson(`climate-veritas-metrics-${new Date().toISOString().slice(0, 10)}.json`, {
      exported_at: new Date().toISOString(),
      production: data.metrics,
      holdout_evaluation: data.holdout.report,
      ablation_study: data.ablation,
      experiments: data.experiments.saved_experiments,
    });
    notify("Métricas exportadas como JSON.");
  };

  if (loading) return <LoadingSpinner label="Cargando experimentos y ablaciones..." />;
  if (!data) return <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">No hay datos de métricas disponibles.</div>;

  const global = data.metrics.metrics;
  const holdoutMetrics = data.holdout.report?.metrics;
  const ablationData = Object.entries(data.ablation).map(([key, values]) => ({
    name: ablationLabels[key] ?? key,
    accuracy: values.accuracy ?? 0,
    macroF1: values.macro_f1 ?? 0,
  }));
  const modelData = [
    { name: "Baseline BERT-tiny", score: data.experiments.transformer_baseline.metrics?.f1_score ?? 0 },
    { name: "Híbrido interno", score: data.experiments.hybrid_model.f1_score ?? 0 },
    { name: "Híbrido holdout", score: holdoutMetrics?.macro_f1 ?? 0 },
  ];
  const savedExperiments = data.experiments.saved_experiments.map((experiment) => {
    const record = asRecord(experiment);
    const dataset = asRecord(record.dataset);
    const crossValidation = asRecord(record.cross_validation);
    const macro = asRecord(crossValidation.macro_f1);
    return {
      name: String(record.name ?? "Experimento"),
      records: typeof dataset.records === "number" ? dataset.records : null,
      macroF1: typeof macro.mean === "number" ? macro.mean : null,
      accuracy: typeof asRecord(crossValidation.accuracy).mean === "number" ? asRecord(crossValidation.accuracy).mean as number : null,
    };
  });
  const evaluationRunning = data.holdout.evaluation.status === "running";

  return <div className="mx-auto max-w-7xl">
    <header className="mb-8 flex flex-col justify-between gap-5 border-b border-white/10 pb-6 lg:flex-row lg:items-end">
      <div>
        <p className="eyebrow text-emerald-300">MODEL EVALUATION LAB</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-100 sm:text-4xl">Experimentos y validación</h1>
        <p className="mt-2 max-w-2xl text-sm text-zinc-400">Métricas del artefacto promovido, benchmark retenido y ablaciones. No se reentrena desde la interfaz.</p>
      </div>
      <div className="flex flex-wrap gap-3">
        <button onClick={exportMetrics} className="inline-flex items-center gap-2 rounded border border-white/10 bg-white/5 px-4 py-2 text-xs text-zinc-300 hover:bg-white/10"><Download size={15} />Exportar métricas</button>
        <button onClick={() => void load()} className="inline-flex items-center gap-2 rounded border border-white/10 bg-white/5 px-4 py-2 text-xs text-zinc-300 hover:bg-white/10"><RefreshCw size={15} />Actualizar</button>
        <button onClick={() => void runEvaluation()} disabled={startingEvaluation || evaluationRunning} className="inline-flex items-center gap-2 rounded border border-emerald-400/40 bg-emerald-400/10 px-4 py-2 text-xs text-emerald-300 disabled:cursor-not-allowed disabled:opacity-50">
          <Play size={15} />{evaluationRunning ? "Evaluando…" : "Ejecutar evaluación"}
        </button>
      </div>
    </header>

    <section className="mb-8 rounded-xl border border-emerald-400/20 bg-emerald-400/5 p-5">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div><p className="eyebrow text-emerald-300">Modelo promovido · holdout_bert</p><h2 className="mt-2 text-lg font-semibold text-zinc-100">Benchmark retenido: {data.holdout.report?.records ?? "—"} afirmaciones</h2><p className="mt-1 text-sm text-zinc-400">El holdout se excluyó del entrenamiento. Es la referencia principal de generalización disponible.</p></div>
        <p className={`rounded-full px-3 py-1 text-xs font-medium ${evaluationRunning ? "bg-amber-400/15 text-amber-300" : "bg-emerald-400/15 text-emerald-300"}`}>{evaluationRunning ? "Evaluación en ejecución" : data.holdout.evaluation.status === "failed" ? "Última evaluación falló" : "Último informe disponible"}</p>
      </div>
      {data.holdout.evaluation.error && <p className="mt-3 rounded bg-rose-500/10 p-3 text-xs text-rose-200">{data.holdout.evaluation.error}</p>}
      <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="Accuracy holdout" value={holdoutMetrics?.accuracy} color="#00ff88" /><MetricCard label="Macro-F1 holdout" value={holdoutMetrics?.macro_f1} color="#adc6ff" /><MetricCard label="Precisión FAKE" value={holdoutMetrics?.fake.precision} color="#ffb3ae" /><MetricCard label="Recall FAKE" value={holdoutMetrics?.fake.recall} color="#00ff88" /></div>
    </section>

    <section className="mb-8"><div className="mb-3 flex items-center gap-2"><AlertCircle size={16} className="text-amber-400" /><p className="text-xs text-zinc-500">Las métricas siguientes corresponden al split interno 80/20, no al holdout. Precision, recall y F1 se calculan para la clase REAL.</p></div><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="Accuracy interna" value={global.accuracy} color="#00ff88" /><MetricCard label="Precision · REAL" value={global.precision} color="#adc6ff" /><MetricCard label="Recall · REAL" value={global.recall} color="#ffb3ae" /><MetricCard label="F1 · REAL" value={global.f1_score} color="#00ff88" /></div></section>

    <div className="grid gap-6 xl:grid-cols-2">
      <ChartPanel title="Comparación con contexto" subtitle="F1 del split interno para baseline/híbrido; macro-F1 para holdout"><ResponsiveContainer width="100%" height={280}><BarChart data={modelData}><CartesianGrid stroke="rgba(255,255,255,.07)" vertical={false} /><XAxis dataKey="name" stroke="#849585" tickLine={false} axisLine={false} /><YAxis domain={[0, 1]} stroke="#849585" tickFormatter={(value) => `${Math.round(value * 100)}%`} /><Tooltip contentStyle={{ background: "#201f1f", border: "1px solid rgba(255,255,255,.14)", borderRadius: 8 }} formatter={(value) => typeof value === "number" ? percent(value) : "N/D"} /><Bar dataKey="score" fill="#00ff88" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel>
      <ChartPanel title="Impacto de ablación" subtitle="Accuracy y macro-F1 por variante, split interno"><ResponsiveContainer width="100%" height={280}><BarChart data={ablationData}><CartesianGrid stroke="rgba(255,255,255,.07)" vertical={false} /><XAxis dataKey="name" stroke="#849585" tickLine={false} axisLine={false} /><YAxis domain={[0, 1]} stroke="#849585" tickFormatter={(value) => `${Math.round(value * 100)}%`} /><Tooltip contentStyle={{ background: "#201f1f", border: "1px solid rgba(255,255,255,.14)", borderRadius: 8 }} formatter={(value) => typeof value === "number" ? percent(value) : "N/D"} /><Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" radius={[5, 5, 0, 0]} /><Bar dataKey="macroF1" name="Macro-F1" fill="#00ff88" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel>
    </div>

    <section className="glass-panel mt-6 overflow-x-auto rounded-xl p-6"><h2 className="text-lg font-semibold text-zinc-100">Experimentos guardados</h2><p className="mt-1 text-sm text-zinc-500">Resultados de validación cruzada interna. No se comparan como si usaran necesariamente el mismo conjunto de datos.</p><table className="mt-5 w-full min-w-[620px] text-left text-sm"><thead className="border-b border-white/10 text-xs uppercase tracking-wider text-zinc-500"><tr><th className="pb-3">Experimento</th><th className="pb-3">Registros</th><th className="pb-3">Accuracy CV</th><th className="pb-3">Macro-F1 CV</th><th className="pb-3">Estado</th></tr></thead><tbody>{savedExperiments.map((experiment) => <tr key={experiment.name} className="border-b border-white/5 text-zinc-300 last:border-0"><td className="py-4 font-medium">{experiment.name}</td><td className="py-4 text-zinc-400">{experiment.records ?? "—"}</td><td className="py-4">{percent(experiment.accuracy)}</td><td className="py-4">{percent(experiment.macroF1)}</td><td className="py-4 text-zinc-500">{experiment.name === "holdout_bert" ? "Promovido" : "Archivado"}</td></tr>)}</tbody></table></section>
  </div>;
}

function MetricCard({ label, value, color }: { label: string; value?: number; color: string }) {
  const numericValue = typeof value === "number" ? value : 0;
  const percentage = Math.round(numericValue * 100);
  return <section className="glass-panel relative overflow-hidden rounded-xl p-5"><p className="eyebrow text-zinc-500">{label}</p><div className="mt-5 flex items-end justify-between"><p className="mono text-3xl font-semibold" style={{ color }}>{percent(value)}</p><div className="h-12 w-24 overflow-hidden"><svg viewBox="0 0 100 50" className="h-full w-full"><path d="M 10 45 A 40 40 0 0 1 90 45" fill="none" stroke="rgba(255,255,255,.1)" strokeWidth="7" /><path d="M 10 45 A 40 40 0 0 1 90 45" fill="none" stroke={color} strokeWidth="7" strokeLinecap="round" pathLength="100" strokeDasharray="100" strokeDashoffset={100 - percentage} /></svg></div></div></section>;
}

function ChartPanel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return <section className="glass-panel rounded-xl p-6"><div className="mb-5 flex items-start gap-3"><span className="rounded-lg bg-emerald-400/10 p-2 text-emerald-300"><BarChart3 size={18} /></span><div><h2 className="font-semibold text-zinc-100">{title}</h2><p className="mt-1 text-xs text-zinc-500">{subtitle}</p></div></div>{children}</section>;
}
