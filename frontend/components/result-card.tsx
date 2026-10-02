"use client";

import { ChevronDown, ChevronUp, Database, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { StatusBadge } from "@/components/status-badge";
import type { PredictionResponse } from "@/lib/types";

export function ResultCard({ result }: { result: PredictionResponse }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const confidence = Math.round(result.confidence_score * 100);
  const real = result.prediction === "REAL";
  const evidenceLabel = result.verification_status === "SUPPORTED" ? "Evidencia recuperada: respalda" : result.verification_status === "REFUTED" ? "Evidencia recuperada: refuta" : result.verification_status === "ONTOLOGY_CONFLICT" ? "Regla ontológica crítica" : result.verification_status === "ONTOLOGY_SUPPORT" ? "Relación respaldada por regla ontológica" : "Evidencia insuficiente";

  return <section className="glass-panel mt-8 rounded-lg p-6">
    <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-start">
      <div><p className="eyebrow mb-3 text-[var(--puce-blue)]">Resultado del análisis</p><div className="flex items-center gap-3"><StatusBadge label={result.prediction} /><span className="text-sm text-slate-600">Categoría: {result.category}</span></div></div>
      <div className="min-w-44"><div className="mb-2 flex justify-between text-sm"><span className="text-slate-600">Confianza</span><span className="font-semibold text-[var(--puce-ink)]">{confidence}%</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-200"><div className={`h-full rounded-full transition-all ${real ? "bg-emerald-600" : "bg-rose-600"}`} style={{ width: `${confidence}%` }} /></div></div>
    </div>
    <div className="mt-6 rounded-lg border border-slate-200 bg-slate-50 p-4"><div className="mb-2 flex items-center gap-2 text-sm font-semibold text-[var(--puce-ink)]"><ShieldCheck size={17} className="text-emerald-700" />Explicación automática</div><p className="text-sm leading-6 text-slate-700">{result.explanation}</p><p className="mt-3 text-xs text-slate-600">{evidenceLabel}</p><p className="mt-1 text-xs text-slate-600">Afirmación analizada: {result.extracted_claim}</p></div>
    {result.evidence.length > 0 && <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 p-4"><p className="text-sm font-semibold text-emerald-800">Evidencia recuperada</p>{result.evidence.slice(0, 1).map((evidence) => <div key={`${evidence.claim}-${evidence.similarity}`} className="mt-2"><p className="text-sm text-slate-700">{evidence.evidence_text}</p><p className="mt-2 text-xs text-slate-600">Fuente: {evidence.source_dataset} · Similitud: {Math.round(evidence.similarity * 100)}%</p>{evidence.source_url && <a className="mt-1 inline-block text-xs font-medium text-[var(--puce-blue)] hover:underline" href={evidence.source_url} target="_blank" rel="noreferrer">Abrir fuente</a>}</div>)}</div>}
    <button onClick={() => setDetailsOpen((open) => !open)} className="mt-5 flex min-h-11 w-full items-center justify-between rounded-md border border-slate-300 bg-white px-4 py-3 text-sm font-medium text-slate-700 transition hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--puce-cyan)]"><span className="flex items-center gap-2"><Database size={16} />Entidades y conflictos ontológicos</span>{detailsOpen ? <ChevronUp size={17} /> : <ChevronDown size={17} />}</button>
    {detailsOpen && <div className="mt-3 grid gap-4 lg:grid-cols-2"><DetailList title="Entidades ambientales" empty="No se detectaron entidades ambientales específicas." entries={result.detected_entities.map((entity) => ({ title: entity.text, detail: [entity.label, entity.normalized].filter(Boolean).join(" · ") }))} /><DetailList title="Conflictos ontológicos" empty="No se detectaron conflictos explícitos." entries={result.ontological_conflicts.map((conflict) => ({ title: conflict.subject ?? "Conflicto", detail: conflict.reason ?? conflict.expected_relation ?? "" }))} /></div>}
  </section>;
}

function DetailList({ title, empty, entries }: { title: string; empty: string; entries: Array<{ title: string; detail: string }> }) {
  return <div className="rounded-lg border border-slate-200 bg-white p-4"><h3 className="mb-3 text-sm font-semibold text-[var(--puce-ink)]">{title}</h3>{entries.length === 0 ? <p className="text-sm text-slate-600">{empty}</p> : <ul className="space-y-2">{entries.map((entry, index) => <li key={`${entry.title}-${index}`} className="rounded-md bg-slate-50 px-3 py-2"><p className="text-sm text-slate-800">{entry.title}</p>{entry.detail && <p className="mt-1 text-xs text-slate-600">{entry.detail}</p>}</li>)}</ul>}</div>;
}
