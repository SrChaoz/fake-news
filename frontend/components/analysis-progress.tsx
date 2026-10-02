"use client";

import { BrainCircuit, Check, LoaderCircle, SearchCheck, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

const stages = [
  { label: "Preparando contenido", detail: "Identificando la afirmación principal.", icon: BrainCircuit },
  { label: "Contrastando evidencia", detail: "Consultando fuentes y reglas ambientales.", icon: SearchCheck },
  { label: "Generando resultado", detail: "Integrando evidencia y predicción del modelo.", icon: Sparkles },
];

export function AnalysisProgress() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const first = window.setTimeout(() => setStage(1), 750);
    const second = window.setTimeout(() => setStage(2), 1550);
    return () => { window.clearTimeout(first); window.clearTimeout(second); };
  }, []);

  return <section className="glass-panel rounded-lg p-6 sm:p-8" role="status" aria-live="polite" aria-label="Análisis de contenido en progreso">
    <div className="flex items-start gap-4"><span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-sky-50 text-[var(--puce-blue)]"><LoaderCircle size={22} className="animate-spin motion-reduce:animate-none" /></span><div><p className="eyebrow text-[var(--puce-cyan)]">Análisis en curso</p><h2 className="mt-1 text-lg font-semibold text-[var(--puce-ink)]">El sistema está verificando la afirmación</h2><p className="mt-1 text-sm leading-6 text-slate-600">Este proceso puede tardar unos segundos. Mantén esta página abierta.</p></div></div>
    <ol className="mt-6 grid gap-3 sm:grid-cols-3">{stages.map(({ label, detail, icon: Icon }, index) => { const complete = index < stage; const active = index === stage; return <li key={label} className={`rounded-md border p-4 transition-colors duration-200 ${active ? "border-sky-200 bg-sky-50" : complete ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-white"}`}><div className={`mb-3 flex h-7 w-7 items-center justify-center rounded-full ${active ? "bg-[var(--puce-blue)] text-white" : complete ? "bg-emerald-600 text-white" : "bg-slate-100 text-slate-500"}`}>{complete ? <Check size={16} /> : <Icon size={16} className={active ? "animate-pulse motion-reduce:animate-none" : ""} />}</div><p className="text-sm font-semibold text-[var(--puce-ink)]">{label}</p><p className="mt-1 text-xs leading-5 text-slate-600">{detail}</p></li>; })}</ol>
  </section>;
}
