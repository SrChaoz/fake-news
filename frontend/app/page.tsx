"use client";

import { Link2, Send, Sparkles } from "lucide-react";
import { useState } from "react";
import { ResultCard } from "@/components/result-card";
import { AnalysisProgress } from "@/components/analysis-progress";
import { api, ApiError } from "@/lib/api";
import type { PredictionResponse } from "@/lib/types";
import { useToast } from "@/components/toast";

export default function AnalyzerPage() {
  const [text, setText] = useState(""); const [result, setResult] = useState<PredictionResponse | null>(null); const [loading, setLoading] = useState(false); const { notify } = useToast();
  async function analyze() { if (!text.trim()) { notify("Escribe una publicación antes de analizarla.", "error"); return; } setLoading(true); setResult(null); try { setResult(await api.predict(text.trim())); notify("Análisis completado y guardado en el historial."); } catch (error) { notify(error instanceof ApiError ? error.message : "No se pudo analizar la publicación.", "error"); } finally { setLoading(false); } }
  return <div className="mx-auto max-w-7xl space-y-6">
    <header className="flex flex-col justify-between gap-4 border-b border-slate-200 pb-6 sm:flex-row sm:items-end"><div><p className="eyebrow text-[var(--puce-cyan)]">PUCE Manabí</p><h1 className="mt-3 text-2xl font-semibold tracking-tight text-[var(--puce-ink)] sm:text-3xl">Bilingual Climate Misinformation Detection System</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">Analiza afirmaciones climáticas con evidencia, reglas ontológicas y un clasificador híbrido.</p></div><span className="inline-flex items-center gap-2 text-xs font-medium text-slate-600"><i className="h-2 w-2 rounded-full bg-emerald-500" />Sistema disponible</span></header>
    <section className="glass-panel rounded-lg p-6 sm:p-8"><label htmlFor="publication" className="eyebrow mb-3 block text-[var(--puce-blue)]">Contenido para analizar</label><p className="mb-4 text-sm text-slate-600">Pega una noticia, publicación o afirmación climática. El sistema mostrará la clasificación y su evidencia disponible.</p><textarea id="publication" value={text} onChange={(e) => setText(e.target.value)} placeholder="Ejemplo: El CO₂ no contribuye al calentamiento global..." className="min-h-44 w-full resize-y rounded-md border border-slate-300 bg-white p-4 text-base leading-6 text-slate-800 outline-none placeholder:text-slate-400 focus:border-[var(--puce-cyan)] focus:ring-2 focus:ring-sky-100" /><div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><span className="inline-flex items-center gap-2 text-xs text-slate-500"><Link2 size={15} />Verificación basada en fuentes y reglas ambientales</span><button onClick={analyze} disabled={loading} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-md bg-[var(--puce-blue)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--puce-blue-deep)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--puce-cyan)] disabled:cursor-not-allowed disabled:opacity-50">{loading ? <><Sparkles className="animate-pulse" size={16} />Analizando</> : <><Send size={16} />Analizar contenido</>}</button></div></section>
    {loading ? <AnalysisProgress /> : result ? <ResultCard result={result} /> : <div className="grid gap-4 md:grid-cols-3"><Info title="Evidencia disponible" text="Contrasta la afirmación con fuentes climáticas curadas." /><Info title="Reglas ambientales" text="Identifica conflictos con relaciones ambientales conocidas." /><Info title="Resultado explicable" text="Combina evidencia y predicción del modelo." /></div>}
  </div>;
}
function Info({ title, text }: { title: string; text: string }) { return <div className="rounded-lg border border-slate-200 bg-white p-5"><p className="text-sm font-semibold text-[var(--puce-blue)]">{title}</p><p className="mt-2 text-sm leading-6 text-slate-600">{text}</p></div>; }
