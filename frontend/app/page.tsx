"use client";

import { Link2, Send, Sparkles } from "lucide-react";
import { useState } from "react";
import { ResultCard } from "@/components/result-card";
import { api, ApiError } from "@/lib/api";
import type { PredictionResponse } from "@/lib/types";
import { useToast } from "@/components/toast";

export default function AnalyzerPage() {
  const [text, setText] = useState(""); const [result, setResult] = useState<PredictionResponse | null>(null); const [loading, setLoading] = useState(false); const { notify } = useToast();
  async function analyze() { if (!text.trim()) { notify("Escribe una publicación antes de analizarla.", "error"); return; } setLoading(true); setResult(null); try { setResult(await api.predict(text.trim())); notify("Análisis completado y guardado en el historial."); } catch (error) { notify(error instanceof ApiError ? error.message : "No se pudo analizar la publicación.", "error"); } finally { setLoading(false); } }
  return <div className="mx-auto max-w-7xl space-y-6">
    <header className="flex items-end justify-between border-b border-white/10 pb-6"><div><p className="eyebrow text-emerald-300">AI VERIFICATION SYSTEM · v2.4</p><h1 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-100 sm:text-4xl">Climate News Verification</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">Analiza afirmaciones climáticas con evidencia, ontologías ambientales y un clasificador híbrido.</p></div><span className="hidden items-center gap-2 text-xs text-emerald-300 md:flex"><i className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#00ff88]" />LIVE ENGINE</span></header>
    <section className="glass-panel relative overflow-hidden rounded-xl p-6 sm:p-8"><div className="absolute -left-12 -top-12 h-40 w-40 rounded-full bg-emerald-400/10 blur-3xl" /><label htmlFor="publication" className="eyebrow relative mb-4 block text-zinc-400">INPUT SOURCE MATERIAL</label><textarea id="publication" value={text} onChange={(e) => setText(e.target.value)} placeholder="Pega una noticia, publicación o afirmación climática para verificar..." className="relative min-h-44 w-full resize-y rounded-lg border border-white/10 bg-black/40 p-5 text-sm leading-6 text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-emerald-400/70 focus:shadow-[0_0_20px_rgba(0,255,136,.08)]" /><div className="relative mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><button className="inline-flex w-fit items-center gap-2 rounded border border-white/10 bg-white/5 px-3 py-2 text-xs text-zinc-400 hover:text-emerald-300"><Link2 size={15} />Adjuntar URL</button><button onClick={analyze} disabled={loading} className="inline-flex items-center justify-center gap-2 rounded bg-emerald-400 px-6 py-3 text-xs font-bold uppercase tracking-wider text-emerald-950 shadow-[0_0_18px_rgba(0,255,136,.24)] transition hover:bg-emerald-300 disabled:opacity-50">{loading ? <><Sparkles className="animate-pulse" size={16} />Procesando</> : <><Send size={16} />Ejecutar análisis</>}</button></div></section>
    {result ? <ResultCard result={result} /> : <div className="grid gap-5 md:grid-cols-3"><Info title="Evidence retrieval" text="Contrasta la afirmación con fuentes climáticas verificadas." /><Info title="Ontology checks" text="Detecta conflictos con relaciones ambientales conocidas." /><Info title="Hybrid decision" text="Combina la evidencia con la predicción del modelo." /></div>}
  </div>;
}
function Info({ title, text }: { title: string; text: string }) { return <div className="rounded-xl border border-white/10 bg-zinc-900/40 p-5"><p className="eyebrow text-emerald-300">{title}</p><p className="mt-3 text-sm leading-6 text-zinc-400">{text}</p></div>; }
