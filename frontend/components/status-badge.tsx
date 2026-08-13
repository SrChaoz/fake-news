import { CheckCircle2, ShieldAlert } from "lucide-react";
import type { PredictionLabel } from "@/lib/types";

export function StatusBadge({ label }: { label: PredictionLabel }) {
  const real = label === "REAL";
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${real ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300" : "border-rose-400/30 bg-rose-400/10 text-rose-300"}`}>{real ? <CheckCircle2 size={14} /> : <ShieldAlert size={14} />}{label}</span>;
}
