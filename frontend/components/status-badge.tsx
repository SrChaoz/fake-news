import { CheckCircle2, ShieldAlert } from "lucide-react";
import type { PredictionLabel } from "@/lib/types";

export function StatusBadge({ label }: { label: PredictionLabel }) {
  const real = label === "REAL";
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${real ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-rose-200 bg-rose-50 text-rose-700"}`}>{real ? <CheckCircle2 size={14} /> : <ShieldAlert size={14} />}{label}</span>;
}
