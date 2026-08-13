import { LoaderCircle } from "lucide-react";

export function LoadingSpinner({ label = "Cargando..." }: { label?: string }) {
  return <div className="flex items-center gap-2 py-8 text-sm text-zinc-400"><LoaderCircle size={18} className="animate-spin text-emerald-400" />{label}</div>;
}
