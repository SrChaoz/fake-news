"use client";

import { CheckCircle2, XCircle } from "lucide-react";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

type Toast = { id: number; message: string; type: "success" | "error" };
type ToastContextValue = { notify: (message: string, type?: Toast["type"]) => void };
const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const notify = useCallback((message: string, type: Toast["type"] = "success") => { const id = Date.now(); setToasts((current) => [...current, { id, message, type }]); window.setTimeout(() => setToasts((current) => current.filter((toast) => toast.id !== id)), 3800); }, []);
  const value = useMemo(() => ({ notify }), [notify]);
  return <ToastContext.Provider value={value}>{children}<div className="fixed right-5 top-5 z-50 space-y-3">{toasts.map((toast) => <div key={toast.id} className={`flex max-w-sm items-center gap-3 rounded-lg border px-4 py-3 text-sm font-medium shadow-panel ${toast.type === "success" ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-rose-200 bg-rose-50 text-rose-800"}`}>{toast.type === "success" ? <CheckCircle2 size={18} /> : <XCircle size={18} />}<span>{toast.message}</span></div>)}</div></ToastContext.Provider>;
}

export function useToast() { const context = useContext(ToastContext); if (!context) throw new Error("useToast debe usarse dentro de ToastProvider."); return context; }
