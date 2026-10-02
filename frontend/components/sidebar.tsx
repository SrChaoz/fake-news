"use client";

import { BarChart3, FlaskConical, Menu, Search, X } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const navigation = [
  { href: "/", label: "Analizar contenido", icon: Search },
  { href: "/history", label: "Historial", icon: BarChart3 },
  { href: "/experiments", label: "Evaluación del modelo", icon: FlaskConical },
];

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return <>
    <button aria-label="Abrir navegación" onClick={() => setOpen(true)} className="fixed left-4 top-4 z-30 rounded-md border border-slate-200 bg-white p-2 text-[var(--puce-blue)] shadow-sm lg:hidden"><Menu size={20} /></button>
    <aside className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-slate-200 bg-white py-6 transition-transform duration-200 lg:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <button aria-label="Cerrar navegación" onClick={() => setOpen(false)} className="absolute right-4 top-4 text-slate-500 lg:hidden"><X size={20} /></button>
      <Link href="/" className="mb-8 flex items-center gap-3 px-6"><Image src="/puce-manabi-logo.png" alt="PUCE Manabí" width={56} height={56} priority /><strong className="text-sm text-[var(--puce-blue)]">PUCE Manabí</strong></Link>
      <p className="px-6 text-xs font-medium leading-5 text-slate-500">Bilingual Climate Misinformation Detection System</p>
      <nav className="mt-6 space-y-1 px-3">{navigation.map(({ href, label, icon: Icon }) => { const active = pathname === href; return <Link key={href} href={href} onClick={() => setOpen(false)} className={`flex items-center gap-3 rounded-md px-3 py-3 text-sm font-medium transition ${active ? "bg-sky-50 text-[var(--puce-blue)]" : "text-slate-600 hover:bg-slate-50 hover:text-[var(--puce-blue)]"}`}><Icon size={19} /><span>{label}</span></Link>; })}</nav>
      <div className="mt-auto mx-6 border-t border-slate-200 pt-4 text-xs text-slate-500"><span className="inline-flex items-center gap-2"><i className="h-2 w-2 rounded-full bg-emerald-500" />Servicio local disponible</span></div>
    </aside>{open && <button aria-label="Cerrar navegación" onClick={() => setOpen(false)} className="fixed inset-0 z-30 bg-slate-900/30 lg:hidden" />}
  </>;
}
