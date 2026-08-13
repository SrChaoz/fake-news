"use client";

import { BarChart3, FlaskConical, Menu, Search, ShieldCheck, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const navigation = [
  { href: "/", label: "Analyzer", icon: Search },
  { href: "/history", label: "History", icon: BarChart3 },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
];

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return <>
    <button aria-label="Abrir navegación" onClick={() => setOpen(true)} className="fixed left-4 top-4 z-30 rounded-lg border border-white/10 bg-zinc-950/80 p-2 text-zinc-200 lg:hidden"><Menu size={20} /></button>
    <aside className={`group fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-white/10 bg-[#0a0a0a]/80 py-7 backdrop-blur-xl transition-[width,transform] duration-300 lg:w-20 lg:hover:w-60 ${open ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}>
      <button aria-label="Cerrar navegación" onClick={() => setOpen(false)} className="absolute right-4 top-4 text-zinc-400 lg:hidden"><X size={20} /></button>
      <Link href="/" className="mb-12 flex items-center gap-4 px-5 whitespace-nowrap"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-emerald-400/20 bg-emerald-400/10 text-emerald-300 shadow-[0_0_22px_rgba(0,255,136,.15)]"><ShieldCheck size={21} /></span><span className="eyebrow text-emerald-300 opacity-100 lg:opacity-0 lg:group-hover:opacity-100">Climate Veritas</span></Link>
      <nav className="space-y-2 px-3">{navigation.map(({ href, label, icon: Icon }) => { const active = pathname === href; return <Link key={href} href={href} onClick={() => setOpen(false)} className={`flex items-center gap-4 rounded-xl border-l-4 px-3 py-3 text-sm transition ${active ? "border-emerald-400 bg-emerald-400/10 text-emerald-300 shadow-[0_0_18px_rgba(0,255,136,.16)]" : "border-transparent text-zinc-500 hover:bg-white/5 hover:text-zinc-100"}`}><Icon size={20} className="shrink-0" /><span className="eyebrow whitespace-nowrap opacity-100 lg:opacity-0 lg:group-hover:opacity-100">{label}</span></Link>; })}</nav>
      <div className="mt-auto mx-3 border-t border-white/10 pt-4 text-[10px] leading-4 text-zinc-500 lg:overflow-hidden"><span className="hidden whitespace-nowrap lg:group-hover:block">SYSTEM STATUS · ONLINE</span><span className="block text-emerald-400">●</span></div>
    </aside>{open && <button aria-label="Cerrar navegación" onClick={() => setOpen(false)} className="fixed inset-0 z-30 bg-black/60 lg:hidden" />}
  </>;
}
