import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";
import { ToastProvider } from "@/components/toast";

export const metadata: Metadata = { title: "Climate Veritas", description: "Detector de desinformación ambiental" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="es"><body><div className="ambient-grid fixed inset-0 -z-10" /><div className="orb -left-48 top-1/4 -z-10 bg-emerald-400" /><div className="orb -right-64 -top-64 -z-10 bg-blue-500" /><ToastProvider><Sidebar /><main className="min-h-screen px-5 pb-12 pt-24 lg:ml-20 lg:px-10 lg:pt-12">{children}</main></ToastProvider></body></html>;
}
