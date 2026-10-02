import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";
import { ToastProvider } from "@/components/toast";

export const metadata: Metadata = { title: "Bilingual Climate Misinformation Detection System | PUCE Manabí", description: "Detección de desinformación climática de PUCE Manabí" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="es"><body><ToastProvider><Sidebar /><main className="min-h-screen px-5 pb-12 pt-24 lg:ml-72 lg:px-10 lg:pt-10">{children}</main></ToastProvider></body></html>;
}
