from pathlib import Path
import unittest


class InstitutionalBrandingTests(unittest.TestCase):
    def test_puce_logo_and_official_name_are_present(self) -> None:
        self.assertTrue(Path("frontend/public/puce-manabi-logo.png").is_file())
        layout = Path("frontend/app/layout.tsx").read_text(encoding="utf-8")
        sidebar = Path("frontend/components/sidebar.tsx").read_text(encoding="utf-8")
        self.assertIn("Bilingual Climate Misinformation Detection System", layout)
        self.assertIn("/puce-manabi-logo.png", sidebar)
        self.assertIn("PUCE Manabí", sidebar)

    def test_institutional_label_is_replaced_by_puce_manabi(self) -> None:
        sidebar = Path("frontend/components/sidebar.tsx").read_text(encoding="utf-8")
        analyzer = Path("frontend/app/page.tsx").read_text(encoding="utf-8")

        self.assertNotIn("Sistema institucional", sidebar)
        self.assertNotIn("Sistema institucional", analyzer)
        self.assertIn("PUCE Manabí", sidebar)

    def test_institutional_palette_replaces_neon_interface(self) -> None:
        styles = Path("frontend/app/globals.css").read_text(encoding="utf-8")
        self.assertIn("--puce-blue", styles)
        self.assertIn("--puce-cyan", styles)
        self.assertNotIn("ambient-grid", styles)
        self.assertNotIn(".orb", styles)

    def test_data_views_use_legible_light_surfaces(self) -> None:
        """Regression for the low-contrast institutional pages reported in the UI."""
        result_card = Path("frontend/components/result-card.tsx").read_text(encoding="utf-8")
        history = Path("frontend/app/history/page.tsx").read_text(encoding="utf-8")
        experiments = Path("frontend/app/experiments/page.tsx").read_text(encoding="utf-8")
        badge = Path("frontend/components/status-badge.tsx").read_text(encoding="utf-8")

        for source in (result_card, history, experiments):
            self.assertNotIn("bg-zinc-900", source)
            self.assertNotIn("text-zinc-100", source)
        self.assertIn("text-[var(--puce-ink)]", result_card)
        self.assertIn("bg-slate-50", history)
        self.assertIn("border-sky-200", experiments)
        self.assertIn("text-emerald-700", badge)
        self.assertIn("text-rose-700", badge)

    def test_analyzer_has_an_accessible_progress_panel_while_processing(self) -> None:
        analyzer = Path("frontend/app/page.tsx").read_text(encoding="utf-8")
        progress = Path("frontend/components/analysis-progress.tsx")

        self.assertTrue(progress.is_file())
        source = progress.read_text(encoding="utf-8")
        self.assertIn("Preparando contenido", source)
        self.assertIn("Contrastando evidencia", source)
        self.assertIn("Generando resultado", source)
        self.assertIn('role="status"', source)
        self.assertIn("AnalysisProgress", analyzer)
        self.assertIn("loading ? <AnalysisProgress", analyzer)


if __name__ == "__main__":
    unittest.main()
