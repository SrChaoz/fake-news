from pathlib import Path
import unittest


class LocalPostgresScriptTests(unittest.TestCase):
    def test_uses_project_local_loopback_cluster(self) -> None:
        script = Path("scripts/local_postgres.sh").read_text(encoding="utf-8")

        self.assertIn('DATA_DIRECTORY="$PROJECT_ROOT/.local/postgres/data"', script)
        self.assertIn('SOCKET_DIRECTORY="$PROJECT_ROOT/.local/postgres/socket"', script)
        self.assertIn('PORT="54329"', script)
        self.assertIn('listen_addresses=127.0.0.1', script)
        self.assertIn('init|start|stop|status', script)

    def test_gitignore_excludes_local_postgres_state(self) -> None:
        self.assertIn(".local/", Path(".gitignore").read_text(encoding="utf-8"))

    def test_setup_preserves_configuration_and_installs_exact_runtime_model(self) -> None:
        script = Path("scripts/setup_local.sh").read_text(encoding="utf-8")

        self.assertIn('if [ ! -f "$PROJECT_ROOT/.env" ]; then', script)
        self.assertIn('if [ ! -f "$FRONTEND_DIRECTORY/.env.local" ]; then', script)
        self.assertIn("npm ci", script)
        self.assertIn('AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")', script)
        self.assertIn('AutoModel.from_pretrained("prajjwal1/bert-tiny")', script)

    def test_web_launchers_bind_to_loopback_only(self) -> None:
        backend = Path("scripts/start_backend.sh").read_text(encoding="utf-8")
        frontend = Path("scripts/start_frontend.sh").read_text(encoding="utf-8")
        self.assertIn("uvicorn app.main:app --host 127.0.0.1 --port 8000", backend)
        self.assertIn("npm run dev -- --hostname 127.0.0.1 --port 3000", frontend)

    def test_model_runtime_pins_its_serialization_version(self) -> None:
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        self.assertIn("scikit-learn==1.8.0", requirements)


if __name__ == "__main__":
    unittest.main()
