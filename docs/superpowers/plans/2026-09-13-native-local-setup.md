# Native Local Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Climate Veritas reproducibly runnable on one local machine with a user-owned PostgreSQL instance, native Python/Node dependencies, the promoted inference model, and validated API/frontend startup commands.

**Architecture:** A small set of shell launchers owns local lifecycle concerns: a user-owned PostgreSQL cluster lives under `.local/postgres`, the backend runs from `.venv`, and Next.js runs from `frontend/`. A Python database bootstrapper uses the configured `DATABASE_URL` to create only a missing database and load the versioned schema/data only when the dataset tables are empty; it never deletes an existing database or prediction history.

**Tech Stack:** PostgreSQL 18 client/server utilities, Python 3.12+, Bash, virtualenv/pip, FastAPI/Uvicorn, Hugging Face Transformers, Node.js 20+/npm, Next.js 15, TypeScript.

**Spec:** `docs/superpowers/specs/2026-09-13-native-local-setup-design.md`

## Global Constraints

- The setup is native and local only; no Docker, remote deployment, or external publication.
- Local PostgreSQL data must reside under `.local/postgres`, bind only to `127.0.0.1`, and use port `54329` to avoid modifying or colliding with a system PostgreSQL service.
- `.env`, `frontend/.env.local`, `.venv`, frontend dependencies, and `.local/` remain untracked.
- Never remove, recreate, truncate, or overwrite an existing database, dataset, prediction history, or promoted model artifact.
- Inference continues using the already-promoted `app/models/random_forest_model.pkl`, `app/models/tfidf.pkl`, and `prajjwal1/bert-tiny`; training and experiment promotion are out of scope.
- All launchers bind the web services to loopback addresses only.

---

## File structure

| File | Responsibility |
| --- | --- |
| `.gitignore` | Ignore local PostgreSQL state. |
| `scripts/local_postgres.sh` | Initialize, start, stop, and report a user-owned local PostgreSQL cluster. |
| `scripts/bootstrap_database.py` | Safely create and seed the configured local application database. |
| `scripts/setup_local.sh` | Create local configuration, install locked dependencies, bootstrap data, and cache the required encoder. |
| `scripts/start_backend.sh` | Start PostgreSQL if needed, load `.env`, and run Uvicorn on loopback. |
| `scripts/start_frontend.sh` | Check frontend configuration and run Next.js on loopback. |
| `scripts/verify_local.sh` | Run focused regression checks, frontend type/build checks, launch the API temporarily, and execute the existing E2E verifier. |
| `tests/test_bootstrap_database.py` | Unit tests for safe database bootstrap decisions using mocked database connections. |
| `tests/test_local_scripts.py` | Static and behavior tests for launcher safety and required commands. |
| `README.md` | Replace manual setup instructions with native local quick-start, lifecycle, recovery, and validation commands. |

### Task 1: Add safe local PostgreSQL lifecycle management

**Files:**
- Modify: `.gitignore`
- Create: `scripts/local_postgres.sh`
- Test: `tests/test_local_scripts.py`

**Interfaces:**
- Produces: `scripts/local_postgres.sh init|start|stop|status`, exit code `0` on a healthy requested state and nonzero with an actionable message otherwise.
- Produces: a loopback-only cluster at `.local/postgres/data`, Unix socket directory `.local/postgres/socket`, and port `54329`.
- Consumes: PostgreSQL programs `initdb`, `pg_ctl`, and `pg_isready` available on `PATH`.

- [ ] **Step 1: Write the failing launcher tests**

```python
from pathlib import Path


def test_local_postgres_script_uses_project_local_loopback_cluster():
    script = Path("scripts/local_postgres.sh").read_text(encoding="utf-8")
    assert 'DATA_DIRECTORY="$PROJECT_ROOT/.local/postgres/data"' in script
    assert 'SOCKET_DIRECTORY="$PROJECT_ROOT/.local/postgres/socket"' in script
    assert 'PORT="54329"' in script
    assert 'listen_addresses=127.0.0.1' in script
    assert 'init|start|stop|status' in script


def test_gitignore_excludes_local_postgres_state():
    assert ".local/" in Path(".gitignore").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests/test_local_scripts.py -v`

Expected: FAIL because the script and `.local/` ignore rule do not exist.

- [ ] **Step 3: Implement the minimal lifecycle script**

Create a Bash script with `set -euo pipefail`, resolve `PROJECT_ROOT` from its own location, and set exactly:

```bash
DATA_DIRECTORY="$PROJECT_ROOT/.local/postgres/data"
SOCKET_DIRECTORY="$PROJECT_ROOT/.local/postgres/socket"
PORT="54329"
```

For `init`, create the directories and run `initdb --auth=trust --username="$(id -un)" --pgdata="$DATA_DIRECTORY"` only when `PG_VERSION` is absent. For `start`, call `init`, use `pg_isready -h 127.0.0.1 -p "$PORT"`, and otherwise call `pg_ctl start` with `-o "-h 127.0.0.1 -p $PORT -k $SOCKET_DIRECTORY -c listen_addresses=127.0.0.1"`. For `stop`, stop only this data directory with `pg_ctl stop`; for `status`, report the exact connection URL. Reject unknown commands with usage `init|start|stop|status`. Append `.local/` to `.gitignore`.

- [ ] **Step 4: Run the tests and shell validation**

Run: `python -m unittest tests/test_local_scripts.py -v && bash -n scripts/local_postgres.sh && scripts/local_postgres.sh init && scripts/local_postgres.sh start && scripts/local_postgres.sh status`

Expected: PASS; cluster reports healthy at `127.0.0.1:54329`.

- [ ] **Step 5: Commit the task**

```bash
git add .gitignore scripts/local_postgres.sh tests/test_local_scripts.py
git commit -m "feat: add native local postgres lifecycle"
```

### Task 2: Safely bootstrap schema and versioned data

**Files:**
- Create: `scripts/bootstrap_database.py`
- Create: `tests/test_bootstrap_database.py`

**Interfaces:**
- Consumes: `DATABASE_URL`, `docs/schema.sql`, `docs/database_data.sql`, and `psycopg`.
- Produces: `python scripts/bootstrap_database.py` creates the target database only if absent, applies schema, and loads versioned data only when `dataset_records` and `dataset_evidence` are both empty.
- Produces: exit code `0` for a successful bootstrap/no-op; nonzero for an invalid URL, unavailable PostgreSQL, or partially populated dataset tables.

- [ ] **Step 1: Write failing decision tests**

```python
import unittest
from scripts.bootstrap_database import seed_action


class SeedActionTests(unittest.TestCase):
    def test_empty_tables_are_seeded(self):
        self.assertEqual(seed_action(record_count=0, evidence_count=0), "seed")

    def test_populated_tables_are_preserved(self):
        self.assertEqual(seed_action(record_count=1451, evidence_count=451), "preserve")

    def test_partial_dataset_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "parcial"):
            seed_action(record_count=1, evidence_count=0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_bootstrap_database.py -v`

Expected: FAIL with `ModuleNotFoundError` because `scripts.bootstrap_database` does not exist.

- [ ] **Step 3: Implement the bootstrapper**

Implement `seed_action(record_count: int, evidence_count: int) -> Literal["seed", "preserve"]` with the exact decisions in the test. Parse `DATABASE_URL` with `urllib.parse.urlsplit`; reject schemes other than `postgresql`/`postgresql+psycopg` and database names that do not match `[A-Za-z0-9_-]+`. Connect to the maintenance database with `psycopg`, query `pg_database`, and issue `CREATE DATABASE` only if the target is missing. Do not issue `DROP DATABASE`, `TRUNCATE`, `DELETE`, or `ALTER ... RESET`.

Connect to the target database, execute `docs/schema.sql` with `psql --set ON_ERROR_STOP=1 --file`, count both dataset tables, and use `seed_action`. On `seed`, execute `docs/database_data.sql` with the same `psql` flags. On `preserve`, print that existing dataset and prediction history were left untouched. Raise a clear error for a partial dataset so a person can inspect it instead of silently mixing data.

- [ ] **Step 4: Run unit tests and an idempotency integration check**

Run: `.venv/bin/python -m unittest tests/test_bootstrap_database.py -v && DATABASE_URL="postgresql://$(id -un)@127.0.0.1:54329/fakenews-clima" .venv/bin/python scripts/bootstrap_database.py && DATABASE_URL="postgresql://$(id -un)@127.0.0.1:54329/fakenews-clima" .venv/bin/python scripts/bootstrap_database.py`

Expected: unit tests PASS; first execution creates/seeds; second reports preservation and does not change rows/history.

- [ ] **Step 5: Commit the task**

```bash
git add scripts/bootstrap_database.py tests/test_bootstrap_database.py
git commit -m "feat: add safe local database bootstrap"
```

### Task 3: Implement dependency, configuration, and model setup

**Files:**
- Create: `scripts/setup_local.sh`
- Modify: `.env.example`
- Modify: `frontend/.env.example`
- Modify: `tests/test_local_scripts.py`

**Interfaces:**
- Produces: `scripts/setup_local.sh` that creates `.venv`, installs `requirements.txt`, runs `npm ci` in `frontend/`, creates missing local env files, calls the DB lifecycle/bootstrap scripts, and caches `prajjwal1/bert-tiny`.
- Consumes: internet access only for package installation and first-time Hugging Face model retrieval.
- Guarantees: existing `.env` and `frontend/.env.local` are never overwritten.

- [ ] **Step 1: Add failing safety/configuration tests**

```python
def test_setup_does_not_overwrite_local_env_and_uses_lockfile():
    script = Path("scripts/setup_local.sh").read_text(encoding="utf-8")
    assert 'if [ ! -f "$PROJECT_ROOT/.env" ]; then' in script
    assert 'if [ ! -f "$FRONTEND_DIRECTORY/.env.local" ]; then' in script
    assert 'npm ci' in script
    assert 'AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")' in script
    assert 'AutoModel.from_pretrained("prajjwal1/bert-tiny")' in script
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests/test_local_scripts.py -v`

Expected: FAIL because `scripts/setup_local.sh` does not exist.

- [ ] **Step 3: Implement native setup**

Create `scripts/setup_local.sh` with `set -euo pipefail`. It must check `python`, `node`, `npm`, `initdb`, `pg_ctl`, and `psql` before doing work. Create `.venv` with `python -m venv .venv`; upgrade pip; install `requirements.txt`; run `npm ci` from `frontend/`; call `scripts/local_postgres.sh start`; write missing `.env` as `DATABASE_URL=postgresql://$(id -un)@127.0.0.1:54329/fakenews-clima`; copy the frontend example only if `.env.local` is missing; invoke `.venv/bin/python scripts/bootstrap_database.py`; then invoke a short Python heredoc that calls both exact `AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")` and `AutoModel.from_pretrained("prajjwal1/bert-tiny")` and prints the cached model identity.

Keep `.env.example` as a documented template for the local cluster URL and `frontend/.env.example` as `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`. Never print credentials.

- [ ] **Step 4: Run setup and prove repeatability**

Run: `bash -n scripts/setup_local.sh && scripts/setup_local.sh && scripts/setup_local.sh`

Expected: both runs complete; the second preserves environment files and reports an already seeded database/model cache.

- [ ] **Step 5: Commit the task**

```bash
git add scripts/setup_local.sh .env.example frontend/.env.example tests/test_local_scripts.py
git commit -m "feat: automate native local setup"
```

### Task 4: Add loopback launchers and full local verification

**Files:**
- Create: `scripts/start_backend.sh`
- Create: `scripts/start_frontend.sh`
- Create: `scripts/verify_local.sh`
- Modify: `tests/test_local_scripts.py`

**Interfaces:**
- Produces: `scripts/start_backend.sh` starts Uvicorn at `127.0.0.1:8000` after verifying `.env`, database health, and model artifacts.
- Produces: `scripts/start_frontend.sh` starts Next.js at `127.0.0.1:3000` using `frontend/.env.local`.
- Produces: `scripts/verify_local.sh` exits nonzero if a regression test, frontend type/build check, API health, or E2E check fails.

- [ ] **Step 1: Add failing launcher assertions**

```python
def test_web_launchers_bind_to_loopback_only():
    backend = Path("scripts/start_backend.sh").read_text(encoding="utf-8")
    frontend = Path("scripts/start_frontend.sh").read_text(encoding="utf-8")
    assert 'uvicorn app.main:app --host 127.0.0.1 --port 8000' in backend
    assert 'npm run dev -- --hostname 127.0.0.1 --port 3000' in frontend


def test_verifier_uses_existing_regressions_and_e2e():
    verifier = Path("scripts/verify_local.sh").read_text(encoding="utf-8")
    assert 'scripts/test_negation_rules.py' in verifier
    assert 'scripts/test_evidence_engine.py' in verifier
    assert 'npx tsc --noEmit' in verifier
    assert 'npm run build' in verifier
    assert 'scripts/verify_e2e.py' in verifier
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests/test_local_scripts.py -v`

Expected: FAIL because the launchers and verifier do not exist.

- [ ] **Step 3: Implement launchers and verifier**

Each script uses `set -euo pipefail` and resolves its root directory. `start_backend.sh` must require `.venv/bin/python` and `.env`, source only that local file with `set -a`, call `scripts/local_postgres.sh start`, run `pg_isready -h 127.0.0.1 -p 54329`, then `exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`.

`start_frontend.sh` must require `frontend/node_modules` and `frontend/.env.local`, then `cd frontend` and `exec npm run dev -- --hostname 127.0.0.1 --port 3000`.

`verify_local.sh` must run the two existing Python regression scripts, `npx tsc --noEmit`, and `npm run build`. Then start Uvicorn without `--reload` in the background, use a `trap` to always terminate only that PID, poll `/health` for at most 30 seconds, and run `.venv/bin/python scripts/verify_e2e.py --base-url http://127.0.0.1:8000 --timeout 90`.

- [ ] **Step 4: Run the complete validation**

Run: `python -m unittest tests/test_local_scripts.py tests/test_bootstrap_database.py -v && scripts/verify_local.sh`

Expected: all unit/regression, frontend, API, persistence, metrics, experiment, and ablation checks PASS.

- [ ] **Step 5: Commit the task**

```bash
git add scripts/start_backend.sh scripts/start_frontend.sh scripts/verify_local.sh tests/test_local_scripts.py
git commit -m "feat: add native service launchers and verification"
```

### Task 5: Document the operational handoff

**Files:**
- Modify: `README.md`
- Modify: `frontend/README.md`

**Interfaces:**
- Produces: a native local quick start containing exactly `./scripts/setup_local.sh`, `./scripts/start_backend.sh`, and `./scripts/start_frontend.sh`.
- Produces: recovery commands for database status/stop, model-cache failure, and validation.

- [ ] **Step 1: Write failing documentation assertions**

```python
def test_readme_documents_native_one_machine_workflow():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "./scripts/setup_local.sh" in readme
    assert "./scripts/start_backend.sh" in readme
    assert "./scripts/start_frontend.sh" in readme
    assert "./scripts/verify_local.sh" in readme
```

- [ ] **Step 2: Run the assertion to verify it fails**

Run: `python -m unittest tests/test_local_scripts.py -v`

Expected: FAIL because the README does not yet document the new workflow.

- [ ] **Step 3: Update documentation**

Replace the existing manual quick-start with a native local section: prerequisites (Python, Node/npm, PostgreSQL utilities), the one-time setup command, separate backend/frontend startup terminals, URLs for dashboard/API docs, verification command, and lifecycle commands `scripts/local_postgres.sh status|stop`. State that model setup needs network the first time and that setup will never delete existing data. Update the frontend README to point to `scripts/start_frontend.sh` while retaining its environment-variable reference.

- [ ] **Step 4: Run documentation and final state checks**

Run: `python -m unittest tests/test_local_scripts.py -v && git diff --check && git status --short`

Expected: tests PASS, no whitespace errors, and only intended source/documentation changes (no `.env`, `.venv`, `.local`, or `node_modules`).

- [ ] **Step 5: Commit the task**

```bash
git add README.md frontend/README.md tests/test_local_scripts.py
git commit -m "docs: document native local workflow"
```

## Plan self-review

- **Spec coverage:** Task 1 provides a local, loopback-only PostgreSQL service; Task 2 restores schema/data without destructive behavior; Task 3 installs dependencies, writes only missing local config, and caches BERT; Task 4 validates backend/frontend and the end-to-end persistence flow; Task 5 provides the operational handoff. The plan explicitly preserves the promoted model and excludes training/deployment.
- **Placeholder scan:** no TBD/TODO or deferred implementation instructions remain.
- **Interface consistency:** Task 1 defines the cluster/port used by Tasks 2–4; Task 2 consumes `DATABASE_URL` written by Task 3; Task 4 consumes all prior scripts; Task 5 documents the exact Task 3/4 commands.
