# Climate Claim Analysis with Machine Learning

An academic system that analyzes climate-related claims using a hybrid machine-learning classifier, environmental entity recognition, ontology-based rules, and curated evidence retrieval. Each result is designed to be reviewed against the evidence behind it.

> It is a research and educational tool. It does not replace scientific, journalistic, or professional fact-checking.

## What it does

- Analyzes climate-related text through a FastAPI REST API.
- Extracts environmental entities and checks them against ontology rules.
- Uses TF-IDF, text features, BERT-tiny embeddings, and a Random Forest hybrid model.
- Retrieves supporting or refuting evidence from curated climate datasets.
- Records analyses in PostgreSQL and exposes history, metrics, experiments, and ablation results.
- Provides a modern dashboard built with Next.js for analysis, history, and model evaluation.

## Architecture

```text
Next.js dashboard
       │ HTTP
       ▼
FastAPI API ──► NER + ontology rules + evidence retrieval + hybrid classifier
       │
       └──► PostgreSQL (dataset records, evidence, prediction history)
```

## Technology

| Area | Technologies |
| --- | --- |
| API | Python, FastAPI, Uvicorn, Pydantic |
| ML / NLP | scikit-learn, spaCy, NLTK, Transformers, PyTorch, RDFLib |
| Data | PostgreSQL, SQLAlchemy, pandas, PyArrow |
| Frontend | Next.js App Router, React, TypeScript, Tailwind CSS, Recharts, Lucide |

## Requirements

- Python 3.12 or newer
- Node.js 20 or newer with npm
- PostgreSQL 15 or newer
- Local Hugging Face cache for `prajjwal1/bert-tiny` (the production artifact uses it with `local_files_only=True`)

## Quick start

### 1. Configure and start PostgreSQL

Create a database, then export a connection string. Keep it out of source control.

```bash
cp .env.example .env
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'
python init_db.py
```

### Database schema and reproducible training data

The repository includes a PostgreSQL snapshot intended to reproduce the data layer used by the project:

- [Schema](docs/schema.sql): tables, constraints, relationships, JSONB fields, and indexes.
- [Training and evidence data](docs/database_data.sql): current `dataset_records` and `dataset_evidence` contents.

Restore a clean database with:

```bash
createdb fakenews-clima
psql -d fakenews-clima -f docs/schema.sql
psql -d fakenews-clima -f docs/database_data.sql
```

The data dump intentionally excludes `prediction_history`, because it can contain texts submitted by users during local tests. To regenerate the training/evidence dump from your current database:

```bash
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'
python scripts/export_database_dump.py
```

Use `python scripts/export_database_dump.py --include-history` only after reviewing the history data before sharing it.

### 2. Start the API

Create and activate a virtual environment, then install the backend dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

The interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Check the service with:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Start the dashboard

In another terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The default API URL is `http://localhost:8000`; edit `frontend/.env.local` to change it.

## Main API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/predict` | Analyzes one claim and persists the result. |
| `POST` | `/batch_predict` | Analyzes between 1 and 100 texts. |
| `GET` | `/history` | Returns recent persisted analyses. |
| `GET` | `/metrics` | Returns the promoted model metrics. |
| `GET` | `/ablation` | Returns ablation-study results. |
| `GET` | `/experiments` | Returns registered experiment metadata. |
| `GET` | `/evaluation/holdout` | Returns the latest promoted-model holdout evaluation and its status. |
| `POST` | `/evaluation/holdout` | Starts a controlled holdout evaluation of the promoted model. |
| `GET` | `/health` | Service health check. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"CO2 does not contribute to global warming"}'
```

## Repository layout

```text
app/             FastAPI application, services, model configuration and metrics
frontend/        Next.js dashboard
scripts/         Ingestion, training, evaluation and verification commands
data/evaluation/ Versioned evaluation fixtures
docs/            Technical, final-project, and experiments documentation
```

Large downloaded datasets (`data/external/`, the enriched Climate-FEVER JSONL) and generated experiment pickle artifacts are intentionally ignored by Git. The production inference artifacts in `app/models/` are versioned; rerun the relevant scripts in `scripts/` to regenerate training inputs or experiments.

## Documentation

- [Technical report](docs/INFORME_TECNICO.md)
- [Final project report](docs/INFORME_FINAL_PROYECTO.md)
- [Experiments and algorithms](docs/EXPERIMENTOS_Y_ALGORITMOS.md)
- [External comparison, requirements, and phases](docs/COMPARACION_REQUISITOS_Y_FASES.md)
- [Database schema](docs/schema.sql) and [reproducible dataset/evidence dump](docs/database_data.sql)
- [Extended system documentation](documentacion.md)
- [Frontend guide](frontend/README.md)

## Quality checks

```bash
# Frontend
cd frontend && npx tsc --noEmit && npm run build

# Backend smoke check (requires DATABASE_URL and local model dependencies)
python -m uvicorn app.main:app --reload
```
