# Detector de desinformación ambiental

## Propósito

Proyecto académico de clasificación binaria de afirmaciones ambientales:

- `REAL`: la afirmación está respaldada por la fuente de verificación.
- `FAKE`: la afirmación es refutada por la fuente de verificación.

No sustituye revisión científica, periodística ni asesoramiento profesional. La respuesta de la API es una señal de apoyo y expone tanto la probabilidad del modelo como las reglas ontológicas que participaron.

## Arquitectura

```text
Cliente HTTP
   -> FastAPI (`POST /predict`)
      -> NER ambiental (spaCy + diccionario de dominio)
      -> coincidencia/reglas ontológicas (RDFLib)
      -> TF-IDF + métricas textuales + embedding BERT-tiny
      -> Random Forest híbrido
      -> regla de seguridad: contradicción ontológica explícita => FAKE
   -> respuesta explicable

PostgreSQL
   -> dataset_records
   -> dataset_evidence (correcciones y URL de procedencia)
   -> prediction_history
```

La API carga BERT desde la caché local; no necesita conexión a Hugging Face durante inferencia.

## Instalación y base de datos

Requiere Python 3.12+, PostgreSQL y las dependencias declaradas en `requirements.txt`.

```bash
python -m pip install -r requirements.txt
cp .env.example .env
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'
python init_db.py
```

Para iniciar la API:

```bash
python -m uvicorn app.main:app --reload
```

## API REST

La documentación interactiva está disponible en `http://127.0.0.1:8000/docs`.
La API permite CORS únicamente para los orígenes locales de desarrollo habituales
(`localhost`/`127.0.0.1` en puertos 3000, 5173 y 8080).

| Método y ruta | Propósito |
|---|---|
| `POST /predict` | Analiza un texto, aplica NER/ontología/modelo y guarda el resultado. |
| `POST /batch_predict` | Analiza entre 1 y 100 textos y persiste el lote en una transacción. |
| `GET /history?limit=20` | Devuelve los análisis más recientes de PostgreSQL; máximo 100. |
| `GET /metrics` | Métricas principales, matriz de confusión y validación cruzada. |
| `GET /ablation` | Comparación del modelo completo con sus ablaciones. |
| `GET /experiments` | Artefactos BERT, sistema ontológico y experimentos almacenados. |
| `GET /health` | Estado de disponibilidad de la API. |
| `GET /evaluation/holdout` | Estado y último informe del benchmark retenido del modelo promovido. |
| `POST /evaluation/holdout` | Inicia una reevaluación asíncrona y limitada al modelo de producción. |

Ejemplo de lote:

```bash
curl -X POST http://127.0.0.1:8000/batch_predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Methane is a greenhouse gas", "CO2 does not contribute to global warming"]}'
```

Cada resultado de predicción contiene `extracted_claim`, categoría,
entidades detectadas, conflictos ontológicos, etiqueta, confianza y explicación.
Los análisis se guardan en `prediction_history`; el endpoint de benchmark no los
persiste para evitar contaminar el historial operativo.

## Frontend Next.js

La interfaz principal vive en `frontend/` y utiliza Next.js App Router,
TypeScript, Tailwind CSS, Lucide y Recharts. Con FastAPI iniciado en otra
terminal, ejecuta:

```bash
cd frontend
npm install
npm run dev
```

Después abre `http://localhost:3000`. La URL de FastAPI se configura en
`frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

La anterior interfaz Streamlit permanece en `app/frontend/` como cliente
heredado, pero no es la interfaz recomendada para la aplicación.

Comprobación rápida:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"CO2 does not contribute to global warming"}'
```

La respuesta esperada es `FAKE` mediante `ontology_rule_override`, aunque el clasificador estadístico discrepe.

## Datos y procedencia

| Fuente | Uso | Etiquetas incorporadas |
|---|---|---|
| Climate-FEVER enriquecido | entrenamiento base | SUPPORTS → REAL, REFUTES → FAKE; los casos ambiguos se excluyen por defecto |
| ClimateCheck | afirmaciones de consenso estricto | Supports → REAL, Refutes → FAKE |
| ClimaFactsKG | mitos climáticos refutados en inglés | False → FAKE, con texto de corrección y URL en `dataset_evidence` |

Estado actual de la base de datos:

- Climate-FEVER: 500 REAL y 500 FAKE. De los FAKE, 247 proceden de casos ambiguos históricos y no participan en el entrenamiento por defecto.
- ClimateCheck de consenso: 169 REAL y 32 FAKE.
- ClimaFactsKG: 250 FAKE y 250 evidencias enlazadas.
- Conjunto de entrenamiento limpio antes del holdout: 669 REAL y 535 FAKE.

Las fuentes deben mantenerse únicamente para fines académicos, respetando sus licencias y atribución original. No se deben presentar etiquetas de datasets como veredictos nuevos producidos por este proyecto.

## Modelo en producción

El artefacto promovido es `holdout_bert`:

- Encoder: `prajjwal1/bert-tiny` (embedding CLS de 128 dimensiones).
- Clasificador: Random Forest con TF-IDF de unigramas/bigramas, métricas textuales, señales ontológicas y embedding.
- Umbral de decisión REAL: `0.56`, calibrado con datos de entrenamiento.
- Se excluyeron 199 afirmaciones normalizadas del benchmark retenido antes de entrenar.
- Reglas ontológicas de contradicción explícita tienen prioridad sobre el modelo estadístico.

### Resultados de validación cruzada

| Métrica | Validación cruzada de producción |
|---|---:|
| Accuracy | 0.7221 ± 0.0145 |
| Balanced accuracy | 0.7126 ± 0.0142 |
| Macro-F1 | 0.7141 ± 0.0144 |
| F1 REAL | 0.7616 ± 0.0142 |

El estudio de ablación muestra que algunas señales simbólicas no siempre mejoran la métrica agregada. Se conservan porque hacen explícitas contradicciones de alto riesgo y proporcionan trazabilidad; esta decisión debe reevaluarse con el benchmark retenido.

### Verificación basada en evidencia

La API incorpora una capa selectiva de recuperación de evidencia curada. Indexa 250 correcciones de ClimaFactsKG y 201 abstracts científicos de ClimateCheck. Solo anula el clasificador si la similitud TF-IDF entre la afirmación y un claim curado es alta y está claramente separada del segundo resultado. La respuesta incluye:

- `verification_status`: `SUPPORTED`, `REFUTED`, `INSUFFICIENT_EVIDENCE`, `ONTOLOGY_CONFLICT` u `ONTOLOGY_SUPPORT`.
- `evidence`: fragmentos, fuente, URL y similitud de los documentos recuperados.

Además de detectar contradicciones, las reglas ontológicas pueden respaldar un conjunto deliberadamente reducido de relaciones científicas canónicas (por ejemplo, CO2/metano como gases de efecto invernadero y su contribución al calentamiento). Estas reglas solo se aplican si no hay negación; las contradicciones conservan prioridad. El estado `ONTOLOGY_SUPPORT` identifica este caso. Su confianza expresa la certeza de una regla determinista, no una probabilidad estadística generalizable a cualquier noticia ambiental.

Una coincidencia débil **no** se presenta como prueba: mantiene el resultado del modelo y se marca `INSUFFICIENT_EVIDENCE`. Esta es una mejora de trazabilidad y precisión selectiva; no equivale todavía a un modelo NLI entrenado sobre pares claim-evidence.

## Benchmark retenido y protocolo de evaluación

`data/evaluation/holdout_v1.jsonl` contiene 200 casos balanceados y deterministas:

- 100 REAL de ClimateCheck.
- 100 FAKE de ClimaFactsKG.

No es válido evaluar con este archivo un modelo que haya visto esas afirmaciones. El próximo experimento debe excluirlo explícitamente:

```bash
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'

python scripts/train_models.py \
  --experiment-name holdout_bert \
  --transformer-model prajjwal1/bert-tiny \
  --estimators 100 \
  --cv-folds 5 \
  --exclude-claims-file data/evaluation/holdout_v1.jsonl
```

Luego se evalúa el experimento **sin promocionarlo** contra ese benchmark:

```bash
python scripts/evaluate_holdout.py \
  --model-directory app/models/experiments/holdout_bert \
  --output app/models/experiments/holdout_bert/holdout_evaluation.json
```

El resultado incluye matriz de confusión, cada error y conteos de errores por fuente/categoría. `holdout_bert` obtuvo macro-F1 `0.8237` y recall de FAKE `0.9100` en este benchmark nunca visto; por esa evidencia externa fue promovido. Las métricas internas deben seguirse monitorizando para evitar regresiones:

```bash
python scripts/promote_experiment.py holdout_bert
```

También existe `scripts/evaluate_external.py`, que evalúa el test oficial de ClimateCheck. Actualmente contiene apenas ocho casos de consenso binario, por lo que funciona como smoke test y no como estimación fiable de generalización.

## Análisis de errores

Cada entrenamiento exporta:

- `app/models/error_analysis.json`: falsos positivos/negativos del split, entidades y conflictos ontológicos.
- `app/models/error_summary.json`: conteos agregados.
- `app/models/metrics.json`: métricas, estudio de ablación, calibración y validación cruzada.

Al revisar errores, priorizar:

1. Etiquetas ambiguas o evidencia insuficiente.
2. Negaciones, condicionales y comparación de periodos temporales.
3. Entidades ambientales no reconocidas.
4. Reglas ontológicas demasiado amplias que produzcan anulaciones incorrectas.
5. Diferencias de distribución entre fuentes.

## Limitaciones y trabajo pendiente

- El tamaño de datos sigue siendo moderado y hay desequilibrios por fuente/categoría.
- El benchmark retenido utiliza fuentes conocidas, aunque los textos se excluyen del entrenamiento; se requiere además una fuente completamente independiente y suficientemente grande.
- Las reglas ontológicas son un conjunto inicial de conocimiento, no una ontología ambiental completa ni una prueba científica formal.
- BERT-tiny prioriza coste y velocidad sobre máxima exactitud.
- La confianza del clasificador no equivale a certeza factual; se debe mostrar evidencia y enlaces de procedencia en interfaces de usuario.

## Operación segura

No subir `.env`, contraseñas ni URLs con credenciales al repositorio. Antes de reemplazar producción, conservar el experimento en `app/models/experiments/`, comparar métricas y ejecutar pruebas de regresión de reglas ontológicas.
