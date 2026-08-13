# Informe técnico: plataforma neuro-simbólica de detección de desinformación ambiental

## 1. Resumen ejecutivo

Climate Veritas es una plataforma académica para analizar afirmaciones ambientales en inglés y clasificarlas como `REAL` o `FAKE`. Combina procesamiento de lenguaje natural, reglas simbólicas de dominio y aprendizaje automático. La salida no es un veredicto científico definitivo: es una señal explicable de apoyo a la revisión humana.

El sistema integra una API REST en FastAPI, PostgreSQL para trazabilidad, un frontend Next.js y un pipeline híbrido con TF-IDF, señales lingüísticas, entidades ambientales, conflictos ontológicos, embeddings Transformer y Random Forest. Una contradicción explícita de una relación ambiental conocida tiene prioridad como regla de seguridad y fuerza la salida `FAKE`.

### Alcance y estado

- Clasificación binaria: `REAL`/`FAKE`.
- Datos de entrenamiento del experimento de producción: 1,004 registros (569 REAL, 435 FAKE); los 247 casos Climate-FEVER ambiguos y 199 claims normalizados del benchmark retenido se excluyeron.
- Artefacto de producción: experimento `holdout_bert`, seleccionado por evaluación externa retenida.
- Encoder activo: `prajjwal1/bert-tiny`, CPU, embedding CLS de 128 dimensiones.
- Umbral calibrado para REAL: `0.56`.

## 2. Arquitectura del sistema

```mermaid
graph TD
    U[Usuario] --> FE[Frontend Next.js]
    FE -->|HTTP JSON / CORS| API[API REST FastAPI]
    API --> CE[Claim extraction / normalización]
    CE --> NER[spaCy + diccionario Environmental NER]
    NER --> ONT[Entity linking y reglas ontológicas]
    ONT --> KG[(SWEET / ENVO / GEMET / AGROVOC<br/>grafo RDF y conceptos simulados)]
    NER --> FX[Feature Extractor]
    ONT --> FX
    FX --> ML[Clasificador híbrido<br/>TF-IDF + señales + BERT + Random Forest]
    ML --> RULE{¿Conflicto ontológico?}
    RULE -->|Sí| FAKE[Resultado FAKE por regla]
    RULE -->|No| PRED[Resultado del clasificador]
    FAKE --> DB[(PostgreSQL)]
    PRED --> DB
    DB --> HIST[prediction_history]
    DB --> DATA[dataset_records / dataset_evidence]
    API --> FE
```

### Secuencia de `POST /predict`

```mermaid
sequenceDiagram
    actor Usuario
    participant Web as Next.js
    participant API as FastAPI
    participant NLP as NER / Claim
    participant Ont as Ontología RDF
    participant ML as Modelo híbrido
    participant DB as PostgreSQL

    Usuario->>Web: Envía publicación
    Web->>API: POST /predict {text}
    API->>NLP: Normaliza claim y extrae entidades
    NLP-->>API: Claim + entidades ambientales
    API->>Ont: Enlaza conceptos y detecta conflictos
    Ont-->>API: Entidades enlazadas + conflictos
    API->>ML: TF-IDF + métricas + señales + embedding
    ML-->>API: Probabilidad REAL
    alt Conflicto ontológico explícito
        API->>API: Anula resultado estadístico a FAKE
    end
    API->>DB: Inserta PredictionHistory
    DB-->>API: Commit
    API-->>Web: Predicción explicable JSON
    Web-->>Usuario: Badge, confianza y evidencia técnica
```

## 3. Detalles del pipeline neuro-simbólico

### Claim extraction y Environmental NER

La API acepta `text` y opcionalmente un `claim`. Cuando no se aporta claim, aplica una normalización ligera: elimina espacios redundantes y utiliza el texto como afirmación inicial. No se afirma que exista una abstracción semántica generativa independiente; esa mejora queda como trabajo futuro.

`app/services/ner_engine.py` combina spaCy (`en_core_web_sm` o alternativa disponible) con un diccionario específico del dominio. Reconoce gases de efecto invernadero como `CO2` y `CH4`, procesos climáticos, deforestación, biodiversidad, contaminación, energía y otros términos ambientales. Cada entidad incluye texto, forma normalizada, etiqueta, fuente y offsets cuando están disponibles.

### Entity linking y detección de conflictos

`app/services/ontology_engine.py` utiliza RDFLib con conceptos y mapeos iniciales asociados a SWEET, ENVO, GEMET y AGROVOC. El enlace devuelve URI conceptual, etiqueta y ontología asociada.

Las reglas detectan contradicciones explícitas, por ejemplo negar que CO2 o CH4 sean gases de efecto invernadero o negar su contribución al calentamiento global. También hay reglas para relaciones conocidas sobre deforestación, biodiversidad y contaminación. Si `has_conflict=true`, FastAPI responde `FAKE` con `decision_source=ontology_rule_override`, aunque la probabilidad del modelo estadístico sea REAL. Esto prioriza seguridad y explicabilidad, pero las reglas deben revisarse ante falsos positivos.

## 4. Modelos de machine learning y feature engineering

### Representación de características

El vector final de la variante híbrida concatena:

| Grupo | Implementación | Propósito |
|---|---|---|
| Texto | TF-IDF de unigramas y bigramas, máximo 5,000 términos | Captura patrones léxicos y frases relevantes. |
| Longitud/legibilidad | Conteo de palabras de texto y claim; legibilidad | Señales estilísticas complementarias. |
| Simbólico | `linked_entity_count`, `has_conflict` | Representa presencia de conocimiento ambiental y contradicciones. |
| Transformer | Embedding CLS | Codifica contexto semántico del texto y claim. |

TF-IDF se ajusta exclusivamente sobre los índices de entrenamiento en cada split/fold para evitar que el vocabulario o IDF del conjunto de prueba filtren información.

### Transformer y clasificador híbrido

El código puede seleccionar `bert-base-uncased` en un entorno con GPU o un encoder ligero en CPU. El artefacto actualmente promovido usa **`prajjwal1/bert-tiny`**, no `bert-base-uncased`, por razones de coste y tiempo. Sus embeddings se concatenan con las demás características.

El clasificador final es `RandomForestClassifier` con `class_weight="balanced"`. Existe además una línea base de embeddings Transformer con regresión logística. El entrenamiento calibra el umbral de REAL mediante predicciones out-of-fold del train y busca conservar un recall de FAKE objetivo de al menos 0.60 cuando los datos lo permiten.

## 5. Métricas de evaluación y experimentos

Los valores siguientes proceden de `app/models/metrics.json` del artefacto de producción. Precision, recall y F1 principales se calculan sobre la clase REAL; por ello se reportan adicionalmente métricas por clase y macro-F1.

### Modelos disponibles

| Sistema | Accuracy | Precision | Recall | F1 | Estado |
|---|---:|---:|---:|---:|---|
| Línea base Transformer + Logistic Regression | 0.6766 | 0.7094 | 0.7281 | 0.7186 | Entrenado; embeddings BERT-tiny |
| Modelo híbrido Random Forest | 0.6816 | 0.6866 | 0.8070 | 0.7419 | Producción |
| RoBERTa | N/D | N/D | N/D | N/D | No entrenado; no se inventan métricas |
| Engine ontológico aislado | N/D | N/D | N/D | N/D | Reglas de seguridad, no clasificador independiente evaluado |

La validación cruzada del híbrido de producción (5 folds) reporta accuracy `0.7221 ± 0.0145`, balanced accuracy `0.7126 ± 0.0142`, F1 REAL `0.7616 ± 0.0142` y macro-F1 `0.7141 ± 0.0144`.

### Estudio de ablación

| Variante | Accuracy | Balanced accuracy | F1 REAL | Macro-F1 | F1 FAKE |
|---|---:|---:|---:|---:|---:|
| Sistema completo | 0.6816 | 0.6621 | 0.7419 | 0.6632 | 0.5844 |
| Sin ontologías | 0.7164 | 0.6792 | 0.7927 | 0.6720 | 0.5512 |
| Sin claim extraction | 0.7114 | 0.6789 | 0.7836 | 0.6754 | 0.5672 |
| Sin ambos componentes | 0.6866 | 0.6543 | 0.7640 | 0.6487 | 0.5333 |

El split de ablación favorece variantes sin señales simbólicas. Esto no demuestra que las ontologías no aporten valor operacional: las reglas aún interceptan contradicciones explícitas y aumentan explicabilidad. Se requiere una evaluación externa mayor y una auditoría de errores antes de atribuir causalidad.

### Matriz de confusión del sistema completo

La matriz se ordena como `[FAKE, REAL]` en filas (verdadero) y columnas (predicho):

```text
                 Predicho FAKE   Predicho REAL
Verdadero FAKE          45              42
Verdadero REAL          22              92
```

El modelo identificó 45 de 87 ejemplos FAKE del split y clasificó correctamente 92 de 114 REAL. El recall de FAKE en este split es `0.5172`; por ello no debe presentarse el sistema como detector definitivo y se priorizan más datos FAKE verificados y evaluación externa.

### Benchmark retenido

`data/evaluation/holdout_v1.jsonl` contiene 200 casos balanceados (100 REAL ClimateCheck y 100 FAKE ClimaFactsKG). Solo es válido para modelos entrenados con `--exclude-claims-file data/evaluation/holdout_v1.jsonl`. `holdout_bert` obtuvo macro-F1 `0.8237` y recall FAKE `0.9100` en este benchmark; fue promovido porque ese benchmark contiene afirmaciones que el modelo no vio durante el entrenamiento. Su validación cruzada interna sigue documentada y debe monitorizarse junto con futuros benchmarks externos.

## 6. API, persistencia y pruebas E2E

### Endpoints

| Método | Ruta | Función |
|---|---|---|
| `POST` | `/predict` | Ejecuta el pipeline y guarda una fila en `prediction_history`. |
| `POST` | `/batch_predict` | Procesa de 1 a 100 textos en una sola transacción. |
| `GET` | `/history?limit=50` | Obtiene análisis recientes desde PostgreSQL. |
| `GET` | `/metrics` | Métricas del modelo híbrido y validación cruzada. |
| `GET` | `/experiments` | Estado de BERT, RoBERTa, ontologías y experimentos guardados. |
| `GET` | `/ablation` | Resultados de variantes de ablación. |
| `GET` | `/health` | Estado básico de la API. |

FastAPI permite CORS para `localhost` y `127.0.0.1` en cualquier puerto de desarrollo. Esto permite utilizar Next.js en 3000 o 3001 sin rechazar el preflight `OPTIONS`.

### Verificación E2E

El script `scripts/verify_e2e.py` realiza un flujo completo contra una API ya iniciada:

1. Comprueba `GET /health`.
2. Envía una afirmación con un marcador único a `POST /predict`.
3. Valida campos obligatorios, etiqueta binaria y rango `[0, 1]` de confianza.
4. Consulta `/history` y confirma que el marcador quedó persistido.
5. Comprueba contratos de `/metrics`, `/experiments` y `/ablation`.

Ejecutar:

```bash
python scripts/verify_e2e.py
# URL alternativa:
python scripts/verify_e2e.py --base-url http://localhost:8000 --timeout 90
```

El script muestra una línea `PASS` o `FAIL` por verificación y usa un código de salida distinto de cero si existe cualquier fallo; por ello es apto para CI básico. Su ejecución crea una predicción de prueba deliberada en el historial, identificable por el prefijo `[E2E-...]`.

### Resultado de la ejecución E2E

Ejecutado localmente el 2026-07-28 contra `http://localhost:8000`:

```text
[PASS] API disponible
[PASS] POST /predict
[PASS] Persistencia en /history
[PASS] GET /metrics
[PASS] GET /experiments
[PASS] GET /ablation
Resultado: 6/6 verificaciones aprobadas.
```

## 7. Manual de instalación y ejecución local

### Prerrequisitos

- Python 3.12 o superior (el proyecto también se probó con Python 3.14).
- PostgreSQL en ejecución.
- Node.js 20 o superior y npm para el frontend.
- Modelo spaCy inglés disponible; el motor cuenta con mecanismos de fallback.

### Base de datos y backend

Inicia PostgreSQL según tu distribución. En sistemas con `systemd`, el nombre de servicio puede ser `postgresql` o una versión como `postgresql-16`:

```bash
sudo systemctl start postgresql
```

En la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'
python init_db.py
```

Para una instalación nueva, ingiere primero los datos disponibles:

```bash
python scripts/ingest_dataset.py
python scripts/ingest_climatecheck.py --dataset data/external/climatecheck-train.parquet
python scripts/ingest_climafacts.py --dataset data/external/climafacts-kg/data/climafacts_kg.ttl
```

Los comandos de ingesta son deduplicados; revisa sus opciones `--dry-run` antes de operar sobre una base compartida. Después, entrena o usa los artefactos ya generados:

```bash
python scripts/train_models.py \
  --experiment-name holdout_bert \
  --transformer-model prajjwal1/bert-tiny \
  --estimators 100 \
  --cv-folds 5 \
  --exclude-claims-file data/evaluation/holdout_v1.jsonl
```

Para promover un experimento validado:

```bash
python scripts/promote_experiment.py holdout_bert
```

Levanta FastAPI en una terminal separada:

```bash
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'
python -m uvicorn app.main:app --reload
```

La documentación OpenAPI queda disponible en `http://127.0.0.1:8000/docs`.

### Frontend Next.js

En otra terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Abre `http://localhost:3000`. La variable `NEXT_PUBLIC_API_BASE_URL` debe apuntar a `http://localhost:8000` para desarrollo local. Para producción, definir explícitamente una URL HTTPS adecuada y restringir CORS a los dominios reales.

### Ejecución de pruebas

Con FastAPI levantado y `DATABASE_URL` configurada:

```bash
python scripts/verify_e2e.py
cd frontend && npm run build
```

## 8. Limitaciones y próximos pasos

1. Aumentar el benchmark externo con una tercera fuente independiente y más ejemplos FAKE/REAL.
2. Auditar falsos positivos y falsos negativos por categoría, fuente, idioma y regla ontológica.
3. Evaluar RoBERTa solo cuando se entrene y se registre un artefacto comparable; no sustituir valores ausentes por ceros.
4. Mejorar la extracción de claims con un componente dedicado y evaluación propia.
5. Añadir autenticación, rate limiting, observabilidad y políticas de retención antes de despliegue público.
6. No registrar textos sensibles sin consentimiento; `prediction_history` conserva la entrada y debe tener una política de privacidad explícita.

## 9. Capa de evidencia selectiva

Como mejora posterior al clasificador, se incorporó `app/services/evidence_engine.py`. El índice local contiene evidencia curada, no publicaciones de usuarios:

- 250 correcciones asociadas a ClimaFactsKG.
- 201 abstracts científicos asociados a ClimateCheck.

La recuperación usa TF-IDF de unigramas/bigramas sobre claims curados. Un resultado solo activa `evidence_retrieval_override` si la similitud del primer claim supera `0.56` y se separa al menos `0.08` del segundo resultado. Esto reduce el riesgo de que una fuente tangencial se presente como verificación.

La respuesta de `POST /predict` incorpora:

| Campo | Valores | Significado |
|---|---|---|
| `verification_status` | `SUPPORTED`, `REFUTED`, `INSUFFICIENT_EVIDENCE`, `ONTOLOGY_CONFLICT`, `ONTOLOGY_SUPPORT` | Estado de verificación basado en evidencia o reglas. `ONTOLOGY_SUPPORT` señala una relación canónica respaldada de forma determinista; no equivale a una probabilidad estadística general. |
| `evidence` | Lista de fragmentos, fuente, URL y similitud | Solo se muestra para una coincidencia fuerte. |

Si no existe evidencia fuerte, la API conserva el resultado del clasificador como señal estadística, pero declara `INSUFFICIENT_EVIDENCE`. Esta capa aumenta la precisión **selectiva** y la trazabilidad; no debe interpretarse como una métrica global nueva hasta evaluarla con un corpus de evidencia separado del benchmark.

Para cargar abstracts de ClimateCheck en una instalación nueva:

```bash
python scripts/attach_climatecheck_evidence.py --dry-run
python scripts/attach_climatecheck_evidence.py
python scripts/test_evidence_engine.py
```
