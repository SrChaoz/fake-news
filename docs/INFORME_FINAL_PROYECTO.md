# Informe final de proyecto: Climate Veritas

**Plataforma neuro-simbólica para detección asistida de desinformación ambiental**
**Estado:** prototipo académico funcional
**Fecha de consolidación:** 12 de agosto de 2026

## 1. Resumen ejecutivo

Climate Veritas es una plataforma de apoyo al *fact-checking* ambiental. Analiza una publicación, extrae las entidades de dominio, contrasta relaciones ambientales explícitas con reglas ontológicas, recupera evidencia curada cuando la coincidencia es fuerte y finalmente aplica un modelo supervisado híbrido para devolver una etiqueta binaria: `REAL` o `FAKE`.

El resultado principal validado del clasificador actualmente promovido (`holdout_bert`) es **82.50% de accuracy y 82.37% de macro-F1** sobre un *holdout* balanceado de 200 afirmaciones que fueron excluidas del entrenamiento. Para la clase más importante en este proyecto, `FAKE`, alcanzó **77.78% de precisión, 91.00% de recall y 83.87% de F1** en dicho benchmark.

Esta cifra no debe interpretarse como “82.5% de certeza para cualquier noticia de Internet” ni como precisión clínica/científica. Es una estimación puntual en el benchmark retenido disponible. Además, las reglas ontológicas y la recuperación de evidencia añadidas después son mecanismos selectivos de seguridad y trazabilidad; su ganancia global debe medirse de nuevo con un benchmark independiente de mayor tamaño.

## 2. Problema, alcance y criterio de éxito

El proyecto aborda afirmaciones ambientales breves, principalmente en inglés, sobre cambio climático, gases de efecto invernadero, biodiversidad, contaminación y energía. El objetivo es detectar patrones de desinformación y presentar una salida explicable, no sustituir una revisión científica o periodística.

| Elemento | Alcance actual |
|---|---|
| Etiquetas | `REAL` (respaldada por la fuente de datos) y `FAKE` (refutada por la fuente) |
| Idioma de datos de entrenamiento | Predominantemente inglés |
| Español | Aliases y reglas críticas soportan frases comunes; no hay aún benchmark español suficiente |
| Persistencia | PostgreSQL: datasets, evidencia y trazabilidad de predicciones |
| Interfaz | API FastAPI y frontend Next.js |
| Decisión de diseño | Priorizar *recall* de `FAKE` para dejar escapar menos desinformación, sin ocultar falsos positivos |

## 3. Arquitectura implementada

```mermaid
graph TD
    U[Usuario] --> FE[Frontend Next.js<br/>TypeScript + Tailwind]
    FE -->|POST /predict| API[FastAPI]
    API --> C[Normalización del claim]
    C --> NER[spaCy + diccionario ambiental]
    NER --> ONT[RDFLib: enlace y reglas ontológicas]
    ONT --> KG[(Conceptos SWEET / ENVO / GEMET / AGROVOC)]
    C --> EVI[Recuperación selectiva de evidencia<br/>TF-IDF claim-evidence]
    EVI --> EVDB[(dataset_records + dataset_evidence)]
    C --> FX[Feature engineering]
    NER --> FX
    ONT --> FX
    FX --> BERT[Embedding CLS<br/>prajjwal1/bert-tiny]
    BERT --> RF[Random Forest balanceado]
    RF --> DEC{Prioridad de decisión}
    ONT --> DEC
    EVI --> DEC
    DEC --> RES[Respuesta explicable REAL / FAKE]
    RES --> PH[(prediction_history)]
    PH --> API
    API --> FE
```

### Secuencia de inferencia

```mermaid
sequenceDiagram
    actor U as Usuario
    participant W as Frontend Next.js
    participant A as FastAPI
    participant N as NER / Ontología
    participant M as Modelo híbrido
    participant E as Evidencia local
    participant P as PostgreSQL

    U->>W: Introduce una publicación
    W->>A: POST /predict { text }
    A->>N: Claim, entidades, enlaces y reglas
    N-->>A: Entidades + conflicto/soporte ontológico
    A->>M: TF-IDF + señales + embedding
    M-->>A: P(REAL) y predicción estadística
    A->>E: Búsqueda de evidencia curada
    E-->>A: SUPPORTED / REFUTED / insuficiente
    A->>A: Conflicto > soporte ontológico/evidencia > modelo
    A->>P: Inserta PredictionHistory
    P-->>A: Confirmación
    A-->>W: Predicción, confianza, explicación y trazas
    W-->>U: Resultado visual
```

## 4. Flujo neuro-simbólico y lógica de decisión

1. **Normalización de afirmación.** Si no se entrega un `claim` explícito, el texto se limpia de espacios redundantes y se usa como afirmación de trabajo.
2. **NER ambiental.** spaCy más un diccionario de dominio identifica entidades como `CO2`, dióxido de carbono, `CH4`, metano, gas de efecto invernadero, calentamiento global, cambio climático y otros conceptos ambientales.
3. **Enlace ontológico.** RDFLib vincula aliases con un catálogo local inspirado en SWEET, ENVO, GEMET y AGROVOC. El resultado conserva URI, etiqueta y ontología de origen.
4. **Reglas de contradicción.** Negar una relación canónica, por ejemplo “CO2 has no impact on global warming” o “El metano no es un gas de efecto invernadero”, genera `ONTOLOGY_CONFLICT` y fuerza `FAKE`.
5. **Reglas de soporte.** Un conjunto restringido de relaciones canónicas positivas —CO2/metano como gases de efecto invernadero y su contribución al calentamiento— genera `ONTOLOGY_SUPPORT` si no existe negación. Esto corrigió el fallo observado con “Carbon dioxide is a greenhouse gas that contributes to global warming.”
6. **Recuperación de evidencia.** Se buscan claims curados mediante TF-IDF. Solo una coincidencia alta y separada del segundo resultado puede respaldar o refutar la salida (`SUPPORTED`/`REFUTED`). Coincidencias débiles se marcan como `INSUFFICIENT_EVIDENCE` y no se muestran como prueba.
7. **Clasificador estadístico.** Si no hay regla ni evidencia fuerte, se utiliza el modelo híbrido.

La prioridad es deliberada:

```text
conflicto ontológico → FAKE
soporte ontológico → REAL
evidencia fuerte que refuta → FAKE
evidencia fuerte que respalda → REAL
en otro caso → decisión del Random Forest con umbral calibrado
```

Una confianza de `1.0` en una regla significa que la decisión se tomó de forma determinista dentro de esa regla, **no** que se haya medido 100% de precisión sobre todas las afirmaciones del mundo.

## 5. Datos, calidad y prevención de fuga

| Fuente | Papel | Resultado incorporado |
|---|---|---|
| Climate-FEVER enriquecido | Base de entrenamiento | `SUPPORTS → REAL`, `REFUTES → FAKE`; ambiguos excluidos |
| ClimateCheck | Claims de consenso | Ejemplos REAL/FAKE y abstracts para evidencia |
| ClimaFactsKG | Mitos climáticos refutados | Ejemplos FAKE con corrección, URL y evidencia |
| `holdout_v1` | Benchmark retenido | 200 ejemplos balanceados; excluidos del entrenamiento de producción |

El conjunto usado para entrenar `holdout_bert` tiene **1,004 registros**: 569 `REAL` y 435 `FAKE`. Se excluyeron 247 casos ambiguos históricos y 199 claims normalizados del benchmark retenido para evitar evaluar textos que el modelo ya hubiese visto.

La base de evidencia contiene 451 documentos curados: 250 correcciones ClimaFactsKG y 201 abstracts ClimateCheck. La recuperación es una capa de verificación selectiva; no se debe presentar como un modelo de inferencia textual (NLI) completo.

## 6. Algoritmo en producción y representación

El artefacto activo es `app/models/random_forest_model.pkl`, promovido desde el experimento `holdout_bert`.

```text
texto + claim
  ├─ TF-IDF: unigramas y bigramas (5,000 términos máximo)
  ├─ señales: longitud de texto/claim y legibilidad
  ├─ señales simbólicas: linked_entity_count, has_conflict
  └─ embedding contextual CLS de BERT-tiny (128 dimensiones)
                         ↓
       RandomForestClassifier(class_weight="balanced")
                         ↓
          umbral de REAL calibrado: 0.56
```

### Por qué se eligió esta combinación

- **TF-IDF** captura expresiones discriminativas de palabras y frases.
- **BERT-tiny** incorpora contexto semántico de manera viable en CPU; se usa como extractor de embeddings, no como BERT grande afinado extremo a extremo.
- **Random Forest** combina variables densas, dispersas y simbólicas; `class_weight="balanced"` atenúa el desequilibrio de clases.
- **NER y ontología** aportan trazabilidad y corrigen casos de alto riesgo con negación explícita.
- **Umbral 0.56** se calibró con predicciones *out-of-fold* para equilibrar el comportamiento de clases; no es necesariamente el umbral óptimo para todos los contextos de despliegue.

### Decisiones de preprocesamiento

Se detectó que las stop words inglesas convencionales eliminaban `no`, `not`, `nor`, `never`, `neither` y `without`. En fact-checking estas palabras cambian por completo el significado, por lo cual el preprocesamiento actualizado las conserva. La tokenización y el NER no sustituyen el razonamiento factual: son señales para el modelo y para las reglas.

**Word2Vec no se promovió.** Sus vectores estáticos, especialmente al promediar palabras, tienden a perder estructura relacional y negación. En este problema, los embeddings contextuales y las reglas explícitas son más adecuados. Word2Vec queda como posible baseline académica, no como el reemplazo recomendado de la arquitectura actual.

## 7. Experimentos ejecutados

Las métricas de validación cruzada siguientes no son estrictamente comparables si cambian datos o protocolos; se muestran para documentar la evolución.

| Experimento | Datos | Arquitectura | Accuracy CV | Macro-F1 CV | Decisión |
|---|---:|---|---:|---:|---|
| `bert_tiny` | 753 | TF-IDF + señales + BERT-tiny + RF | 0.6932 | 0.6702 | Base inicial |
| `climatecheck_bert` | 954 | Híbrido BERT-tiny + RF | 0.6247 | 0.6043 | No promovido |
| `climafacts_tfidf` | 1,204 | TF-IDF + RF sin Transformer | 0.7284 | 0.7166 | Comparador rápido |
| `climafacts_bert` | 1,204 | Híbrido BERT-tiny + RF, 3 folds | 0.7442 | 0.7355 | Exploratorio |
| `climafacts_bert_5fold` | 1,204 | Híbrido BERT-tiny + RF | 0.7392 | 0.7297 | Producción previa |
| `holdout_bert` | 1,004 | Híbrido con holdout excluido | 0.7221 | 0.7141 | **Promovido** |
| `negation_aware_bert` | 1,204 | Híbrido que conserva negaciones | 0.7417 | 0.7352 | Requiere holdout sin fuga |
| `negation_word_char_holdout` | 1,004 | TF-IDF palabra+carácter + LR | 0.7132 | 0.7047 | No promovido |

### Ablación del artefacto de producción

Resultados del split 80/20 del entrenamiento de producción. Las métricas principales de esta tabla se refieren a `REAL`; se añade macro-F1 para observar ambas clases.

| Variante | Accuracy | Balanced accuracy | F1 REAL | Macro-F1 | F1 FAKE |
|---|---:|---:|---:|---:|---:|
| Híbrido completo | 0.6816 | 0.6621 | 0.7419 | 0.6632 | 0.5844 |
| Sin ontología | 0.7164 | 0.6792 | 0.7927 | 0.6720 | 0.5512 |
| Sin claim extraction | 0.7114 | 0.6789 | 0.7836 | 0.6754 | 0.5672 |
| Sin ambos | 0.6866 | 0.6543 | 0.7640 | 0.6487 | 0.5333 |

La ablación no demuestra una ganancia estadística global de las señales ontológicas en ese split. Su justificación actual es operacional: permiten interceptar negaciones peligrosas y explicar claramente el resultado. Por ello su cobertura y sus falsos positivos deben ser auditados de forma continua.

### Comparación de modelos de texto

| Variante | Accuracy | Macro-F1 | Precisión FAKE | Recall FAKE | F1 FAKE |
|---|---:|---:|---:|---:|---:|
| TF-IDF legado + Logistic Regression | 0.7051 | 0.7022 | 0.6586 | 0.6935 | 0.6745 |
| TF-IDF conservando negación + LR | 0.7242 | 0.7211 | 0.6839 | 0.7047 | 0.6930 |
| TF-IDF palabra + carácter + LR | **0.7442** | **0.7404** | **0.7125** | **0.7103** | **0.7102** |

Preservar negaciones aportó +0.0189 de macro-F1 frente a la variante léxica original. Agregar n-gramas de caracteres aportó +0.0193 adicional. El candidato final de texto no superó al híbrido elegido en el benchmark retenido, pero es un baseline sólido para futuros experimentos.

## 8. Métricas vigentes y cómo interpretar la “precisión”

### Validación cruzada interna del modelo promovido

| Métrica | Resultado (media ± desviación estándar, 5 folds) |
|---|---:|
| Accuracy | 0.7221 ± 0.0145 |
| Balanced accuracy | 0.7126 ± 0.0142 |
| Macro-F1 | 0.7141 ± 0.0144 |
| F1 REAL | 0.7616 ± 0.0142 |

### Resultado principal: benchmark retenido `holdout_v1`

| Métrica | Resultado |
|---|---:|
| Tamaño | 200 claims, balanceados: 100 FAKE y 100 REAL |
| Accuracy | **0.8250** |
| Balanced accuracy | **0.8250** |
| Macro-F1 | **0.8237** |
| Precisión FAKE | 0.7778 |
| Recall FAKE | **0.9100** |
| F1 FAKE | **0.8387** |

Matriz de confusión del holdout, con filas reales y columnas predichas:

```text
                  Predicho FAKE    Predicho REAL
Verdadero FAKE           91                 9
Verdadero REAL           26                74
```

Por tanto, si “precisión” significa **accuracy global**, la mejor estimación actual es **82.5%** en el benchmark retenido. Si significa **precision de la etiqueta FAKE**, es **77.78%**: de cada 100 mensajes marcados como FAKE en ese benchmark, aproximadamente 78 eran realmente FAKE. El *recall* de FAKE de 91% significa que el sistema detectó 91 de cada 100 falsedades del benchmark, a costa de 26 falsos positivos sobre 100 afirmaciones reales.

El archivo `external_evaluation.json` contiene además un smoke test de ocho ejemplos de ClimateCheck; su accuracy es 0.75, pero la propia herramienta advierte que una muestra menor de 50 no estima generalización y no debe usarse como métrica principal.

## 9. API, persistencia, frontend y pruebas

### Endpoints principales

| Método | Ruta | Responsabilidad |
|---|---|---|
| `POST` | `/predict` | Predice y persiste un análisis |
| `POST` | `/batch_predict` | Procesa hasta 100 textos en una transacción |
| `GET` | `/history` | Consulta historial persistido |
| `GET` | `/metrics` | Expone métricas del artefacto activo |
| `GET` | `/experiments` | Lista comparaciones y experimentos |
| `GET` | `/ablation` | Devuelve el estudio de ablación |
| `GET` | `/health` | Comprueba disponibilidad |

La respuesta de predicción incluye la etiqueta, `confidence_score`, probabilidad estadística, umbral, fuente de decisión, entidades, conflictos, explicación, `verification_status` y evidencia fuerte si existe. Los estados son `SUPPORTED`, `REFUTED`, `INSUFFICIENT_EVIDENCE`, `ONTOLOGY_CONFLICT` y `ONTOLOGY_SUPPORT`.

El frontend profesional está en `frontend/` e implementa tres vistas: analizador, historial y experimentos. Usa Next.js App Router, TypeScript, Tailwind, Lucide y Recharts. CORS admite orígenes locales `localhost`/`127.0.0.1` para su comunicación con FastAPI.

La prueba E2E `scripts/verify_e2e.py` valida disponibilidad, `/predict`, persistencia en `/history`, `/metrics`, `/experiments` y `/ablation`. La última ejecución documentada aprobó **6/6** comprobaciones.

## 10. Reproducción local

```bash
# Dependencias Python y URL de PostgreSQL
python -m pip install -r requirements.txt
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'

# Crear tablas e iniciar backend
python init_db.py
python -m uvicorn app.main:app --reload
```

En otra terminal:

```bash
cd frontend
npm install
printf 'NEXT_PUBLIC_API_BASE_URL=http://localhost:8000\n' > .env.local
npm run dev
```

La aplicación queda disponible en `http://localhost:3000` y la documentación interactiva de FastAPI en `http://localhost:8000/docs`.

## 11. Limitaciones, riesgos y mejora prioritaria

1. **Generalización limitada.** El benchmark retenido tiene 200 casos y dos fuentes conocidas. Hace falta un tercer conjunto externo, estratificado por categoría, fuente e idioma.
2. **Cobertura lingüística.** El modelo se entrenó principalmente en inglés. Las reglas en español sirven para relaciones canónicas, no para certificar comprensión robusta de noticias complejas en español. Se necesita un benchmark español antes de activar traducción automática como decisión central.
3. **Clasificación no es verificación.** TF-IDF, BERT-tiny y Random Forest detectan regularidades, pero no prueban factualidad. La evolución más valiosa es recuperación de fuentes primarias + modelo NLI/entailment que evalúe claim-evidence.
4. **Reglas deben gobernarse.** Cada regla ontológica requiere fuente, pruebas de regresión, ámbito limitado y auditoría de falsos positivos.
5. **Calibración.** La probabilidad del Random Forest no debe exponerse como certeza factual. Se recomienda calibración probabilística (isotónica o Platt) y un estado “requiere revisión” para casos de baja confianza.

### Siguiente experimento recomendado

Reentrenar `negation_aware_bert` excluyendo `holdout_v1`, evaluarlo contra el mismo benchmark y promocionarlo solo si supera de manera reproducible macro-F1 0.8237 y recall FAKE 0.91, o si existe un trade-off explícitamente aprobado. Después, crear un benchmark externo español/inglés y comparar un enfoque *retrieval + NLI* con el actual.

## 12. Conclusión

Climate Veritas alcanzó una arquitectura completa y demostrable: ingesta y trazabilidad en PostgreSQL, pipeline NLP/ontológico, modelo híbrido, evidencia selectiva, API, interfaz web y pruebas E2E. El mejor resultado validado es **82.5% de accuracy y 82.37% de macro-F1** en un holdout que se mantuvo fuera del entrenamiento. Las reglas ontológicas cierran de forma explicable los errores críticos de negación y ahora también respaldan relaciones ambientales positivas estrictamente definidas.

La plataforma está lista para una presentación académica como sistema de apoyo explicable. Para una afirmación de exactitud cercana a producción real se requieren más evidencia primaria, evaluación externa grande y multilingüe, calibración y revisión humana de los casos de baja confianza.
