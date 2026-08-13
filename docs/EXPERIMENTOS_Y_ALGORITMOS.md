# Registro técnico de experimentos y algoritmos

## 1. Propósito y criterio de lectura

Este documento registra los experimentos ejecutados para el clasificador de desinformación ambiental, los algoritmos evaluados, los cambios de datos y las decisiones de promoción. Su objetivo es evitar comparar métricas incompatibles o presentar resultados de validación interna como evidencia de generalización.

Las etiquetas son binarias: `FAKE=0` y `REAL=1`. Las métricas `precision`, `recall` y `f1_score` emitidas por `train_models.py` se refieren por defecto a **REAL**; por eso este informe prioriza también `macro_f1`, `balanced_accuracy` y las métricas específicas de FAKE.

## 2. Datos y protocolos

| Conjunto | Registros | REAL | FAKE | Uso |
|---|---:|---:|---:|---|
| Climate-FEVER limpio | 753 | 500 | 253 | Primera fase, excluyendo ambiguos |
| Climate-FEVER + ClimateCheck | 954 | 669 | 285 | Experimento de enriquecimiento inicial |
| Climate-FEVER + ClimateCheck + ClimaFactsKG | 1,204 | 669 | 535 | Entrenamiento enriquecido |
| Entrenamiento con benchmark retenido excluido | 1,004 | 569 | 435 | Evaluación honesta contra holdout |
| `holdout_v1` | 200 | 100 | 100 | Evaluación externa retenida; nunca debe entrar al train |

Los 247 registros Climate-FEVER procedentes de `NOT_ENOUGH_INFO` o `DISPUTED` se excluyen por defecto. ClimaFactsKG aporta falsedades con evidencia asociada; ClimateCheck aporta afirmaciones de consenso estricto.

### Protocolos de evaluación

1. **Split estratificado 80/20:** usado para métricas de prueba y estudio de ablación.
2. **Validación cruzada estratificada de 5 folds:** TF-IDF se ajusta dentro de cada fold para evitar fuga de vocabulario/IDF.
3. **Benchmark retenido:** `data/evaluation/holdout_v1.jsonl`. El experimento que se evalúe allí debe usar `--exclude-claims-file` durante entrenamiento.
4. **Calibración de umbral:** predicciones out-of-fold sobre el train; se prioriza recall de FAKE de al menos 0.60 cuando es viable.

## 3. Algoritmos y representaciones evaluadas

### 3.1 Clasificador híbrido Random Forest

El modelo histórico y actual de producción utiliza `RandomForestClassifier` con `class_weight="balanced"`. El vector de la variante completa concatena:

```text
TF-IDF (unigramas/bigramas)
+ métricas de longitud y legibilidad
+ linked_entity_count
+ has_conflict
+ embedding CLS de Transformer
→ Random Forest
→ regla ontológica de prioridad
```

Ventajas: combina señales heterogéneas sin normalización obligatoria, funciona con pocos miles de ejemplos y ofrece entrenamiento razonablemente rápido. Limitación: aprende patrones estadísticos de texto; no verifica evidencia científica por sí solo.

### 3.2 Embeddings Transformer + Logistic Regression

El baseline usa embedding CLS del Transformer más `LogisticRegression(class_weight="balanced")`. El encoder configurado para CPU es `prajjwal1/bert-tiny` (128 dimensiones), no un BERT grande fine-tuned end-to-end. El baseline permite medir el aporte de la representación contextual frente al vector híbrido.

### 3.3 TF-IDF + Logistic Regression

Se evaluaron tres variantes léxicas rápidas:

1. TF-IDF de palabra con la lista inglesa original de stop words.
2. TF-IDF de palabra conservando negaciones.
3. TF-IDF de palabra más TF-IDF de caracteres (`char_wb`, n-gramas 3–5), ambos con `LogisticRegression` balanceada.

Los n-gramas de caracteres ayudan a tolerar variaciones de escritura, formas flexionadas y formulaciones cercanas. La regresión logística es una baseline fuerte y reproducible para representaciones dispersas.

### 3.4 Word2Vec

**No se promovió ni se usó como artefacto final.** Word2Vec crea un vector estático por palabra; al promediar vectores para una afirmación, relaciones como `contributes` frente a `does not contribute` tienden a perder la estructura y negación. El sistema ya dispone de embeddings contextuales Transformer, por lo que Word2Vec no era la inversión de mayor retorno para este problema. Puede mantenerse como baseline académica, pero no como sustituto del modelo actual.

### 3.5 NER, tokenización y ontologías

spaCy y un diccionario ambiental extraen CO2, CH4, carbono, metano, calentamiento global, cambio climático y otros conceptos. RDFLib realiza el enlace con conceptos de SWEET/ENVO/GEMET/AGROVOC simulados/locales.

La regla ontológica se aplica después del clasificador: si existe contradicción explícita de una relación conocida, devuelve `FAKE` con `decision_source=ontology_rule_override`. También respalda un conjunto pequeño y explícito de relaciones canónicas (CO2/metano como gases de efecto invernadero y su contribución al calentamiento) con `decision_source=ontology_rule_support`. Las contradicciones tienen prioridad y las reglas de soporte no se aplican a texto con negación. No se presenta como un clasificador estadístico independiente.

## 4. Corrección de negaciones

Se detectó que `TfidfVectorizer(stop_words="english")` elimina `no`, `not`, `never` y `without`, que son términos semánticamente críticos en fact-checking. Se implementó una lista de stop words que conserva:

```text
no, not, nor, never, neither, without
```

También se ampliaron reglas ontológicas para detectar, entre otras, estas formulaciones:

```text
CO2 has no impact on global warming.
CO2 does not affect climate.
El CO2 no contribuye al calentamiento global.
El dióxido de carbono no tiene impacto en el cambio climático.
El metano no es un gas de efecto invernadero.
```

La suite `scripts/test_negation_rules.py` contiene siete pares críticos y obtuvo **7/7 PASS**. Estas reglas corrigen el caso de negación explícita aunque el clasificador estadístico asigne REAL.

## 5. Resultados de experimentos de entrenamiento

Las siguientes métricas son medias de validación cruzada de 5 folds, salvo indicación contraria. No se deben comparar directamente experimentos que usan diferente conjunto de entrenamiento.

| Experimento | Datos train | Arquitectura | Accuracy CV | Balanced accuracy CV | Macro-F1 CV | F1 REAL CV | Decisión |
|---|---:|---|---:|---:|---:|---:|---|
| `bert_tiny` | 753 | TF-IDF + señales + BERT-tiny + RF | 0.6932 | 0.6792 | 0.6702 | 0.7567 | Base inicial |
| `climatecheck_bert` | 954 | Híbrido BERT-tiny + RF | 0.6247 | 0.6378 | 0.6043 | 0.6926 | No promovido; empeoró |
| `climafacts_tfidf` | 1,204 | TF-IDF + RF, sin Transformer | 0.7284 | 0.7148 | 0.7166 | 0.7740 | Comparador sin BERT |
| `climafacts_bert` | 1,204 | Híbrido BERT-tiny + RF, 3 folds | 0.7442 | 0.7333 | 0.7355 | 0.7831 | Exploratorio; folds distintos |
| `climafacts_bert_5fold` | 1,204 | Híbrido BERT-tiny + RF | 0.7392 | 0.7286 | 0.7297 | 0.7781 | Producción previa |
| `holdout_bert` | 1,004 | Híbrido BERT-tiny + RF; holdout excluido | 0.7221 | 0.7126 | 0.7141 | 0.7616 | **Producción actual** |
| `negation_aware_bert` | 1,204 | Híbrido BERT-tiny + RF; TF-IDF conserva negación | 0.7417 | 0.7350 | 0.7352 | 0.7739 | Requiere evaluación holdout equivalente |
| `negation_word_char_holdout` | 1,004 | TF-IDF palabra+carácter + Logistic Regression | 0.7132 | 0.7031 | 0.7047 | 0.7545 | No promovido |

### Interpretación

- La incorporación exclusiva de ClimateCheck no mejoró el desempeño (`climatecheck_bert`).
- ClimaFactsKG aumentó la disponibilidad de ejemplos FAKE y mejoró la validación interna.
- Conservar negaciones mejoró el híbrido interno respecto a la versión equivalente anterior: macro-F1 `0.7352` frente a `0.7297`.
- `negation_aware_bert` no se promovió todavía porque fue entrenado con los claims del benchmark retenido; debe repetirse con `--exclude-claims-file` antes de hacer una comparación externa válida.
- La selección de producción se basó en la mejor evidencia de generalización disponible, no solo en CV interna.

## 6. Comparación de preprocesamiento léxico

Resultados de `scripts/compare_text_models.py` sobre 1,204 registros, 5 folds. Todas las variantes usan Logistic Regression balanceada y se entrenan sin embeddings Transformer.

| Variante | Accuracy | Balanced accuracy | Macro-F1 | Precisión FAKE | Recall FAKE | F1 FAKE |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF legado | 0.7051 | 0.7040 | 0.7022 | 0.6586 | 0.6935 | 0.6745 |
| TF-IDF conservando negación | 0.7242 | 0.7223 | 0.7211 | 0.6839 | 0.7047 | 0.6930 |
| TF-IDF palabra + carácter, conservando negación | **0.7442** | **0.7407** | **0.7404** | **0.7125** | **0.7103** | **0.7102** |

Conclusión: preservar negación aporta `+0.0189` de macro-F1 frente al TF-IDF legado; añadir n-gramas de caracteres aporta un incremento adicional de `+0.0193`. Es evidencia para mantener esta variante como baseline y para incorporar el preprocesamiento consciente de negación al próximo híbrido validado.

## 7. Benchmark retenido y decisión de promoción

| Candidato entrenado sin benchmark | Accuracy | Balanced accuracy | Macro-F1 | Precisión FAKE | Recall FAKE | F1 FAKE |
|---|---:|---:|---:|---:|---:|---:|
| `holdout_bert` | **0.8250** | **0.8250** | **0.8237** | 0.7778 | **0.9100** | **0.8387** |
| `negation_word_char_holdout` | 0.7950 | 0.7950 | 0.7947 | **0.8172** | 0.7600 | 0.7876 |

`holdout_bert` se promovió porque ofrece el mayor macro-F1, recall y F1 de FAKE sobre 200 afirmaciones que no vio durante entrenamiento. El clasificador Word+Char es más preciso al marcar FAKE (`0.8172`), pero deja escapar más falsedades; para el objetivo de detección de desinformación se priorizó recall FAKE `0.91`.

El modelo de producción actual se mantiene como:

```text
TF-IDF + métricas textuales + señales ontológicas + embedding BERT-tiny
→ Random Forest balanceado
→ override ontológico para contradicciones explícitas
```

## 8. Artefactos y reproducción

| Artefacto | Función |
|---|---|
| `scripts/train_models.py` | Entrena Random Forest híbrido y ablaciones. |
| `scripts/compare_text_models.py` | Compara preprocesamiento TF-IDF y Logistic Regression. |
| `scripts/train_negation_text_model.py` | Entrena el candidato TF-IDF palabra+carácter. |
| `scripts/test_negation_rules.py` | Prueba regresiones críticas de reglas ontológicas. |
| `scripts/evaluate_holdout.py` | Evalúa artefactos sin promocionarlos contra el benchmark retenido. |
| `scripts/promote_experiment.py` | Copia un experimento validado a `app/models/`. |
| `app/models/experiments/` | Conserva artefactos y métricas de experimentos aislados. |

Ejemplos:

```bash
export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/fakenews-clima'

# Comparación rápida de modelos de texto
python scripts/compare_text_models.py

# Entrenamiento híbrido consciente de negación sin fuga del benchmark
python scripts/train_models.py \
  --experiment-name negation_aware_bert_holdout \
  --transformer-model prajjwal1/bert-tiny \
  --exclude-claims-file data/evaluation/holdout_v1.jsonl \
  --cv-folds 5

# Evaluación antes de promoción
python scripts/evaluate_holdout.py \
  --model-directory app/models/experiments/negation_aware_bert_holdout \
  --output app/models/experiments/negation_aware_bert_holdout/holdout_evaluation.json
```

## 9. Limitaciones y siguiente experimento recomendado

1. El benchmark retenido contiene dos fuentes conocidas y solo 200 ejemplos; hace falta una tercera fuente independiente.
2. El dataset de entrenamiento sigue siendo principalmente inglés. Las reglas críticas tienen aliases españoles, pero la traducción automática español→inglés no debe activarse sin un benchmark español propio porque puede alterar negación, intensidad o sujeto de la afirmación.
3. El próximo experimento prioritario es `negation_aware_bert_holdout`: mismo híbrido con preservación de negaciones, pero excluyendo el benchmark. Solo se promociona si supera `macro_f1=0.8237` y `recall_FAKE=0.91` en el holdout, o si se justifica explícitamente un trade-off de precisión/recall.
4. La mejora de mayor alcance a futuro es recuperación de evidencia científica más NLI; la clasificación por texto no sustituye verificación basada en evidencia.
