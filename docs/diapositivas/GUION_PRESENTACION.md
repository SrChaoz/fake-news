# Guion de presentación · Climate Veritas

Duración sugerida: **10 a 14 minutos**. Las frases entre comillas se pueden leer o adaptar. El objetivo es explicar el sistema con rigor: no presentar la clasificación como un veredicto científico definitivo ni comparar métricas de datasets distintos como si fueran equivalentes.

## 1. Portada · Climate Veritas

> “Buenos días. Presento Climate Veritas, una plataforma neuro-simbólica para la detección asistida de desinformación ambiental. El sistema integra aprendizaje automático, reglas ontológicas y recuperación de evidencia para ofrecer una decisión trazable entre REAL y FAKE.”

> “No pretende reemplazar la revisión científica o periodística; pretende priorizar y explicar afirmaciones que merecen atención.”

Transición: “Primero, veamos por qué una clasificación solo basada en texto no es suficiente.”

## 2. Contexto · El problema

> “En desinformación climática, una frase puede sonar razonable y aun así negar conocimiento científico bien establecido. Por ejemplo, afirmar que el CO₂ no contribuye al calentamiento global.”

> “Un modelo estadístico puede fallar ante negaciones, ambigüedad o formulaciones nuevas. Por eso incorporamos conocimiento explícito y no solo patrones aprendidos.”

Transición: “La solución propuesta combina tres capas complementarias.”

## 3. Propuesta · Sistema híbrido

> “La primera capa es el aprendizaje automático, que identifica patrones lingüísticos y semánticos. La segunda son reglas ontológicas, que representan relaciones ambientales conocidas. La tercera recupera evidencia curada cuando hay una coincidencia fuerte.”

> “La clave es que no toda predicción depende del modelo: cuando existe una contradicción crítica, la regla de seguridad tiene prioridad.”

Transición: “Así se conectan estas capas dentro de la arquitectura.”

## 4. Arquitectura

> “El usuario escribe una afirmación en el frontend. FastAPI normaliza el texto, extrae entidades ambientales, consulta reglas y genera las características del modelo.”

> “Después, el clasificador híbrido estima la clase. Finalmente, las reglas ontológicas o la evidencia pueden intervenir cuando son concluyentes. El resultado queda almacenado en PostgreSQL para auditoría.”

> “La regla principal es simple: una contradicción explícita de una relación ambiental canónica se clasifica como FAKE, aunque el modelo estadístico discrepe.”

Transición: “Ahora explicaré el flujo de una predicción.”

## 5. Pipeline neuro-simbólico

> “Primero usamos el texto o claim enviado. Segundo, detectamos entidades como CO₂, metano, calentamiento global o biodiversidad con spaCy y un diccionario de dominio.”

> “Tercero, RDFLib enlaza esas entidades con conceptos ontológicos y revisa relaciones de soporte o contradicción. Por último, combinamos el modelo, las reglas y la evidencia para producir una respuesta JSON explicable.”

Transición: “La parte estadística del sistema se basa en una representación híbrida.”

## 6. Modelo promovido

> “El modelo que actualmente usa la API es `holdout_bert`. No es BERT grande afinado de extremo a extremo. Usa BERT-tiny como extractor contextual de 128 dimensiones, lo cual permite operar en CPU.”

> “El vector final combina TF-IDF de palabras y bigramas, señales de longitud y legibilidad, señales simbólicas y el embedding contextual de BERT-tiny. Un Random Forest balanceado produce la probabilidad de REAL.”

> “El umbral para REAL se calibró en 0.56. Si no interviene una regla ni evidencia fuerte, esa es la decisión que llega al usuario.”

Transición: “Pero clasificar no es lo mismo que verificar, por eso añadimos evidencia.”

## 7. Evidencia selectiva

> “La recuperación compara el claim con correcciones y abstracts curados. Solo si la similitud es alta y claramente mejor que las alternativas se etiqueta como SUPPORTED o REFUTED.”

> “Una coincidencia débil se marca como evidencia insuficiente y no se muestra como una fuente falsa de certeza. Esto evita justificar una predicción con un documento tangencial.”

Transición: “Para elegir el modelo no usamos solo una métrica de entrenamiento.”

## 8. Protocolo experimental

> “Trabajamos con 1,004 registros de entrenamiento y un holdout balanceado de 200 afirmaciones. Los claims del holdout se excluyeron antes de entrenar el modelo promovido.”

> “Además del split estratificado, usamos validación cruzada de cinco folds y calibración del umbral con predicciones out-of-fold. Esto reduce el riesgo de fuga de información.”

> “Priorizamos macro-F1 y recall de FAKE, porque el objetivo es no dejar pasar desinformación.”

Transición: “Estos son los resultados que justificaron la promoción del modelo.”

## 9. Resultados del holdout

> “En el benchmark retenido, `holdout_bert` obtuvo 82.5% de accuracy y 0.8237 de macro-F1.”

> “Para la clase FAKE, el recall fue 91%. En términos simples: detectó 91 de cada 100 afirmaciones falsas del holdout. Su precisión FAKE fue 77.78%, por lo que existen falsos positivos y eso debe comunicarse.”

> “Se eligió frente al modelo de word-plus-character Logistic Regression porque obtuvo mejor macro-F1 y dejó escapar menos contenido FAKE.”

Transición: “Es útil contrastar este resultado con la literatura, pero sin comparar indebidamente.”

## 10. Estado del arte / referencia china

> “Como referencia externa usamos MCFEND, un benchmark chino multifuente de detección de fake news. Su BERT-EMO reporta macro-F1 de 0.818 en evaluación cross-source.”

> “Nuestro macro-F1 de 0.8237 no prueba que Climate Veritas sea superior: son idiomas, fuentes, modalidades y datasets diferentes.”

> “La aportación diferencial de nuestro proyecto está en el manejo explícito de negaciones, las reglas de seguridad, la evidencia selectiva y la trazabilidad de cada decisión.”

Transición: “Estos objetivos se convirtieron en requisitos verificables.”

## 11. Requisitos

> “Los requisitos funcionales cubren análisis individual y por lote, NER ambiental, reglas, explicación, persistencia, historial, métricas y evaluación controlada.”

> “Los no funcionales cubren seguridad, reproducibilidad, explicabilidad, protección contra fuga en evaluación y honestidad al mostrar métricas.”

> “Un punto importante es que el navegador no envía comandos al servidor: solo puede solicitar una evaluación limitada del modelo promovido contra el benchmark autorizado.”

Transición: “Veamos ejemplos concretos que muestran el comportamiento esperado.”

## 12. Casos representativos

> “Estas frases representan pruebas de regresión, no una demostración completa de validez científica.”

> “Las frases que indican que CO₂ o metano son gases de efecto invernadero se clasifican como REAL por soporte ontológico delimitado. Las frases que niegan estas relaciones se clasifican como FAKE mediante una regla crítica.”

> “El soporte también contempla aliases en español. La confianza 100% en estos ejemplos expresa certeza de la regla aplicada, no una garantía universal sobre cualquier noticia.”

Transición: “El desarrollo siguió un ciclo de ciencia de datos y software.”

## 13. Fases metodológicas

> “La primera fase fue ingesta de Climate-FEVER, ClimateCheck y ClimaFactsKG. Luego realizamos limpieza, exclusión de registros ambiguos y preservación de negaciones.”

> “Después diseñamos características, entrenamos varias alternativas, validamos con CV y holdout, promovimos el mejor modelo y desplegamos API y frontend.”

> “La última fase es monitoreo: historial, pruebas E2E, exportación de métricas y reevaluación controlada.”

> “Estas son fases de desarrollo de software y ciencia de datos; el sistema no es un modelo de predicción meteorológica.”

Transición: “Uno de los hallazgos más claros fue el tratamiento de negaciones.”

## 14. Hallazgo: negaciones

> “Descubrimos que la lista estándar de stop words podía eliminar `no`, `not`, `never` y `without`. Para fact-checking, borrar estas palabras puede invertir completamente el significado.”

> “Conservar negaciones mejoró el macro-F1 de la baseline TF-IDF de 0.7022 a 0.7211. Al añadir n-gramas de caracteres llegó a 0.7404.”

> “Además, la suite de reglas críticas de negación obtuvo 7 de 7 pruebas aprobadas.”

Transición: “Finalmente, debemos presentar los límites y el trabajo futuro.”

## 15. Límites y cierre

> “Climate Veritas aporta una señal explicable, reglas auditables, evidencia curada y una evaluación reproducible. Sin embargo, no sustituye revisión humana ni prueba factual universal.”

> “El principal límite es la generalización: el benchmark retenido tiene 200 casos y los datos de entrenamiento son principalmente en inglés. También falta análisis multimodal y un benchmark externo grande en español.”

> “El siguiente paso recomendado es evaluar el modelo consciente de negación sin fuga de holdout, crear un benchmark multilingüe independiente y evolucionar hacia recuperación de evidencia más inferencia NLI.”

> “En conclusión, Climate Veritas combina precisión estadística, conocimiento explícito y supervisión humana para apoyar el análisis responsable de desinformación ambiental. Gracias.”

## Preguntas frecuentes para la defensa

**¿Por qué 82.5% y no 99%?**  
Porque la tarea es abierta, las fuentes y las redacciones cambian, y un 99% sin un benchmark grande e independiente sería una afirmación no sustentada. Se prefiere una métrica honesta con sus límites documentados.

**¿Por qué usar Random Forest y BERT-tiny?**  
Random Forest integra características heterogéneas con pocos miles de muestras; BERT-tiny añade contexto en CPU. Es una decisión de coste, reproducibilidad y tamaño de datos, no la afirmación de que sea el mejor Transformer posible.

**¿La regla ontológica reemplaza al modelo?**  
No. Solo cubre relaciones canónicas y explícitas. En los demás casos decide el modelo o la evidencia fuerte recuperada.

**¿Por qué no afirmar que supera a MCFEND?**  
Porque MCFEND trabaja en chino, con fuentes y modalidades distintas. La comparación sirve para contextualizar, no para proclamar una victoria entre benchmarks incompatibles.

**¿Qué significa 91% de recall FAKE?**  
Que detectó 91 de cada 100 falsedades del holdout. No es lo mismo que precisión FAKE: al buscar detectar más falsedades, puede marcar como FAKE algunas afirmaciones reales.
