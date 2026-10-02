# Diseño: puesta en marcha nativa local

## Objetivo

Dejar Climate Veritas listo para utilizarse en una sola máquina, sin Docker ni
despliegue externo. Después de la preparación inicial, la persona operadora
debe poder iniciar PostgreSQL, la API y el panel desde comandos documentados y
verificar que el flujo completo funciona.

## Arquitectura local

PostgreSQL se ejecutará como un servicio local de la máquina y almacenará una
base `fakenews-clima`. El backend se ejecutará dentro de `.venv` con Uvicorn en
`127.0.0.1:8000`; el frontend usará Next.js en `127.0.0.1:3000`. El frontend
apuntará a la API mediante `NEXT_PUBLIC_API_BASE_URL`. La API se conectará a la
base mediante `DATABASE_URL` y cargará los artefactos versionados de
`app/models/`.

La configuración local no se versionará: `.env` y `frontend/.env.local` se
generarán desde sus ejemplos y permanecerán ignorados. Se añadirá una forma
reproducible de establecer la URL de la base de datos sin exponer secretos en
el repositorio.

## Datos y modelo

La preparación creará la base, aplicará `docs/schema.sql` y restaurará
`docs/database_data.sql`. La operación será explícita e idempotente: el script
no eliminará una base existente ni un historial de predicciones sin una acción
deliberada de la persona operadora.

El modelo de producción no se reentrenará ni se promocionará durante el
bootstrap. Se usarán los artefactos ya promovidos (`random_forest_model.pkl`,
`tfidf.pkl` y su configuración). El setup descargará y verificará el encoder
`prajjwal1/bert-tiny`, pues es una dependencia de inferencia indicada por el
artefacto actual. Si no puede descargarse, el comando fallará con una
instrucción clara; la API no iniciará con un modelo incompleto.

## Automatización

Se crearán comandos para: crear/actualizar `.venv` e instalar dependencias de
Python; instalar dependencias del frontend usando el lockfile; preparar la
base; descargar/verificar el encoder; iniciar API y frontend; y ejecutar la
verificación E2E. Los comandos no asumirán que hay procesos ya en ejecución y
mostrarán errores accionables cuando PostgreSQL no esté activo o la URL no sea
válida.

## Validación

La validación comprenderá importaciones del backend, las pruebas existentes de
reglas y evidencia, comprobación de tipos y build de Next.js, arranque de la
API, y `scripts/verify_e2e.py` contra una base restaurada. El criterio de
terminación es que ambos servicios arranquen localmente y el flujo
`/health` → `/predict` → `/history` complete correctamente.

## Fuera de alcance

No se publica ningún servicio ni dato en Internet; no se cambia el algoritmo
de clasificación, no se reentrena el modelo y no se sobrescribe un experimento
promovido. Una promoción futura requerirá una evaluación independiente y una
decisión explícita.
