"""Interfaz Streamlit para consumir la API FastAPI local."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 90


def api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any] | list[Any] | None:
    """Llama a FastAPI y muestra errores accionables en la interfaz."""
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        st.error(f"No se pudo conectar con la API en {API_BASE_URL}. Iníciala antes de usar el frontend.")
    except requests.Timeout:
        st.error("La API tardó demasiado en responder. El primer análisis puede cargar BERT desde caché.")
    except requests.HTTPError:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            pass
        st.error(f"La API devolvió HTTP {response.status_code}. {detail}")
    except ValueError:
        st.error("La API respondió un formato JSON no válido.")
    return None


def apply_theme() -> None:
    """Aplica estilos ligeros para enfatizar los resultados de verificación."""
    st.markdown(
        """
        <style>
        .result-card {padding: 1rem 1.2rem; border-radius: .75rem; margin-bottom: 1rem;}
        .result-real {background: #e8f7ee; border: 1px solid #22a06b;}
        .result-fake {background: #fff0f0; border: 1px solid #d64545;}
        .explanation-card {padding: 1rem; background: #f6f8fa; border-radius: .75rem; border-left: 4px solid #4f7cac;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def analyzer_view() -> None:
    """Renderiza el formulario de análisis individual y su explicación."""
    st.header("🔍 Analizador de Noticias")
    st.caption("Introduce una publicación o afirmación ambiental. El análisis se guardará en el historial.")
    text = st.text_area(
        "Noticia o publicación",
        height=180,
        placeholder="Ejemplo: CO2 does not contribute to global warming",
    )
    if not st.button("Analizar Publicación", type="primary", use_container_width=True):
        return
    if not text.strip():
        st.warning("Escribe una publicación antes de analizarla.")
        return
    with st.spinner("Extrayendo entidades, verificando reglas ontológicas y consultando el modelo..."):
        result = api_request("POST", "/predict", json={"text": text.strip()})
    if not isinstance(result, dict):
        return

    prediction = result.get("prediction", "FAKE")
    confidence = float(result.get("confidence_score", 0))
    color_class = "result-real" if prediction == "REAL" else "result-fake"
    st.markdown(f'<div class="result-card {color_class}"><h2>Predicción: {prediction}</h2></div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    first.metric("Confianza", f"{confidence:.1%}")
    second.metric("Categoría", result.get("category", "Climate Change"))
    third.metric("Entidades NER", len(result.get("detected_entities", [])))
    st.markdown(f'<div class="explanation-card"><strong>Explicación automática</strong><br>{result.get("explanation", "Sin explicación disponible.")}</div>', unsafe_allow_html=True)
    st.caption(f"Afirmación analizada: {result.get('extracted_claim', '')}")

    with st.expander("Entidades ambientales detectadas", expanded=True):
        entities = result.get("detected_entities", [])
        if entities:
            st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron entidades ambientales específicas.")
    with st.expander("Conflictos ontológicos"):
        conflicts = result.get("ontological_conflicts", [])
        if conflicts:
            st.json(conflicts)
        else:
            st.success("No se detectaron conflictos ontológicos explícitos.")


def history_view() -> None:
    """Muestra las predicciones persistidas y sus estadísticas básicas."""
    st.header("📜 Historial de Análisis")
    with st.spinner("Cargando historial desde PostgreSQL..."):
        payload = api_request("GET", "/history?limit=50")
    if not isinstance(payload, dict):
        return
    items = payload.get("items", [])
    if not items:
        st.info("Todavía no hay análisis guardados.")
        return
    frame = pd.DataFrame(items)
    total = len(frame)
    real_count = int((frame["prediction"] == "REAL").sum())
    fake_count = int((frame["prediction"] == "FAKE").sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Análisis mostrados", total)
    col2.metric("REAL", f"{real_count / total:.1%}")
    col3.metric("FAKE", f"{fake_count / total:.1%}")
    display_columns = ["id", "created_at", "input_text", "category", "prediction", "confidence_score", "explanation"]
    visible = frame[[column for column in display_columns if column in frame.columns]].copy()
    if "input_text" in visible:
        visible["input_text"] = visible["input_text"].str.slice(0, 180)
    st.dataframe(visible, use_container_width=True, hide_index=True)


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def metrics_view() -> None:
    """Visualiza métricas, experimentos disponibles y ablaciones."""
    st.header("📊 Experimentos y Ablación")
    with st.spinner("Cargando resultados del entrenamiento..."):
        metrics_payload = api_request("GET", "/metrics")
        experiments_payload = api_request("GET", "/experiments")
        ablation_payload = api_request("GET", "/ablation")
    if not all(isinstance(payload, dict) for payload in (metrics_payload, experiments_payload, ablation_payload)):
        return
    assert isinstance(metrics_payload, dict)
    assert isinstance(experiments_payload, dict)
    assert isinstance(ablation_payload, dict)

    metrics = metrics_payload.get("metrics", {})
    st.subheader("Modelo híbrido en producción")
    columns = st.columns(4)
    for column, label, key in zip(columns, ("Accuracy", "Precision", "Recall", "F1-Score"), ("accuracy", "precision", "recall", "f1_score"), strict=True):
        value = metric_value(metrics, key)
        column.metric(label, f"{value:.1%}" if value is not None else "N/D")
    matrix = metrics.get("confusion_matrix")
    if matrix:
        st.caption(f"Matriz de confusión [FAKE, REAL]: {matrix}")

    st.subheader("Rendimiento entre modelos")
    hybrid = experiments_payload.get("hybrid_model", {})
    baseline = experiments_payload.get("transformer_baseline", {}).get("metrics", {})
    rows = []
    for name, values in (("BERT / línea base", baseline), ("Híbrido", hybrid)):
        f1 = metric_value(values, "f1_score")
        if f1 is not None:
            rows.append({"Modelo": name, "F1-Score": f1, "Accuracy": metric_value(values, "accuracy") or 0.0})
    if rows:
        model_frame = pd.DataFrame(rows).set_index("Modelo")
        st.bar_chart(model_frame)
    roberta_status = experiments_payload.get("roberta", {}).get("status", "not_trained")
    ontology_status = experiments_payload.get("ontology_system", {}).get("status", "unknown")
    st.info(f"RoBERTa: {roberta_status}. Sistema ontológico: {ontology_status}; se usa como regla explicable, no como clasificador independiente con métricas comparables.")

    st.subheader("Impacto del estudio de ablación")
    labels = {
        "full_hybrid": "Sistema completo",
        "without_ontology": "Sin ontologías",
        "without_claim_extraction": "Sin claim extraction",
        "without_ontology_and_claim": "Sin ambos",
    }
    ablation_rows = []
    for key, label in labels.items():
        values = ablation_payload.get(key, {})
        if values:
            ablation_rows.append({"Variante": label, "Accuracy": metric_value(values, "accuracy") or 0.0, "F1-Score": metric_value(values, "f1_score") or 0.0, "Macro-F1": metric_value(values, "macro_f1") or 0.0})
    if ablation_rows:
        st.bar_chart(pd.DataFrame(ablation_rows).set_index("Variante"))
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)


def main() -> None:
    """Punto de entrada de Streamlit."""
    st.set_page_config(page_title="Detector de Desinformación Ambiental", page_icon="🌿", layout="wide")
    apply_theme()
    with st.sidebar:
        st.title("🌿 Climate Fact Check")
        st.caption(f"API: {API_BASE_URL}")
        view = st.radio("Navegación", ("🔍 Analizador de Noticias", "📜 Historial de Análisis", "📊 Experimentos y Ablación"))
        if st.button("Comprobar API"):
            health = api_request("GET", "/health")
            if health:
                st.success("API disponible")
    if view == "🔍 Analizador de Noticias":
        analyzer_view()
    elif view == "📜 Historial de Análisis":
        history_view()
    else:
        metrics_view()


if __name__ == "__main__":
    main()
