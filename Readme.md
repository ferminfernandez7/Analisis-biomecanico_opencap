# Biomecánica — Análisis de Carrera 3D (OpenCap)

## Descripción
Proyecto de **análisis biomecánico** de carrera a partir de datos 3D obtenidos mediante la plataforma OpenCap. Implementación de una interfaz interactiva tipo dashboard para la evaluación clínica y de rendimiento de deportistas. Se incluye el procesamiento de datos cinemáticos, detección de fases de carrera, renderizado de modelos musculoesqueléticos y generación de métricas avanzadas de asimetría.

## Contenido
- `scripts/app.py` — Dashboard principal interactivo que presenta el reporte visual y las gráficas clínicas.
- `scripts/gait_analysis.py` — Módulo para el procesamiento de la marcha, cálculo de tiempos de contacto/vuelo y métricas articulares.
- `scripts/data_utils.py` — Utilidades para el manejo, filtrado y transformación de datos cinemáticos (`.mot`) y de marcadores (`.trc`).
- `scripts/opencap_api.py` — Integración con la API de OpenCap para la descarga automatizada de sesiones y datos 3D.
- `scripts/opensim_renderer.py` — Sistema de renderizado 3D de modelos esqueléticos utilizando Plotly.
- `scripts/setup_auth.py` — Script de configuración para la autenticación con los servicios de OpenCap.

## Tecnologías
`Biomecánica` · `Python` · `Streamlit` · `OpenCap` · `OpenSim` · `Plotly` · `Pandas` · `SciPy`
