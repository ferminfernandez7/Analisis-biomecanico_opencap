"""
Dashboard Biomecánico Avanzado - OpenCap
Pipeline real: descarga datos de la API de OpenCap y visualiza.
Ejecutar: conda activate py311prue && streamlit run scripts/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, tempfile

from data_utils import (parse_mot, parse_trc,
                         get_marker_color, get_skeleton_connections,
                         get_body_segments, get_mesh_groups,
                         make_cylinder_mesh, make_quad_mesh)
from opencap_api import get_token_from_env, get_session, list_trials, download_trial_data, login
import gait_analysis as ga

# --- Config ---
st.set_page_config(page_title="ANALISIS BIOMECANICO", layout="wide")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
/* Hacer el header nativo transparente pero sin ocultarlo, para no perder los botones de la sidebar */
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stStatusWidget"] { visibility: hidden; }

/* Título fijo 100% visible siempre (Fixed Header) */
.main-title-container {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    z-index: 999980; /* Mayor que el sidebar pero menor que el header nativo (999990) */
    padding: 15px 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    text-align: center;
}
.main-title-container h1 { color:#fff; font-size:2.0rem; margin:0; font-weight: 700; }
.main-title-container p { color:#94d2bd; font-size:0.95rem; margin:4px 0 0 0; }

/* Empujar contenido hacia abajo para no quedar tapado por la cabecera fija */
div[data-testid="stMainBlockContainer"] { padding-top: 100px !important; }
section[data-testid="stSidebar"] { padding-top: 100px !important; }

/* Tarjetas métricas */
div[data-testid="stMetric"] { 
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    padding:14px; border-radius:12px; border:1px solid #30475e; 
}

/* Estilización avanzada de Pestañas (Modo Nuevo) */
button[data-baseweb="tab"] {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #8898aa !important;
    background-color: #1a2332 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 12px 24px !important;
    margin-right: 4px !important;
    border: 1px solid #2a3b4c !important;
    border-bottom: none !important;
    transition: all 0.3s ease;
}
button[data-baseweb="tab"]:hover {
    color: #cbd5e1 !important;
    background-color: #233040 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #2c5364 !important;
    color: #ffffff !important;
    border-top: 3px solid #94d2bd !important;
    box-shadow: 0 -4px 10px rgba(0,0,0,0.1) !important;
}
</style>

<div class="main-title-container">
    <h1>ANALISIS BIOMECANICO - OPENCAP</h1>
</div>
""", unsafe_allow_html=True)

token = get_token_from_env()

if not token:
    st.markdown("<br><br><h3 style='text-align: center;'>🔒 Acceso Requerido</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Por favor, inicia sesión con tu cuenta de OpenCap para continuar.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        user = st.text_input("Usuario", key="user")
        pwd = st.text_input("Contraseña", type="password", key="pwd")
        if st.button("Iniciar Sesión", use_container_width=True):
            try:
                new_token = login(user, pwd)
                env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
                with open(env_path, "w") as f:
                    f.write(f'API_TOKEN="{new_token}"\n')
                st.success("✅ Login exitoso")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    st.stop()

# ===================== SIDEBAR: SELECCIÓN (SOLO SI HAY TOKEN) =====================
def logout_callback():
    env_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_p):
        try:
            os.remove(env_p)
        except:
            pass
    # Limpiar estado de sesión por completo para un logout limpio
    for key in list(st.session_state.keys()):
        del st.session_state[key]

with st.sidebar:
    st.success("Autenticado en OpenCap")
    st.button("Cerrar Sesión", on_click=logout_callback)

    st.markdown("---")
    st.markdown("### Sesión y trial")
    session_id = st.text_input(
        "Session ID",
        value="4d5c3eb1-1a59-4ea1-9178-d3634610561c",
        help="ID de la sesión en app.opencap.ai/session/<id>"
    )

    # Listar trials disponibles
    @st.cache_data(show_spinner=False, ttl=600)
    def cached_list_trials(sid, tkn):
        return list_trials(sid, tkn)

    trial_names_available = []
    if token and session_id:
        try:
            trials_list = cached_list_trials(session_id, token)
            trial_names_available = [t["name"] for t in trials_list]
        except Exception as e:
            st.error(f"Error listando trials: {e}")

    # Lógica de botón de descarga
    if "active_session" not in st.session_state:
        st.session_state.active_session = None
        st.session_state.active_trial = None

    # Usar un formulario para que cambiar el Trial en la lista NO recargue el script
    with st.form("download_form"):
        selected_trial = "STS"
        if trial_names_available:
            selected_trial = st.selectbox("Trial", trial_names_available)
        else:
            selected_trial = st.text_input("Nombre del trial", value="STS")
            
        submitted = st.form_submit_button("Descargar datos", type="primary", use_container_width=True)
        if submitted:
            st.session_state.active_session = session_id
            st.session_state.active_trial = selected_trial

    st.markdown("---")
    st.caption("Demo session ID:\n`4d5c3eb1-1a59-4ea1-9178-d3634610561c`")

# ===================== CARGA DE DATOS =====================
if st.session_state.active_session is None:
    st.info("Pestañas bloqueadas. Selecciona una sesión y un trial, luego pulsa 'Descargar datos' para comenzar.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_real_data(sid, trial, tkn):
    out_dir = os.path.join(DATA_DIR, sid)
    paths = download_trial_data(sid, trial, out_dir, tkn)
    # Fallback: buscar vídeos en disco si la API no devolvió ruta
    if not paths.get("video"):
        paths["video"] = []
        vid_dir = os.path.join(out_dir, "Videos", trial)
        if os.path.isdir(vid_dir):
            for f in os.listdir(vid_dir):
                if f.endswith((".mov", ".mp4")):
                    paths["video"].append(os.path.join(vid_dir, f))
    return paths

@st.cache_data(show_spinner="Generando animación 3D del esqueleto...", max_entries=20)
def get_cached_animation(model_path, mot_path):
    try:
        import opensim_renderer as orr
        return orr.get_animated_skeleton(model_path, mot_path, max_frames=50)
    except Exception as e:
        print(f"Error animating bones: {e}")
        return []

# Intentar carga
data_paths = None
data_source = "real (OpenCap API)"

if token:
    try:
        data_paths = load_real_data(st.session_state.active_session, st.session_state.active_trial, token)
    except Exception as e:
        st.sidebar.error(f"Error descargando: {e}")
        st.stop()

if not data_paths:
    st.warning("No se han podido cargar los datos reales. Verifica la ID de sesión.")
    st.stop()

# ===================== PESTAÑAS =====================
tab1, tab2, tab3 = st.tabs(["Grabación Original", "Reconstrucción 3D", "Análisis de Carrera"])

@st.cache_data(show_spinner="Parseando cinemática (.mot)...")
def cached_parse_mot(p):
    return parse_mot(p)

@st.cache_data(show_spinner="Preparando coordenadas anatómicas 3D (.trc)...")
def cached_parse_trc(p):
    return parse_trc(p)

# --- TAB 1: Vídeo ---
with tab1:
    st.markdown("### Captura de video original")
    
    vid_paths = data_paths.get("video", [])
    if not isinstance(vid_paths, list):
        vid_paths = [vid_paths] if vid_paths else []
        
    if vid_paths and any(os.path.exists(p) for p in vid_paths):
        valid_vids = [p for p in vid_paths if os.path.exists(p)]
        
        # Usar selector de radio en vez de pestañas para no interferir con el CSS
        cam_names = [f"Cámara {i+1}" for i in range(len(valid_vids))]
        selected_cam = st.radio("Selecciona cámara:", cam_names, horizontal=True)
        vid_p = valid_vids[cam_names.index(selected_cam)]
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with open(vid_p, "rb") as vf:
                vid_bytes = vf.read()
            st.video(vid_bytes)
            st.caption(f"Cámara: {os.path.basename(vid_p)} — Fuente: {data_source}")
    else:
        st.warning("Video no disponible para este trial.")

@st.cache_data(show_spinner="Ensamblando y renderizando visor 3D (puede tardar unos segundos)...", max_entries=5)
def build_3d_figure_cached(trc_path, model_path, mot_path, show_markers):
    import plotly.graph_objects as go
    import numpy as np
    
    trc_frames, trc_markers = cached_parse_trc(trc_path)
    if not trc_frames:
        return None
        
    anim_frames = get_cached_animation(model_path, mot_path) if (model_path and mot_path) else []
    has_rendered_bones = len(anim_frames) > 0
    
    fig3d = go.Figure()
    
    if has_rendered_bones:
        f0 = anim_frames[0]
        fig3d.add_trace(go.Mesh3d(
            x=f0['x'], y=f0['y'], z=f0['z'],
            i=f0['i'], j=f0['j'], k=f0['k'],
            color="#eccc68", opacity=1.0,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1),
            lightposition=dict(x=10, y=10, z=10),
            showlegend=False, name="Esqueleto"
        ))
        
        if show_markers:
            mk = trc_frames[f0['frame_idx']]["markers"]
            colors = [get_marker_color(m) for m in mk]
            xs = [mk[m][0] for m in mk]
            ys = [mk[m][1] for m in mk]
            zs = [mk[m][2] for m in mk]
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=zs, z=ys, mode="markers",
                marker=dict(size=3, color=colors, opacity=0.7),
                name="Marcadores", hovertext=list(mk.keys())
            ))
        
        frames = []
        for f in anim_frames:
            frame_data = [go.Mesh3d(x=f['x'], y=f['y'], z=f['z'])]
            if show_markers:
                trc_idx = f['frame_idx']
                if trc_idx < len(trc_frames):
                    mk = trc_frames[trc_idx]["markers"]
                    mxs = [mk[m][0] for m in mk]
                    mys = [mk[m][1] for m in mk]
                    mzs = [mk[m][2] for m in mk]
                    frame_data.append(go.Scatter3d(x=mxs, y=mzs, z=mys))
            frames.append(go.Frame(data=frame_data, name=str(f['frame_idx'])))
        fig3d.frames = frames
        
        sliders_dict = dict(
            active=0, yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=14, color="#e0e0e0"), prefix="Frame: ", visible=True, xanchor="right"),
            transition=dict(duration=0), pad=dict(b=10, t=10), len=0.85, x=0.15, y=0.0, steps=[]
        )
        for f in frames:
            slider_step = dict(
                args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
                label=f.name, method="animate"
            )
            sliders_dict["steps"].append(slider_step)
            
        fig3d.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=0.0, x=0.12, xanchor="right", yanchor="top",
                pad=dict(t=10, b=10, r=10), direction="left",
                buttons=[
                    dict(label="▶", method="animate",
                         args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
                ]
            )],
            sliders=[sliders_dict]
        )
    else:
        mk = trc_frames[0]["markers"]
        colors = [get_marker_color(m) for m in mk]
        xs = [mk[m][0] for m in mk]
        ys = [mk[m][1] for m in mk]
        zs = [mk[m][2] for m in mk]
        fig3d.add_trace(go.Scatter3d(
            x=xs, y=zs, z=ys, mode="markers",
            marker=dict(size=3, color=colors, opacity=0.7),
            name="Marcadores", hovertext=list(mk.keys())
        ))
        
    if has_rendered_bones:
        all_x, all_y, all_z = [], [], []
        for f in anim_frames[::max(1, len(anim_frames)//10)]:
            all_x.extend(f['x'])
            all_y.extend(f['y'])
            all_z.extend(f['z'])
        cx, cy, cz = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        span = max(np.max(all_x)-np.min(all_x), np.max(all_y)-np.min(all_y), np.max(all_z)-np.min(all_z)) / 2.0
        span = span * 1.15
    else:
        mk_ref = trc_frames[0]["markers"]
        all_x = [mk_ref[m][0] for m in mk_ref]
        all_y = [mk_ref[m][2] for m in mk_ref]
        all_z = [mk_ref[m][1] for m in mk_ref]
        cx, cy, cz = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        span = max(max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z), 0.5) * 0.8
        
    fig3d.update_layout(
        height=700, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(range=[cx-span, cx+span], title="X", backgroundcolor="#0e1117", showgrid=True, gridcolor="#1a2332"),
            yaxis=dict(range=[cy-span, cy+span], title="Z", backgroundcolor="#0e1117", showgrid=True, gridcolor="#1a2332"),
            zaxis=dict(range=[cz-span, cz+span], title="Y", backgroundcolor="#0e1117", showgrid=True, gridcolor="#1a2332"),
            bgcolor="#0e1117", aspectmode="cube", camera=dict(eye=dict(x=1.8, y=1.2, z=0.5))
        ),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0"
    )
    return fig3d

# --- TAB 2: Esqueleto 3D ---
with tab2:
    st.markdown("### Animación 3D del esqueleto")
    
    if data_paths.get("trc") and os.path.exists(data_paths["trc"]):
        show_markers = st.checkbox("Mostrar Marcadores", value=False)
        
        m_path = data_paths.get("model") if data_paths.get("model") and os.path.exists(data_paths["model"]) else None
        mo_path = data_paths.get("mot") if data_paths.get("mot") and os.path.exists(data_paths["mot"]) else None
        
        fig3d = build_3d_figure_cached(data_paths["trc"], m_path, mo_path, show_markers)
        
        if fig3d is None:
            st.error("No se pudieron parsear los datos TRC.")
        else:
            with st.spinner("Proyectando modelo 3D en la pantalla (procesando gráficos)..."):
                st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.error("Archivo TRC no encontrado.")

# --- TAB 3: Análisis de Carrera ---
with tab3:
    _has_mot = data_paths.get("mot") and os.path.exists(data_paths["mot"])
    _has_trc = data_paths.get("trc") and os.path.exists(data_paths["trc"])

    if not _has_mot or not _has_trc:
        st.error("Se requieren archivos .mot y .trc para el análisis de carrera.")
    else:
        with st.spinner("Procesando cinemática y marcadores..."):
            mot_df = cached_parse_mot(data_paths["mot"])
            trc_frames, trc_markers = cached_parse_trc(data_paths["trc"])

        if mot_df.empty or not trc_frames:
            st.error("No se pudieron leer los datos cinemáticos o de marcadores.")
        else:
            # --- Determinar FPS ---
            _tc = [c for c in mot_df.columns if "time" in c.lower()]
            _time_col = _tc[0] if _tc else mot_df.columns[0]
            _times_mot = mot_df[_time_col].values
            _dt_mot = np.median(np.diff(_times_mot)) if len(_times_mot) > 1 else 1/60
            _fps_mot = 1.0 / _dt_mot if _dt_mot > 0 else 60.0

            _trc_times = [f["time"] for f in trc_frames]
            _dt_trc = np.median(np.diff(_trc_times)) if len(_trc_times) > 1 else 1/60
            _fps_trc = 1.0 / _dt_trc if _dt_trc > 0 else 60.0

            # --- Detección de eventos ---
            events = ga.detect_gait_events(trc_frames, fps=_fps_trc)
            spatiotemporal = ga.compute_spatiotemporal(events, fps=_fps_trc)
            normalized = ga.normalize_to_gait_cycle(mot_df, events)
            joint_metrics = ga.compute_joint_metrics(normalized, events, mot_df, fps=_fps_mot)
            vert_osc = ga.compute_vertical_oscillation(mot_df)
            contact_phases = ga.build_contact_phases(events)

            n_events_r = len(events.get('R', {}).get('IC', []))
            n_events_l = len(events.get('L', {}).get('IC', []))

            if n_events_r < 2 and n_events_l < 2:
                st.warning("No se detectaron ciclos de zancada suficientes. "
                           "Verifica que el trial contiene locomoción (marcha o carrera).")
            else:
                # ============================
                # SECCIÓN 1: MÉTRICAS ESPACIO-TEMPORALES
                # ============================
                st.markdown("#### Parametros Espacio-Temporales")
                st.caption("Metricas temporales de cada paso. Valores promedio de todos los ciclos detectados.")

                _gct_r = spatiotemporal['R'].get('gct_mean')
                _gct_l = spatiotemporal['L'].get('gct_mean')
                _ft_r = spatiotemporal['R'].get('flight_mean')
                _ft_l = spatiotemporal['L'].get('flight_mean')
                _str_r = spatiotemporal['R'].get('stride_mean')
                _str_l = spatiotemporal['L'].get('stride_mean')
                _df_r = spatiotemporal['R'].get('duty_factor')
                _df_l = spatiotemporal['L'].get('duty_factor')
                _cad = spatiotemporal['global'].get('cadence', 0)

                _asym_gct = ga.compute_asymmetry(_gct_r, _gct_l)
                _asym_ft = ga.compute_asymmetry(_ft_r, _ft_l)
                _asym_str = ga.compute_asymmetry(_str_r, _str_l)

                # Tarjetas principales
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Cadencia", f"{_cad:.0f} ppm",
                              help="Ref. sprint futbol: 220-260 ppm\nBaja cadencia = zancada larga con mayor impacto. Alta cadencia = patron mas eficiente.")
                with mc2:
                    _tc_avg = np.mean([x for x in [_gct_r, _gct_l] if x is not None]) if any([_gct_r, _gct_l]) else 0
                    st.metric("Tiempo de Contacto (TC)", f"{_tc_avg*1000:.0f} ms",
                              help="Ref. sprint futbol: 80-130 ms\nTC elevado indica mayor tiempo de frenado y menor eficiencia propulsiva.")
                with mc3:
                    _ft_avg = np.mean([x for x in [_ft_r, _ft_l] if x is not None]) if any([_ft_r, _ft_l]) else 0
                    st.metric("Tiempo de Vuelo", f"{_ft_avg*1000:.0f} ms",
                              help="Ref. sprint futbol: 120-160 ms\nMayor vuelo = mayor velocidad de desplazamiento. Ausencia de vuelo = patron de marcha.")
                with mc4:
                    _osc_val = vert_osc['oscillation_cm'] if vert_osc else 0
                    st.metric("Oscilacion Vertical", f"{_osc_val:.1f} cm",
                              help="Ref. sprint futbol: 8-12 cm\nOscilacion excesiva implica energia invertida en movimiento vertical en lugar de horizontal.")

                # Tabla detallada
                _rows = []
                _param_data = [
                    ("Tiempo de Contacto — TC (ms)", _gct_r, _gct_l, 'ms', _asym_gct),
                    ("Tiempo de Vuelo (ms)", _ft_r, _ft_l, 'ms', _asym_ft),
                    ("Tiempo de Zancada (s)", _str_r, _str_l, 's', _asym_str),
                    ("Duty Factor (TC / Zancada)", _df_r, _df_l, 'ratio', ga.compute_asymmetry(_df_r, _df_l)),
                ]
                for label, vr, vl, unit, asym in _param_data:
                    if unit == 'ms':
                        vr_str = f"{vr*1000:.0f}" if vr else "—"
                        vl_str = f"{vl*1000:.0f}" if vl else "—"
                    elif unit == 'ratio':
                        vr_str = f"{vr:.2f}" if vr else "—"
                        vl_str = f"{vl:.2f}" if vl else "—"
                    else:
                        vr_str = f"{vr:.3f}" if vr else "—"
                        vl_str = f"{vl:.3f}" if vl else "—"
                    asym_str = f"{asym:.1f}%" if asym is not None else "—"
                    asym_label = ga.get_asymmetry_label(asym)
                    _rows.append({
                        "Parametro": label,
                        "Derecha": vr_str,
                        "Izquierda": vl_str,
                        "Asimetria (SI%)": asym_str,
                        "Valoracion": asym_label,
                    })
                _st_df = pd.DataFrame(_rows)
                st.dataframe(_st_df, use_container_width=True, hide_index=True)

                # ============================
                # SECCIÓN 2: FASES DE CONTACTO
                # ============================
                st.markdown("---")
                st.markdown("#### Fases de Contacto y Vuelo")
                st.caption("Rojo = pie en el suelo (stance). Azul = pie en el aire (swing). Se puede observar la alternancia y simetria entre piernas.")

                _t_all_r = events['R'].get('time', np.array([]))
                _t_all_l = events['L'].get('time', np.array([]))
                _t_min = float(min(
                    _t_all_r.min() if len(_t_all_r) else 0,
                    _t_all_l.min() if len(_t_all_l) else 0,
                ))
                _t_max = float(max(
                    _t_all_r.max() if len(_t_all_r) else 1,
                    _t_all_l.max() if len(_t_all_l) else 1,
                ))

                fig_timeline = go.Figure()

                _lane_cfg = [('R', 1, 'Derecha', '#e74c3c'), ('L', 0, 'Izquierda', '#e74c3c')]
                for side, y_pos, side_label, sc in _lane_cfg:
                    # Fondo completo (swing = azul muy tenue)
                    fig_timeline.add_shape(
                        type='rect', x0=_t_min, x1=_t_max,
                        y0=y_pos - 0.42, y1=y_pos + 0.42,
                        fillcolor='rgba(52,152,219,0.12)', line=dict(width=0), layer='below'
                    )
                    # Bloques de stance encima
                    for (t0, t1, phase) in contact_phases.get(side, []):
                        if phase == 'stance':
                            fig_timeline.add_shape(
                                type='rect', x0=t0, x1=t1,
                                y0=y_pos - 0.42, y1=y_pos + 0.42,
                                fillcolor='rgba(231,76,60,0.88)', line=dict(width=0)
                            )
                            fig_timeline.add_trace(go.Scatter(
                                x=[(t0 + t1) / 2], y=[y_pos],
                                mode='markers', marker=dict(size=1, opacity=0),
                                showlegend=False,
                                hovertemplate=f"{side_label} — Stance: {(t1-t0)*1000:.0f} ms<extra></extra>"
                            ))
                    # Hover en las zonas de swing
                    for (t0, t1, phase) in contact_phases.get(side, []):
                        if phase == 'swing':
                            fig_timeline.add_trace(go.Scatter(
                                x=[(t0 + t1) / 2], y=[y_pos],
                                mode='markers', marker=dict(size=1, opacity=0),
                                showlegend=False,
                                hovertemplate=f"{side_label} — Swing: {(t1-t0)*1000:.0f} ms<extra></extra>"
                            ))

                # Trazas dummy para leyenda
                fig_timeline.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=14, color='rgba(231,76,60,0.88)', symbol='square'),
                    name='Stance'
                ))
                fig_timeline.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=14, color='rgba(52,152,219,0.35)', symbol='square'),
                    name='Swing'
                ))

                fig_timeline.update_layout(
                    height=180,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0",
                    xaxis=dict(title="Tiempo (s)", gridcolor="#1e2a3a",
                               range=[_t_min - 0.05, _t_max + 0.05], zeroline=False),
                    yaxis=dict(
                        tickmode='array', tickvals=[0, 1],
                        ticktext=['Izquierda', 'Derecha'],
                        range=[-0.65, 1.65], gridcolor="#1e2a3a", zeroline=False,
                    ),
                    margin=dict(l=80, r=20, t=10, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05,
                                xanchor="right", x=1, font=dict(size=11)),
                    showlegend=True,
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

                # ============================
                # SECCIÓN 3: CURVAS ARTICULARES NORMALIZADAS
                # ============================
                st.markdown("---")
                st.markdown("#### Cinematica Articular — Ciclo de Zancada")
                st.caption("Media ± SD de todos los ciclos, normalizado 0-100% (contacto inicial a contacto inicial). Rojo = derecha, verde = izquierda.")

                _joint_config = [
                    ('hip_flexion', 'Cadera — Flexion / Extension', 'Angulo (deg)'),
                    ('knee_angle', 'Rodilla — Flexion', 'Angulo (deg)'),
                    ('ankle_angle', 'Tobillo — Dorsiflexion / Plantarflexion', 'Angulo (deg)'),
                    ('pelvis_tilt', 'Pelvis — Tilt', 'Angulo (deg)'),
                ]

                _jc1, _jc2 = st.columns(2)
                _col_targets = [_jc1, _jc2, _jc1, _jc2]

                for idx, (joint_key, title, ylabel) in enumerate(_joint_config):
                    with _col_targets[idx]:
                        fig_j = go.Figure()
                        x_pct = np.linspace(0, 100, 101)
                        _has_data = False

                        for side, color, name in [('R', '#ff6b6b', 'Derecha'), ('L', '#4ecdc4', 'Izquierda')]:
                            side_norm = normalized.get(side, {}).get(joint_key)
                            if side_norm is not None and len(side_norm) > 0:
                                _has_data = True
                                mean_c = np.mean(side_norm, axis=0)
                                std_c = np.std(side_norm, axis=0) if len(side_norm) > 1 else np.zeros_like(mean_c)

                                fig_j.add_trace(go.Scatter(
                                    x=x_pct, y=mean_c + std_c,
                                    mode='lines', line=dict(width=0), showlegend=False,
                                    hoverinfo='skip'
                                ))
                                fig_j.add_trace(go.Scatter(
                                    x=x_pct, y=mean_c - std_c,
                                    mode='lines', line=dict(width=0), showlegend=False,
                                    fill='tonexty', fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba') if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                                    hoverinfo='skip'
                                ))
                                fig_j.add_trace(go.Scatter(
                                    x=x_pct, y=mean_c,
                                    mode='lines', line=dict(color=color, width=2.5),
                                    name=name
                                ))

                        fig_j.update_layout(
                            title=dict(text=title, font=dict(size=13)),
                            height=300,
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0",
                            xaxis=dict(title="Ciclo de Zancada (%)", gridcolor="#1e2a3a", range=[0, 100]),
                            yaxis=dict(title=ylabel, gridcolor="#1e2a3a"),
                            margin=dict(l=50, r=10, t=40, b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            showlegend=True,
                        )

                        if _has_data:
                            st.plotly_chart(fig_j, use_container_width=True)
                        else:
                            st.info(f"Sin datos para {title}.")

                # ============================
                # SECCIÓN 4: MÉTRICAS ARTICULARES CLAVE (BULLET GAUGES)
                # ============================
                st.markdown("---")
                st.markdown("#### Metricas Articulares Clave")
                st.caption("Franja verde = rango optimo en sprint de futbol. Valor numerico sobre cada marcador. D = derecha (rojo), I = izquierda (verde).")

                # (jkey, mkey, label, ref_lo, ref_hi, ax_min, ax_max, tooltip_hover, label_left, label_right)
                _gauge_items = [
                    ('hip_flexion', 'rom',       'ROM Cadera',       40, 50, -10, 70,
                     'Amplitud total de la cadera en el ciclo',
                     'Corto — zancada limitada', 'Amplio — mayor exigencia lumbar'),
                    ('knee_angle',  'ic_angle',  'Rodilla en IC',     5, 15, -5,  35,
                     'Flexion al contactar. >20 indica overstriding',
                     'Ext. — posible rigidez', 'Flex. — overstriding / frenado'),
                    ('knee_angle',  'rom',       'ROM Rodilla',      55, 75,  20, 95,
                     'Amplitud total de la rodilla en el ciclo',
                     'Reducido — menor recuperacion', 'Elevado — mayor velocidad de balanceo'),
                    ('ankle_angle', 'rom',       'ROM Tobillo',      25, 35,   0, 55,
                     'Limitacion reduce propulsion',
                     'Rigido — menor propulsion', 'Hipermovilidad — posible inestabilidad'),
                    ('ankle_angle', 'peak_dorsi','Pico Dorsiflexion',15, 20,  -5, 35,
                     'Capacidad de absorcion en apoyo',
                     'Limitado — impacto elevado', 'Excesivo — posible compensacion'),
                ]

                fig_bullet = go.Figure()
                _n_gauges = len(_gauge_items)
                _y_labels = []
                _y_positions = []

                for i, (jkey, mkey, label, ref_lo, ref_hi, ax_min, ax_max, tooltip, lbl_left, lbl_right) in enumerate(_gauge_items):
                    jm = joint_metrics.get(jkey, {})
                    vr = jm.get('R', {}).get(mkey)
                    vl = jm.get('L', {}).get(mkey)
                    y_base = (_n_gauges - i) * 1.3
                    _y_labels.append(label)
                    _y_positions.append(y_base)

                    # Franja de referencia
                    fig_bullet.add_shape(
                        type='rect', x0=ref_lo, x1=ref_hi,
                        y0=y_base - 0.45, y1=y_base + 0.45,
                        fillcolor='rgba(46,204,113,0.28)',
                        line=dict(color='rgba(46,204,113,0.7)', width=1.5),
                        layer='below'
                    )
                    # Rango numerico centrado sobre la franja
                    fig_bullet.add_annotation(
                        x=(ref_lo + ref_hi) / 2, y=y_base + 0.58,
                        text=f"<i>{ref_lo}–{ref_hi}</i>",
                        showarrow=False, font=dict(size=10, color='rgba(46,204,113,0.9)'),
                    )
                    # Etiqueta lado izquierdo (por debajo del rango)
                    fig_bullet.add_annotation(
                        x=ref_lo, y=y_base - 0.62,
                        text=f"<span style='color:#e07b54'>◄ {lbl_left}</span>",
                        showarrow=False, xanchor='right',
                        font=dict(size=9, color='#c0785a'),
                    )
                    # Etiqueta lado derecho (por encima del rango)
                    fig_bullet.add_annotation(
                        x=ref_hi, y=y_base - 0.62,
                        text=f"<span style='color:#7fb3d3'>{lbl_right} ►</span>",
                        showarrow=False, xanchor='left',
                        font=dict(size=9, color='#6fa8c8'),
                    )

                    # Linea de fondo gris
                    fig_bullet.add_shape(
                        type='line', x0=ax_min, x1=ax_max,
                        y0=y_base, y1=y_base,
                        line=dict(color='#2a3b4c', width=4),
                        layer='below'
                    )

                    # Marcador derecho
                    if vr is not None:
                        fig_bullet.add_trace(go.Scatter(
                            x=[vr], y=[y_base + 0.12],
                            mode='markers+text', text=[f"{vr:.0f}"], textposition='top center',
                            textfont=dict(size=11, color='#ff6b6b'),
                            marker=dict(symbol='diamond', size=15, color='#ff6b6b',
                                        line=dict(color='#fff', width=1.5)),
                            name='Derecha' if i == 0 else None, showlegend=(i == 0),
                            hovertemplate=f"{label} D: {vr:.1f}\u00b0<br>{tooltip}<extra></extra>"
                        ))
                    # Marcador izquierdo
                    if vl is not None:
                        fig_bullet.add_trace(go.Scatter(
                            x=[vl], y=[y_base - 0.12],
                            mode='markers+text', text=[f"{vl:.0f}"], textposition='bottom center',
                            textfont=dict(size=11, color='#4ecdc4'),
                            marker=dict(symbol='diamond', size=15, color='#4ecdc4',
                                        line=dict(color='#fff', width=1.5)),
                            name='Izquierda' if i == 0 else None, showlegend=(i == 0),
                            hovertemplate=f"{label} I: {vl:.1f}\u00b0<br>{tooltip}<extra></extra>"
                        ))

                fig_bullet.update_layout(
                    height=80 + _n_gauges * 110,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0",
                    xaxis=dict(title="Angulo (deg)", gridcolor="#1e2a3a", zeroline=False),
                    yaxis=dict(
                        tickmode='array', tickvals=_y_positions, ticktext=_y_labels,
                        gridcolor='rgba(0,0,0,0)', zeroline=False,
                    ),
                    margin=dict(l=170, r=40, t=20, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=11)),
                    showlegend=True,
                )
                st.plotly_chart(fig_bullet, use_container_width=True)

                # ============================
                # SECCIÓN 4b: RADAR DE ASIMETRÍAS (CENTRADO, ANCHO COMPLETO)
                # ============================
                st.markdown("---")
                st.markdown("#### Radar de Asimetrias")
                st.caption("Cada eje = |SI%| de una metrica. Cuanto mas cerca del centro, mas simetrico. Zonas concentricas: verde <5%, naranja 10%, rojo >15%.")

                _radar_cats = []
                _radar_vals = []
                _radar_colors_list = []

                for label, val in [("TC", _asym_gct), ("Vuelo", _asym_ft), ("Zancada", _asym_str)]:
                    if val is not None:
                        _radar_cats.append(label)
                        _radar_vals.append(abs(val))
                        _radar_colors_list.append(ga.get_asymmetry_color(val))

                for joint_key, label in [('hip_flexion', 'ROM Cadera'), ('knee_angle', 'ROM Rodilla'), ('ankle_angle', 'ROM Tobillo')]:
                    jm = joint_metrics.get(joint_key, {})
                    asym_data = jm.get('asymmetry', {})
                    rom_asym = asym_data.get('rom')
                    if rom_asym is not None:
                        _radar_cats.append(label)
                        _radar_vals.append(abs(rom_asym))
                        _radar_colors_list.append(ga.get_asymmetry_color(rom_asym))

                if _radar_cats:
                    _rc1, _rc2, _rc3 = st.columns([1, 3, 1])
                    with _rc2:
                        fig_radar = go.Figure()

                        fig_radar.add_trace(go.Scatterpolar(
                            r=[15] * (len(_radar_cats) + 1),
                            theta=_radar_cats + [_radar_cats[0]],
                            fill='toself', fillcolor='rgba(231,76,60,0.08)',
                            line=dict(color='rgba(231,76,60,0.3)', dash='dot'),
                            name='Significativa (>15%)', showlegend=True,
                        ))
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[10] * (len(_radar_cats) + 1),
                            theta=_radar_cats + [_radar_cats[0]],
                            fill='toself', fillcolor='rgba(243,156,18,0.08)',
                            line=dict(color='rgba(243,156,18,0.3)', dash='dot'),
                            name='Moderada (10-15%)', showlegend=True,
                        ))
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[5] * (len(_radar_cats) + 1),
                            theta=_radar_cats + [_radar_cats[0]],
                            fill='toself', fillcolor='rgba(46,204,113,0.08)',
                            line=dict(color='rgba(46,204,113,0.3)', dash='dot'),
                            name='Normal (<5%)', showlegend=True,
                        ))

                        fig_radar.add_trace(go.Scatterpolar(
                            r=_radar_vals + [_radar_vals[0]],
                            theta=_radar_cats + [_radar_cats[0]],
                            fill='toself', fillcolor='rgba(52,152,219,0.2)',
                            line=dict(color='#3498db', width=2.5),
                            marker=dict(size=8, color=_radar_colors_list + [_radar_colors_list[0]]),
                            name='Sujeto',
                        ))

                        fig_radar.update_layout(
                            polar=dict(
                                bgcolor="#0e1117",
                                radialaxis=dict(visible=True, range=[0, max(max(_radar_vals) * 1.3, 20)],
                                                gridcolor="#1e2a3a", tickfont=dict(size=11, color="#888")),
                                angularaxis=dict(gridcolor="#1e2a3a", tickfont=dict(size=13, color="#ccc")),
                            ),
                            paper_bgcolor="#0e1117", font_color="#e0e0e0",
                            height=520, margin=dict(l=60, r=60, t=40, b=60),
                            legend=dict(font=dict(size=11), orientation="h", yanchor="bottom",
                                        y=-0.18, x=0.5, xanchor="center"),
                            showlegend=True,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Datos insuficientes para el radar de asimetrias.")

                # ============================
                # SECCIÓN 5: OSCILACIÓN VERTICAL
                # ============================
                st.markdown("---")
                st.markdown("#### Oscilacion Vertical del Centro de Masa")
                st.caption("Desplazamiento vertical de la pelvis con tendencia eliminada. Menor oscilacion = mayor eficiencia energetica.")

                if vert_osc:
                    _osc1, _osc2 = st.columns([3, 1])

                    with _osc1:
                        fig_osc = go.Figure()
                        fig_osc.add_trace(go.Scatter(
                            x=vert_osc['time'], y=vert_osc['detrended'] * 100,
                            mode='lines', line=dict(color='#9b59b6', width=2),
                            name='Oscilacion vertical'
                        ))
                        fig_osc.add_hline(y=0, line_dash="dot", line_color="#555")
                        fig_osc.update_layout(
                            height=280,
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0",
                            xaxis=dict(title="Tiempo (s)", gridcolor="#1e2a3a"),
                            yaxis=dict(title="Desplazamiento (cm)", gridcolor="#1e2a3a"),
                            margin=dict(l=50, r=10, t=10, b=40),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_osc, use_container_width=True)

                    with _osc2:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                                    padding:20px;border-radius:12px;border:1px solid #30475e;margin-top:10px;">
                            <p style="color:#94d2bd;font-size:0.85rem;margin:0 0 8px 0;">Amplitud Pico-a-Pico</p>
                            <p style="color:#fff;font-size:2.0rem;font-weight:700;margin:0;">
                                {vert_osc['oscillation_cm']:.1f} cm</p>
                            <p style="color:#888;font-size:0.75rem;margin:8px 0 0 0;">
                                Referencia running: 5-8 cm</p>
                            <hr style="border-color:#30475e;margin:12px 0;">
                            <p style="color:#94d2bd;font-size:0.85rem;margin:0 0 8px 0;">Altura Media Pelvis</p>
                            <p style="color:#fff;font-size:1.4rem;font-weight:700;margin:0;">
                                {vert_osc['mean_height_m']:.3f} m</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Columna pelvis_ty no disponible en este trial.")



# --- Footer ---
st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:#666;font-size:.8rem;'>"
    f"OpenCap Dashboard · Datos: {data_source} · Streamlit + Plotly"
    f"</p>", unsafe_allow_html=True
)
