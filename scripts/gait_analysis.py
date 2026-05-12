"""
gait_analysis.py — Análisis biomecánico de carrera
Detección de eventos de zancada, métricas espacio-temporales,
análisis articular normalizado al ciclo y cálculo de asimetrías.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# FILTRADO
# ============================================================

def _butter_lowpass(data, cutoff=6.0, fs=60.0, order=4):
    """Filtro Butterworth paso bajo."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)


# ============================================================
# DETECCIÓN DE EVENTOS DE ZANCADA
# ============================================================

def _extract_calc_height(trc_frames, marker_name):
    """Extrae serie temporal de altura (Y) del marcador del calcáneo."""
    heights = []
    times = []
    for frame in trc_frames:
        mk = frame["markers"]
        if marker_name in mk:
            heights.append(mk[marker_name][1])  # Y = vertical
            times.append(frame["time"])
    return np.array(times), np.array(heights)


def detect_gait_events(trc_frames, fps=60.0):
    """
    Detecta eventos de contacto inicial (IC) y despegue (TO) para cada pierna
    basándose en la altura del calcáneo.

    Método: La altura mínima del calcáneo indica la fase de contacto.
    Se detectan los mínimos locales (IC) y los máximos locales (TO)
    de la señal filtrada del calcáneo.

    Returns:
        dict con claves 'R' y 'L', cada una conteniendo:
            - 'IC': lista de tiempos de contacto inicial
            - 'TO': lista de tiempos de despegue
            - 'height': serie temporal de altura filtrada
            - 'time': serie temporal
    """
    # Marcadores del calcáneo
    calc_markers = {
        'R': ['r_calc_study', 'RHeel'],
        'L': ['L_calc_study', 'LHeel']
    }

    events = {}

    for side, candidates in calc_markers.items():
        # Encontrar marcador disponible
        marker = None
        sample_mk = trc_frames[0]["markers"]
        for c in candidates:
            if c in sample_mk:
                marker = c
                break
        if marker is None:
            events[side] = {'IC': [], 'TO': [], 'height': np.array([]), 'time': np.array([])}
            continue

        time_arr, height_arr = _extract_calc_height(trc_frames, marker)

        if len(height_arr) < 10:
            events[side] = {'IC': [], 'TO': [], 'height': height_arr, 'time': time_arr}
            continue

        # Filtrar la señal de altura
        h_filt = _butter_lowpass(height_arr, cutoff=6.0, fs=fps)

        # Calcular velocidad vertical
        vel = np.gradient(h_filt, 1.0 / fps)

        # Umbral adaptativo: el contacto ocurre cuando la altura está cerca del mínimo
        h_range = np.max(h_filt) - np.min(h_filt)
        h_threshold = np.min(h_filt) + 0.20 * h_range  # 20% del rango desde el mínimo

        # Detectar IC: mínimos locales de la altura (contacto con el suelo)
        # Usar prominencia para evitar ruido
        min_distance = int(fps * 0.25)  # Mínimo 0.25s entre contactos
        ic_indices, _ = find_peaks(-h_filt, distance=min_distance, prominence=h_range * 0.1)

        # Detectar TO: máximos locales de la altura (pie en el aire)
        to_indices, _ = find_peaks(h_filt, distance=min_distance, prominence=h_range * 0.1)

        # Convertir a tiempos
        ic_times = time_arr[ic_indices].tolist()
        to_times = time_arr[to_indices].tolist()

        events[side] = {
            'IC': ic_times,
            'TO': to_times,
            'IC_idx': ic_indices.tolist(),
            'TO_idx': to_indices.tolist(),
            'height': h_filt,
            'time': time_arr
        }

    return events


# ============================================================
# MÉTRICAS ESPACIO-TEMPORALES
# ============================================================

def compute_spatiotemporal(events, fps=60.0):
    """
    Calcula métricas espacio-temporales a partir de los eventos detectados.

    Returns:
        dict con métricas para cada lado y globales
    """
    metrics = {}

    for side in ['R', 'L']:
        ev = events.get(side, {})
        ic_list = ev.get('IC', [])
        to_list = ev.get('TO', [])

        side_metrics = {
            'contact_times': [],
            'flight_times': [],
            'stride_times': [],
            'gct_mean': None,
            'gct_std': None,
            'flight_mean': None,
            'flight_std': None,
            'stride_mean': None,
            'stride_std': None,
        }

        if len(ic_list) < 2:
            metrics[side] = side_metrics
            continue

        # Stride time: IC[i] -> IC[i+1]
        for i in range(len(ic_list) - 1):
            side_metrics['stride_times'].append(ic_list[i + 1] - ic_list[i])

        # Ground Contact Time: IC -> siguiente TO
        for ic in ic_list:
            # Buscar el primer TO después de este IC
            to_after = [t for t in to_list if t > ic]
            if to_after:
                gct = to_after[0] - ic
                if 0.05 < gct < 0.8:  # Filtrar valores fisiológicos
                    side_metrics['contact_times'].append(gct)

        # Flight Time: TO -> siguiente IC
        for to in to_list:
            ic_after = [t for t in ic_list if t > to]
            if ic_after:
                ft = ic_after[0] - to
                if 0.0 < ft < 0.8:  # Filtrar valores fisiológicos
                    side_metrics['flight_times'].append(ft)

        # Medias y desviaciones
        if side_metrics['contact_times']:
            ct = side_metrics['contact_times']
            side_metrics['gct_mean'] = np.mean(ct)
            side_metrics['gct_std'] = np.std(ct)

        if side_metrics['flight_times']:
            ft = side_metrics['flight_times']
            side_metrics['flight_mean'] = np.mean(ft)
            side_metrics['flight_std'] = np.std(ft)

        if side_metrics['stride_times']:
            st = side_metrics['stride_times']
            side_metrics['stride_mean'] = np.mean(st)
            side_metrics['stride_std'] = np.std(st)

        metrics[side] = side_metrics

    # Métricas globales
    global_m = {}

    # Cadencia (pasos/min)
    all_strides = []
    for side in ['R', 'L']:
        all_strides.extend(metrics[side].get('stride_times', []))
    if all_strides:
        # stride_time = 2 steps → cadencia = 2 * 60 / stride_time_medio
        mean_stride = np.mean(all_strides)
        global_m['cadence'] = 2 * 60.0 / mean_stride if mean_stride > 0 else 0
    else:
        global_m['cadence'] = 0

    # Step time (IC_R -> IC_L y viceversa)
    ic_r = events.get('R', {}).get('IC', [])
    ic_l = events.get('L', {}).get('IC', [])
    step_times_rl = []
    step_times_lr = []

    for ic in ic_r:
        ic_l_after = [t for t in ic_l if t > ic]
        if ic_l_after:
            st = ic_l_after[0] - ic
            if 0.1 < st < 1.0:
                step_times_rl.append(st)
    for ic in ic_l:
        ic_r_after = [t for t in ic_r if t > ic]
        if ic_r_after:
            st = ic_r_after[0] - ic
            if 0.1 < st < 1.0:
                step_times_lr.append(st)

    global_m['step_time_rl'] = np.mean(step_times_rl) if step_times_rl else None
    global_m['step_time_lr'] = np.mean(step_times_lr) if step_times_lr else None

    # Duty factor
    for side in ['R', 'L']:
        gct = metrics[side].get('gct_mean')
        stride = metrics[side].get('stride_mean')
        if gct and stride and stride > 0:
            metrics[side]['duty_factor'] = gct / stride
        else:
            metrics[side]['duty_factor'] = None

    metrics['global'] = global_m
    return metrics


def compute_asymmetry(val_r, val_l):
    """
    Índice de simetría estándar: SI = (R - L) / ((R + L) / 2) * 100
    Retorna valor en %. Positivo = dominancia derecha.
    """
    if val_r is None or val_l is None:
        return None
    avg = (val_r + val_l) / 2.0
    if abs(avg) < 1e-9:
        return 0.0
    return (val_r - val_l) / avg * 100.0


# ============================================================
# NORMALIZACIÓN AL CICLO DE ZANCADA
# ============================================================

def normalize_to_gait_cycle(mot_df, events, n_points=101):
    """
    Normaliza las columnas articulares del .mot al ciclo de zancada (0-100%).
    Usa los eventos IC para definir cada ciclo.

    Returns:
        dict con claves 'R' y 'L', cada una conteniendo un dict de:
            columna -> array (n_ciclos, n_points)
    """
    # Columnas de interés para running
    joint_cols = {
        'hip_flexion': ('hip_flexion_r', 'hip_flexion_l'),
        'knee_angle': ('knee_angle_r', 'knee_angle_l'),
        'ankle_angle': ('ankle_angle_r', 'ankle_angle_l'),
        'hip_adduction': ('hip_adduction_r', 'hip_adduction_l'),
        'hip_rotation': ('hip_rotation_r', 'hip_rotation_l'),
        'pelvis_tilt': ('pelvis_tilt', 'pelvis_tilt'),
        'pelvis_list': ('pelvis_list', 'pelvis_list'),
        'pelvis_rotation': ('pelvis_rotation', 'pelvis_rotation'),
    }

    # Identificar columna de tiempo
    time_col = None
    for c in mot_df.columns:
        if 'time' in c.lower():
            time_col = c
            break
    if time_col is None:
        time_col = mot_df.columns[0]

    mot_time = mot_df[time_col].values

    normalized = {'R': {}, 'L': {}}

    for side_key, side_label in [('R', 'R'), ('L', 'L')]:
        ic_times = events.get(side_key, {}).get('IC', [])

        if len(ic_times) < 2:
            continue

        for joint_name, (col_r, col_l) in joint_cols.items():
            col = col_r if side_key == 'R' else col_l

            if col not in mot_df.columns:
                continue

            signal = mot_df[col].values
            cycles = []

            for i in range(len(ic_times) - 1):
                t_start = ic_times[i]
                t_end = ic_times[i + 1]

                # Índices en el mot
                idx_start = np.argmin(np.abs(mot_time - t_start))
                idx_end = np.argmin(np.abs(mot_time - t_end))

                if idx_end <= idx_start + 3:
                    continue

                segment = signal[idx_start:idx_end + 1]

                # Interpolar a n_points
                x_orig = np.linspace(0, 100, len(segment))
                x_new = np.linspace(0, 100, n_points)
                interp = np.interp(x_new, x_orig, segment)
                cycles.append(interp)

            if cycles:
                normalized[side_label][joint_name] = np.array(cycles)

    return normalized


# ============================================================
# MÉTRICAS ARTICULARES
# ============================================================

def compute_joint_metrics(normalized, events, mot_df, fps=60.0):
    """
    Extrae métricas articulares clave para análisis de carrera.

    Returns:
        dict con métricas por articulación y lado
    """
    results = {}

    # Para cada articulación, calcular métricas sobre los ciclos normalizados
    joint_analysis = {
        'hip_flexion': {
            'label': 'Flexion Cadera',
            'metrics': ['ic_angle', 'peak_flex', 'peak_ext', 'rom']
        },
        'knee_angle': {
            'label': 'Flexion Rodilla',
            'metrics': ['ic_angle', 'peak_flex_stance', 'peak_flex_swing', 'rom']
        },
        'ankle_angle': {
            'label': 'Flexion Tobillo',
            'metrics': ['ic_angle', 'peak_dorsi', 'peak_plantar', 'rom']
        },
    }

    for joint_name, config in joint_analysis.items():
        results[joint_name] = {'label': config['label']}

        for side in ['R', 'L']:
            side_data = normalized.get(side, {}).get(joint_name)
            if side_data is None or len(side_data) == 0:
                results[joint_name][side] = {}
                continue

            mean_curve = np.mean(side_data, axis=0)

            # Ángulo en contacto inicial (punto 0%)
            ic_angle = mean_curve[0]

            # ROM = max - min del ciclo
            rom = np.max(mean_curve) - np.min(mean_curve)

            side_results = {
                'ic_angle': ic_angle,
                'rom': rom,
                'mean_curve': mean_curve,
                'std_curve': np.std(side_data, axis=0) if len(side_data) > 1 else np.zeros_like(mean_curve),
                'n_cycles': len(side_data),
            }

            if joint_name == 'hip_flexion':
                side_results['peak_flex'] = np.max(mean_curve)
                side_results['peak_ext'] = np.min(mean_curve)

            elif joint_name == 'knee_angle':
                # Stance ~ 0-60%, Swing ~ 60-100%
                stance_end = int(0.6 * len(mean_curve))
                side_results['peak_flex_stance'] = np.max(mean_curve[:stance_end])
                side_results['peak_flex_swing'] = np.max(mean_curve[stance_end:])

            elif joint_name == 'ankle_angle':
                side_results['peak_dorsi'] = np.max(mean_curve)
                side_results['peak_plantar'] = np.min(mean_curve)

            results[joint_name][side] = side_results

        # Asimetrías
        r_data = results[joint_name].get('R', {})
        l_data = results[joint_name].get('L', {})
        asym = {}
        for key in ['ic_angle', 'rom']:
            if key in r_data and key in l_data:
                asym[key] = compute_asymmetry(abs(r_data[key]), abs(l_data[key]))
        results[joint_name]['asymmetry'] = asym

    return results


# ============================================================
# OSCILACIÓN VERTICAL
# ============================================================

def compute_vertical_oscillation(mot_df):
    """
    Calcula la oscilación vertical del centro de masa (pelvis_ty).

    Returns:
        dict con time, signal filtrada, amplitud, métricas
    """
    time_col = None
    for c in mot_df.columns:
        if 'time' in c.lower():
            time_col = c
            break
    if time_col is None:
        time_col = mot_df.columns[0]

    if 'pelvis_ty' not in mot_df.columns:
        return None

    time_arr = mot_df[time_col].values
    pelvis_y = mot_df['pelvis_ty'].values

    # Determinar fps del mot
    if len(time_arr) > 1:
        dt = np.median(np.diff(time_arr))
        fs = 1.0 / dt if dt > 0 else 60.0
    else:
        fs = 60.0

    # Filtrar
    y_filt = _butter_lowpass(pelvis_y, cutoff=8.0, fs=fs)

    # Eliminar tendencia (detrend lineal)
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(time_arr, y_filt, 1)
    trend = P.polyval(time_arr, coeffs)
    y_detrended = y_filt - trend

    # Oscilación = rango pico-a-pico por ciclo (en metros -> cm)
    peaks_up, _ = find_peaks(y_detrended, distance=int(fs * 0.2))
    peaks_down, _ = find_peaks(-y_detrended, distance=int(fs * 0.2))

    if len(peaks_up) > 0 and len(peaks_down) > 0:
        oscillation_m = np.mean(y_detrended[peaks_up]) - np.mean(y_detrended[peaks_down])
        oscillation_cm = abs(oscillation_m) * 100
    else:
        oscillation_cm = 0

    return {
        'time': time_arr,
        'signal': y_filt,
        'detrended': y_detrended,
        'oscillation_cm': oscillation_cm,
        'mean_height_m': np.mean(pelvis_y),
    }


# ============================================================
# TIMELINE DE CONTACTO
# ============================================================

def build_contact_phases(events):
    """
    Construye las fases de contacto/vuelo para visualización tipo timeline.

    Returns:
        dict con 'R' y 'L', cada uno con lista de tuplas (t_start, t_end, phase)
        phase = 'stance' o 'swing'
    """
    phases = {}

    for side in ['R', 'L']:
        ev = events.get(side, {})
        ic_list = sorted(ev.get('IC', []))
        to_list = sorted(ev.get('TO', []))

        side_phases = []

        if not ic_list or not to_list:
            phases[side] = side_phases
            continue

        # Construir secuencia alternando IC -> TO (stance) y TO -> IC (swing)
        all_events = []
        for t in ic_list:
            all_events.append((t, 'IC'))
        for t in to_list:
            all_events.append((t, 'TO'))
        all_events.sort(key=lambda x: x[0])

        for i in range(len(all_events) - 1):
            t0, ev0 = all_events[i]
            t1, ev1 = all_events[i + 1]

            if ev0 == 'IC' and ev1 == 'TO':
                side_phases.append((t0, t1, 'stance'))
            elif ev0 == 'TO' and ev1 == 'IC':
                side_phases.append((t0, t1, 'swing'))

        phases[side] = side_phases

    return phases


# ============================================================
# UTILIDADES DE FORMATO
# ============================================================

def format_metric(value, unit='', decimals=1):
    """Formatea un valor numérico para mostrar."""
    if value is None:
        return "—"
    if unit == 'ms':
        return f"{value * 1000:.0f}"
    elif unit == '%':
        return f"{value:.{decimals}f}"
    elif unit == 'deg':
        return f"{value:.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"


def get_asymmetry_color(asym_pct):
    """Retorna color según nivel de asimetría."""
    if asym_pct is None:
        return "#666666"
    abs_val = abs(asym_pct)
    if abs_val < 5:
        return "#2ecc71"  # Verde — simétrico
    elif abs_val < 10:
        return "#f39c12"  # Amarillo — leve
    elif abs_val < 15:
        return "#e67e22"  # Naranja — moderado
    else:
        return "#e74c3c"  # Rojo — significativo


def get_asymmetry_label(asym_pct):
    """Retorna etiqueta según nivel de asimetría."""
    if asym_pct is None:
        return "Sin datos"
    abs_val = abs(asym_pct)
    if abs_val < 5:
        return "Normal"
    elif abs_val < 10:
        return "Leve"
    elif abs_val < 15:
        return "Moderada"
    else:
        return "Significativa"
