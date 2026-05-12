"""
data_utils.py — Parsers para archivos .trc y .mot de OpenCap/OpenSim
y generación de datos sintéticos de fallback.
"""
import numpy as np
import pandas as pd
import os

# ============================================================
# CONEXIONES ANATÓMICAS COMPLETAS DEL ESQUELETO OPENCAP
# Basadas en el modelo LaiUhlrich2022 de OpenSim
# ============================================================
SKELETON_CONNECTIONS = [
    # Torso
    ("C7_study", "r_shoulder_study"), ("C7_study", "L_shoulder_study"),
    ("r.PSIS_study", "L.PSIS_study"), ("r.ASIS_study", "L.ASIS_study"),
    ("r.ASIS_study", "r.PSIS_study"), ("L.ASIS_study", "L.PSIS_study"),
    # Torso vertical
    ("C7_study", "r.PSIS_study"), ("C7_study", "L.PSIS_study"),
    ("r_shoulder_study", "r.ASIS_study"), ("L_shoulder_study", "L.ASIS_study"),
    # Pierna derecha
    ("r.ASIS_study", "r_knee_study"), ("r_knee_study", "r_mknee_study"),
    ("r_knee_study", "r_ankle_study"), ("r_ankle_study", "r_mankle_study"),
    ("r_ankle_study", "r_calc_study"), ("r_ankle_study", "r_toe_study"),
    ("r_calc_study", "r_toe_study"), ("r_toe_study", "r_5meta_study"),
    ("r_calc_study", "r_5meta_study"),
    # Pierna izquierda
    ("L.ASIS_study", "L_knee_study"), ("L_knee_study", "L_mknee_study"),
    ("L_knee_study", "L_ankle_study"), ("L_ankle_study", "L_mankle_study"),
    ("L_ankle_study", "L_calc_study"), ("L_ankle_study", "L_toe_study"),
    ("L_calc_study", "L_toe_study"), ("L_toe_study", "L_5meta_study"),
    ("L_calc_study", "L_5meta_study"),
    # Brazos
    ("r_shoulder_study", "r_melbow_study"), ("L_shoulder_study", "L_melbow_study"),
]

# Conexiones simplificadas (para datos sintéticos con nombres cortos)
SKELETON_CONNECTIONS_SIMPLE = [
    ("C7","RSHO"),("C7","LSHO"),("RSHO","RELB"),("LSHO","LELB"),
    ("RELB","RWRA"),("LELB","LWRA"),("RSHO","RHIP"),("LSHO","LHIP"),
    ("RHIP","LHIP"),("RHIP","RKNE"),("LHIP","LKNE"),
    ("RKNE","RANK"),("LKNE","LANK"),("RANK","RTOE"),("LANK","LTOE"),
    ("RPSIS","LPSIS"),("RASIS","LASIS"),("RASIS","RPSIS"),("LASIS","LPSIS"),
    ("C7","RPSIS"),("C7","LPSIS"),
]

# Colores por segmento corporal
MARKER_COLORS = {
    "ASIS": "#ff6b6b", "PSIS": "#ff6b6b", "HIP": "#ff6b6b",  # Pelvis
    "knee": "#4ecdc4", "KNE": "#4ecdc4",  # Rodilla
    "ankle": "#45b7d1", "ANK": "#45b7d1",  # Tobillo
    "toe": "#96ceb4", "TOE": "#96ceb4", "calc": "#96ceb4", "meta": "#96ceb4",  # Pie
    "shoulder": "#ffd93d", "SHO": "#ffd93d",  # Hombro
    "elbow": "#6c5ce7", "ELB": "#6c5ce7",  # Codo
    "C7": "#fd79a8", "wrist": "#a29bfe", "WRA": "#a29bfe",  # Otros
}

def get_marker_color(name):
    for key, color in MARKER_COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#00d4ff"

def get_skeleton_connections(marker_names):
    """Determina qué set de conexiones usar según los nombres de marcadores."""
    if any("_study" in m for m in marker_names):
        return SKELETON_CONNECTIONS
    return SKELETON_CONNECTIONS_SIMPLE


# ============================================================
# SEGMENTOS CORPORALES CON VOLUMEN
# Cada segmento: (marcador_A, marcador_B, radio, color)
# ============================================================

# Para datos reales OpenCap (con sufijo _study)
BODY_SEGMENTS = [
    # Torso
    ("C7_study", "r_shoulder_study", 0.03, "#e17055"),
    ("C7_study", "L_shoulder_study", 0.03, "#e17055"),
    ("r_shoulder_study", "r.ASIS_study", 0.04, "#e17055"),
    ("L_shoulder_study", "L.ASIS_study", 0.04, "#e17055"),
    # Brazos
    ("r_shoulder_study", "r_melbow_study", 0.025, "#fdcb6e"),
    ("L_shoulder_study", "L_melbow_study", 0.025, "#fdcb6e"),
    # Muslo
    ("r.ASIS_study", "r_knee_study", 0.045, "#74b9ff"),
    ("L.ASIS_study", "L_knee_study", 0.045, "#74b9ff"),
    # Pierna
    ("r_knee_study", "r_ankle_study", 0.035, "#55efc4"),
    ("L_knee_study", "L_ankle_study", 0.035, "#55efc4"),
    # Pie
    ("r_ankle_study", "r_toe_study", 0.02, "#dfe6e9"),
    ("L_ankle_study", "L_toe_study", 0.02, "#dfe6e9"),
    ("r_ankle_study", "r_calc_study", 0.02, "#dfe6e9"),
    ("L_ankle_study", "L_calc_study", 0.02, "#dfe6e9"),
]

# Para datos sintéticos (nombres cortos)
BODY_SEGMENTS_SIMPLE = [
    ("C7", "RSHO", 0.03, "#e17055"), ("C7", "LSHO", 0.03, "#e17055"),
    ("RSHO", "RHIP", 0.04, "#e17055"), ("LSHO", "LHIP", 0.04, "#e17055"),
    ("RSHO", "RELB", 0.025, "#fdcb6e"), ("LSHO", "LELB", 0.025, "#fdcb6e"),
    ("RELB", "RWRA", 0.02, "#fdcb6e"), ("LELB", "LWRA", 0.02, "#fdcb6e"),
    ("RHIP", "RKNE", 0.045, "#74b9ff"), ("LHIP", "LKNE", 0.045, "#74b9ff"),
    ("RKNE", "RANK", 0.035, "#55efc4"), ("LKNE", "LANK", 0.035, "#55efc4"),
    ("RANK", "RTOE", 0.02, "#dfe6e9"), ("LANK", "LTOE", 0.02, "#dfe6e9"),
]

# Polígonos de superficie (marcadores que forman una malla rellena)
TORSO_MESH = {
    "real": ["r_shoulder_study", "L_shoulder_study", "L.ASIS_study", "r.ASIS_study"],
    "simple": ["RSHO", "LSHO", "LHIP", "RHIP"],
}
PELVIS_MESH = {
    "real": ["r.ASIS_study", "L.ASIS_study", "L.PSIS_study", "r.PSIS_study"],
    "simple": ["RASIS", "LASIS", "LPSIS", "RPSIS"],
}

def get_body_segments(marker_names):
    """Devuelve segmentos corporales según tipo de datos."""
    if any("_study" in m for m in marker_names):
        return BODY_SEGMENTS
    return BODY_SEGMENTS_SIMPLE

def get_mesh_groups(marker_names):
    """Devuelve grupos de malla (torso, pelvis) según tipo de datos."""
    key = "real" if any("_study" in m for m in marker_names) else "simple"
    return {"torso": TORSO_MESH[key], "pelvis": PELVIS_MESH[key]}


def make_cylinder_mesh(p1, p2, radius=0.03, n_sides=8):
    """
    Genera vértices y caras para un cilindro 3D entre dos puntos.
    Devuelve (x, y, z, i, j, k) para plotly Mesh3d.
    """
    p1, p2 = np.array(p1), np.array(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    d = direction / length

    # Encontrar un vector perpendicular
    if abs(d[0]) < 0.9:
        perp = np.cross(d, np.array([1, 0, 0]))
    else:
        perp = np.cross(d, np.array([0, 1, 0]))
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(d, perp)

    # Generar vértices del cilindro
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    verts = []
    for angle in angles:
        offset = radius * (np.cos(angle) * perp + np.sin(angle) * perp2)
        verts.append(p1 + offset)
        verts.append(p2 + offset)

    verts = np.array(verts)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # Generar caras (triángulos)
    ii, jj, kk = [], [], []
    for s in range(n_sides):
        s_next = (s + 1) % n_sides
        v0 = s * 2
        v1 = s * 2 + 1
        v2 = s_next * 2
        v3 = s_next * 2 + 1
        # Dos triángulos por cara lateral
        ii += [v0, v1]
        jj += [v1, v3]
        kk += [v2, v2]

    return x, y, z, ii, jj, kk


def make_quad_mesh(points):
    """
    Genera malla para un cuadrilátero (4 puntos) como 2 triángulos.
    points: lista de 4 tuplas (x,y,z)
    """
    pts = np.array(points)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Dos triángulos: 0-1-2 y 0-2-3
    return x, y, z, [0, 0], [1, 2], [2, 3]


# ============================================================
# PARSERS
# ============================================================

def parse_mot(path):
    """Parsea archivo .mot de OpenSim, ignorando encabezado."""
    with open(path, "r") as f:
        raw = f.readlines()
    # Buscar línea 'endheader' o la fila de columnas
    start = 0
    for i, line in enumerate(raw):
        if "endheader" in line.lower():
            start = i + 1
            break
    # La siguiente línea no vacía son las columnas
    while start < len(raw) and not raw[start].strip():
        start += 1
    if start >= len(raw):
        return pd.DataFrame()
    cols = raw[start].strip().split("\t")
    if len(cols) < 2:
        cols = raw[start].strip().split()
    data = []
    for line in raw[start + 1:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            parts = line.strip().split()
        if len(parts) >= len(cols):
            try:
                data.append([float(x) for x in parts[:len(cols)]])
            except ValueError:
                continue
    return pd.DataFrame(data, columns=cols)


def parse_trc(path):
    """Parsea archivo .trc y devuelve lista de frames con coordenadas de marcadores."""
    with open(path, "r") as f:
        raw = f.readlines()
    # Buscar línea de marcadores
    mk_line = -1
    for i, line in enumerate(raw):
        stripped = line.strip()
        if stripped.startswith("Frame#") or stripped.startswith("Frame"):
            mk_line = i
            break
    if mk_line < 0:
        mk_line = 3  # Posición típica en .trc
    parts = raw[mk_line].strip().split("\t")
    markers = [h.strip() for h in parts[2:] if h.strip()]
    data_start = mk_line + 2  # Saltar sub-header X/Y/Z
    frames = []
    for line in raw[data_start:]:
        vals = line.strip().split("\t")
        if len(vals) < 5:
            continue
        try:
            frame_num = int(float(vals[0]))
            time_val = float(vals[1])
            coords = [float(x) for x in vals[2:]]
            marker_data = {}
            for mi, mk in enumerate(markers):
                idx = mi * 3
                if idx + 2 < len(coords):
                    marker_data[mk] = (coords[idx], coords[idx + 1], coords[idx + 2])
            frames.append({"frame": frame_num, "time": time_val, "markers": marker_data})
        except (ValueError, IndexError):
            continue
    return frames, markers


# ============================================================
# GENERADORES SINTÉTICOS (FALLBACK)
# ============================================================

def gen_synthetic_mot(output_path):
    t = np.linspace(0, 2, 200)
    kr = 15 + 60 * np.sin(np.pi * t / 2) ** 2 + np.random.normal(0, 2, len(t))
    kl = 15 + 55 * np.sin(np.pi * t / 2) ** 2 + np.random.normal(0, 2, len(t))
    lines = ["Coordinates\n", "version=1\n", f"nRows={len(t)}\n", "nColumns=3\n",
             "inDegrees=yes\n", "\n", "endheader\n",
             "time\tknee_angle_r\tknee_angle_l\n"]
    for i in range(len(t)):
        lines.append(f"{t[i]:.4f}\t{kr[i]:.4f}\t{kl[i]:.4f}\n")
    with open(output_path, "w") as f:
        f.writelines(lines)
    return output_path


def gen_synthetic_trc(output_path):
    n, fps = 200, 100
    t = np.linspace(0, 2, n)
    markers = ["C7","RSHO","LSHO","RELB","LELB","RWRA","LWRA",
               "RASIS","LASIS","RPSIS","LPSIS",
               "RHIP","LHIP","RKNE","LKNE","RANK","LANK","RTOE","LTOE"]
    base = {
        "C7":(0,1.55,-0.05),"RSHO":(0.2,1.45,0),"LSHO":(-0.2,1.45,0),
        "RELB":(0.35,1.15,0.02),"LELB":(-0.35,1.15,0.02),
        "RWRA":(0.35,0.85,0.05),"LWRA":(-0.35,0.85,0.05),
        "RASIS":(0.12,0.95,0.08),"LASIS":(-0.12,0.95,0.08),
        "RPSIS":(0.08,0.92,-0.08),"LPSIS":(-0.08,0.92,-0.08),
        "RHIP":(0.1,0.9,0),"LHIP":(-0.1,0.9,0),
        "RKNE":(0.1,0.5,0.05),"LKNE":(-0.1,0.5,0.05),
        "RANK":(0.1,0.08,0),"LANK":(-0.1,0.08,0),
        "RTOE":(0.1,0.02,0.15),"LTOE":(-0.1,0.02,0.15),
    }
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\tsynth.trc\n",
        f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{fps}\t{fps}\t{n}\t{len(markers)}\tm\t{fps}\t1\t{n}\n",
        "Frame#\tTime\t" + "\t\t\t".join(markers) + "\n",
        "\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(markers))]) + "\n",
    ]
    for f_idx in range(n):
        phase = np.sin(np.pi * t[f_idx] / 2) ** 2
        dy = -0.30 * phase
        asym = 0.03 * np.sin(2 * np.pi * t[f_idx]) * phase
        vals = [str(f_idx + 1), f"{t[f_idx]:.4f}"]
        for mk in markers:
            bx, by, bz = base[mk]
            y_off = dy if "TOE" not in mk else dy * 0.15
            side_asym = asym if "R" in mk else -asym
            vals += [
                f"{bx + side_asym + np.random.normal(0, 0.003):.5f}",
                f"{by + y_off + np.random.normal(0, 0.003):.5f}",
                f"{bz + np.random.normal(0, 0.003):.5f}",
            ]
        lines.append("\t".join(vals) + "\n")
    with open(output_path, "w") as f:
        f.writelines(lines)
    return output_path
