"""
opencap_api.py — Módulo ligero para descargar datos de OpenCap.
Replica las funciones clave del repo oficial sin importar opensim en el módulo API.
"""
import os
import requests
import urllib.request
import shutil

API_URL = "https://api.opencap.ai/"


def get_token_from_env(env_path=None):
    """Lee el token del archivo .env local."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return None
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith("API_TOKEN"):
                # Formato: API_TOKEN="xxx"
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                return token
    return None


def login(username, password):
    """Login directo y devuelve token."""
    resp = requests.post(API_URL + "login/", data={"username": username, "password": password})
    data = resp.json()
    if "token" not in data:
        detail = data.get("detail", "Login fallido.")
        raise Exception(detail)
    return data["token"]


def get_session(session_id, token):
    """Obtiene el JSON completo de una sesión."""
    resp = requests.get(
        f"{API_URL}sessions/{session_id}/",
        headers={"Authorization": f"Token {token}"}
    )
    if resp.status_code == 500:
        raise Exception("Sesión no válida o servidor no responde.")
    data = resp.json()
    if "trials" not in data:
        raise Exception("No tienes acceso a esta sesión.")
    # Ordenar trials por fecha
    data["trials"].sort(key=lambda t: t.get("created_at", ""))
    return data


def get_trial(trial_id, token):
    """Obtiene JSON de un trial específico."""
    resp = requests.get(
        f"{API_URL}trials/{trial_id}/",
        headers={"Authorization": f"Token {token}"}
    )
    return resp.json()


def download_file(url, file_name):
    """Descarga un archivo desde URL."""
    with urllib.request.urlopen(url) as response, open(file_name, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def download_trial_data(session_id, trial_name, output_dir, token):
    """
    Descarga los archivos .trc, .mot y vídeos de un trial.
    Devuelve dict con las rutas de los archivos descargados.
    """
    session = get_session(session_id, token)
    is_mono = session.get("isMono", False)
    
    # Encontrar el trial por nombre
    trial_dict = None
    for t in session["trials"]:
        if t["name"] == trial_name:
            trial_dict = t
            break
    if trial_dict is None:
        available = [t["name"] for t in session["trials"]]
        raise Exception(f"Trial '{trial_name}' no encontrado. Disponibles: {available}")
    
    trial_id = trial_dict["id"]
    trial = get_trial(trial_id, token)
    result_tags = [res["tag"] for res in trial["results"]]
    
    paths = {"trc": None, "mot": None, "video": None, "model": None}
    
    # Descargar marcadores TRC
    if "marker_data" in result_tags:
        trc_dir = os.path.join(output_dir, "MarkerData")
        os.makedirs(trc_dir, exist_ok=True)
        trc_path = os.path.join(trc_dir, f"{trial_name}.trc")
        if not os.path.exists(trc_path):
            url = trial["results"][result_tags.index("marker_data")]["media"]
            download_file(url, trc_path)
        paths["trc"] = trc_path
    
    # Descargar cinemática MOT (IK results)
    if "ik_results" in result_tags:
        mot_dir = os.path.join(output_dir, "OpenSimData", "Kinematics")
        os.makedirs(mot_dir, exist_ok=True)
        mot_path = os.path.join(mot_dir, f"{trial_name}.mot")
        if not os.path.exists(mot_path):
            url = trial["results"][result_tags.index("ik_results")]["media"]
            download_file(url, mot_path)
        paths["mot"] = mot_path
    
    # Descargar modelo OpenSim (desde neutral trial si no es mono)
    if not is_mono:
        neutral_ids = [t["id"] for t in session["trials"] if t["name"] == "neutral"]
        if neutral_ids:
            neutral_trial = get_trial(neutral_ids[-1], token)
            neutral_tags = [r["tag"] for r in neutral_trial["results"]]
            if "opensim_model" in neutral_tags:
                model_url = neutral_trial["results"][neutral_tags.index("opensim_model")]["media"]
                model_name = model_url[model_url.rfind("-") + 1:model_url.rfind("?")]
                model_dir = os.path.join(output_dir, "OpenSimData", "Model")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, model_name)
                if not os.path.exists(model_path):
                    download_file(model_url, model_path)
                paths["model"] = model_path
    
    # Descargar vídeos
    paths["video"] = []
    if trial.get("videos"):
        vid_dir = os.path.join(output_dir, "Videos", trial_name)
        os.makedirs(vid_dir, exist_ok=True)
        for i, video in enumerate(trial["videos"]):
            if video.get("video"):
                vid_path = os.path.join(vid_dir, f"cam{i}.mov")
                if not os.path.exists(vid_path):
                    try:
                        download_file(video["video"], vid_path)
                    except Exception:
                        pass
                # Asignar ruta si el archivo existe (descargado ahora o antes)
                if os.path.exists(vid_path):
                    paths["video"].append(vid_path)
    
    return paths


def list_trials(session_id, token):
    """Lista los trials disponibles en una sesión (excluyendo calibration/neutral)."""
    session = get_session(session_id, token)
    trials = []
    for t in session["trials"]:
        if t["name"] not in ("calibration", "neutral"):
            trials.append({
                "name": t["name"],
                "id": t["id"],
                "status": t.get("status", "unknown"),
                "created_at": t.get("created_at", "")
            })
    return trials
