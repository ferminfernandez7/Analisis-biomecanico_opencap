import os
import xml.etree.ElementTree as ET
import numpy as np
import opensim as osim

def parse_vtp(file_path):
    """
    Parsea un archivo .vtp de OpenSim (VTK XML format).
    Devuelve (vertices, faces) donde:
      vertices: array Nx3 de coordenadas
      faces: array Mx3 de índices (triángulos)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    verts = []
    faces = []
    
    # En OpenSim los .vtp suelen tener un solo Piece
    for piece in root.iter('Piece'):
        points_da = None
        polys_da = None
        polys_off = None
        
        # Buscar DataArrays de forma robusta iterando elementos:
        for points in piece.iter('Points'):
            for da in points.iter('DataArray'):
                points_da = da.text.strip()
                
        for polys in piece.iter('Polys'):
            for da in polys.iter('DataArray'):
                if da.attrib.get('Name') == 'connectivity':
                    polys_da = da.text.strip()
                elif da.attrib.get('Name') == 'offsets':
                    polys_off = da.text.strip()
                    
        if points_da:
            v_flat = np.fromstring(points_da, sep=' ')
            verts = v_flat.reshape(-1, 3)
            
        if polys_da and polys_off:
            c_flat = np.fromstring(polys_da, sep=' ', dtype=int)
            o_flat = np.fromstring(polys_off, sep=' ', dtype=int)
            
            # En OpenSim las mallas suelen ser triángulos, pero por si acaso
            idx = 0
            for off in o_flat:
                poly_len = off - idx
                if poly_len == 3:
                    faces.append([c_flat[idx], c_flat[idx+1], c_flat[idx+2]])
                elif poly_len == 4:
                    # Dividir quad en 2 triángulos
                    faces.append([c_flat[idx], c_flat[idx+1], c_flat[idx+2]])
                    faces.append([c_flat[idx], c_flat[idx+2], c_flat[idx+3]])
                idx = off
                
    return verts, np.array(faces)


def load_model_and_kinematics(model_path, mot_path):
    """Carga modelo OpenSim y tabla de cinemática."""
    model = osim.Model(model_path)
    model.initSystem()
    table = osim.TimeSeriesTable(mot_path)
    return model, table


def get_bone_meshes_at_time(model, table, time_idx, geom_dir, vtp_cache=None):
    """
    Para un índice de tiempo, extrae la geometría de todos los huesos,
    transformados a coordenadas globales.
    Devuelve lista de diccionarios con x,y,z, i,j,k y nombre.
    """
    state = model.initSystem()
    
    if vtp_cache is None:
        vtp_cache = {}
    
    # Leer valores de cinemática para ese frame
    times = table.getIndependentColumn()
    if time_idx >= len(times):
        time_idx = len(times) - 1
    t = times[time_idx]
    
    # Asignar coordenadas al state
    coords = model.getCoordinateSet()
    for c in range(coords.getSize()):
        coord = coords.get(c)
        name = coord.getName()
        # Buscar en tabla (los nombres a veces acaban en /value)
        col_name = f"/jointset/{coord.getJoint().getName()}/{name}/value"
        if not table.hasColumn(col_name):
            # Alternativa antigua
            col_name = name
            
        if table.hasColumn(col_name):
            col = table.getDependentColumn(col_name)
            val = col[time_idx]
            # Convertir a radianes si la tabla está en grados y es rotacional
            if coord.getMotionType() == osim.Coordinate.Rotational and table.getTableMetaDataAsString('inDegrees') == 'yes':
                val = np.deg2rad(val)
            coord.setValue(state, val)
            
    # Realizar cálculos cinemáticos con el estado actualizado
    model.realizePosition(state)
    
    meshes = []
    
    # Iterar bodies y sus geometrías
    bodies = model.getBodySet()
    for b in range(bodies.getSize()):
        body = bodies.get(b)
        bname = body.getName()
        transform = body.getTransformInGround(state)
        R = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                R[i,j] = transform.R().get(i,j)
        p = np.array([transform.p()[0], transform.p()[1], transform.p()[2]])
        
        for g in range(body.getPropertyByName('attached_geometry').size()):
            geom_base = body.get_attached_geometry(g)
            geom = osim.Mesh.safeDownCast(geom_base)
            if geom:
                mesh_file = geom.get_mesh_file()
                vtp_path = os.path.join(geom_dir, mesh_file)
                if os.path.exists(vtp_path):
                    if mesh_file not in vtp_cache:
                        vtp_cache[mesh_file] = parse_vtp(vtp_path)
                    
                    verts, faces = vtp_cache[mesh_file]
                    if len(verts) > 0 and len(faces) > 0:
                        # Aplicar transformación: verts_g = R * verts + p
                        scale = np.array([1.0, 1.0, 1.0])
                        sf = geom.get_scale_factors()
                        try:
                            scale = np.array([sf.get(0), sf.get(1), sf.get(2)])
                        except:
                            pass
                            
                        verts_scaled = verts * scale
                        verts_global = (R @ verts_scaled.T).T + p
                        
                        meshes.append({
                            'name': f"{bname}_{mesh_file}",
                            'x': verts_global[:, 0],
                            'y': verts_global[:, 1],
                            'z': verts_global[:, 2],
                            'i': faces[:, 0],
                            'j': faces[:, 1],
                            'k': faces[:, 2]
                        })
                else:
                    pass # print(f"Missing: {vtp_path}")
                        
    return meshes

def get_animated_skeleton(model_path, mot_path, max_frames=50):
    """
    Lee modelo y cinemática, extrae frames espaciados (hasta max_frames),
    y concatena las mallas de cada frame en un único Mesh3d.
    Devuelve lista de dicts con x,y,z,i,j,k (uno por frame).
    """
    model, table = load_model_and_kinematics(model_path, mot_path)
    geom_dir = os.path.join(os.path.dirname(model_path), "Geometry")
    
    total_frames = table.getNumRows()
    step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, step))
    # Asegurar no pasarnos de max_frames
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
        
    vtp_cache = {}
    animated_frames = []
    
    for idx in frame_indices:
        meshes = get_bone_meshes_at_time(model, table, idx, geom_dir, vtp_cache=vtp_cache)
        
        # Concatenar
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        v_offset = 0
        
        for m in meshes:
            # Note: in Plotly, we swap Y/Z, so we do it here directly to save processing in app.py
            # Original: x -> x, y -> up, z -> forward. Plotly: z is up.
            all_x.extend(m['x'])
            all_y.extend(m['z'])  # swapped
            all_z.extend(m['y'])  # swapped
            all_i.extend([i + v_offset for i in m['i']])
            all_j.extend([j + v_offset for j in m['j']])
            all_k.extend([k + v_offset for k in m['k']])
            v_offset += len(m['x'])
            
        animated_frames.append({
            'frame_idx': idx,
            'time': table.getIndependentColumn()[idx],
            'x': all_x, 'y': all_y, 'z': all_z,
            'i': all_i, 'j': all_j, 'k': all_k
        })
        
    return animated_frames
