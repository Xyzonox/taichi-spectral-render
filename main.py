import taichi as ti
import taichi.math as tm
import time
import numpy as np
import yaml
import data 

# --- Engine Initialization ---
ti.init(arch=ti.cuda, default_fp=ti.f32)

# --- Scene and Render Configuration ---

ASPECT_RATIO = 1
IMAGE_WIDTH = 800
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)
RESOLUTION_SCALE = 2

# Scene
EPSILON = 1e-4
ESCAPE = 1e2
MAX_PATH_LENGTH = 7
SAMPLES_PER_PIXEL = 7000
SCATTER_BOUNCES = 5

# Spectral
SPECTRAL_BANDS = 25
WAVELENGTH_MAX = 1210
WAVELENGTH_MIN = 10

# Post Processing
EXPOSURE = 1.0
GAMMA = 2.0

WAVELENGTH_STEP = (WAVELENGTH_MAX - WAVELENGTH_MIN) / (SPECTRAL_BANDS - 1)
RENDER_WIDTH = IMAGE_WIDTH // RESOLUTION_SCALE
RENDER_HEIGHT = IMAGE_HEIGHT // RESOLUTION_SCALE

# --- Python-side constants for material property indices ---
# --- For the f32 field ---
PROP_REFRACTIVE_INDEX = 0
PROP_EXTINCTION_COEFFICIENT = 1
PROP_ROUGHNESS = 2
PROP_SCATTERING_COEFFICIENT = 3
PROP_EMISSION = 4
PROP_ANISTROPY_FACTOR = 5
NUM_F32_PROPERTIES = 6 # Total number of float properties

# --- For the int field ---
PROP_IS_TRUE_VOLUME = 0
PROP_SURFACE_EMISSION = 1
NUM_INT_PROPERTIES = 2 # Total number of integer properties

# --- Data Loading

cs_srgb = data.ColourSystem(
    file_path="data/CMF/cie_cmf_xray_swir_vis.txt",
    red=[0.64, 0.33],
    green=[0.30, 0.60],
    blue=[0.15, 0.06],
    white=[0.33242, 0.34743],
    spectral_bands=SPECTRAL_BANDS,
    wavelength_min=WAVELENGTH_MIN,
    wavelength_max=WAVELENGTH_MAX
)
gold_ior = data.RefractiveIndex(
    file_path='data/refraction_index/silicon.yml', 
    spectral_bands=SPECTRAL_BANDS,
    wavelength_min=WAVELENGTH_MIN,
    wavelength_max=WAVELENGTH_MAX
)
k_1 = data.SpecularVec(
    file_path='data/SpecVecs/spectrum_data_k_0_10.txt', 
    scale = 1e-6,
    spectral_bands=SPECTRAL_BANDS,
    wavelength_min=WAVELENGTH_MIN,
    wavelength_max=WAVELENGTH_MAX)

fluorescent_glass = data.EEMMatrix(
    file_path='data/EEMs/uv_green.txt', 
    wavelength_min=WAVELENGTH_MIN,
    wavelength_max=WAVELENGTH_MAX,
    spectral_bands=SPECTRAL_BANDS
)

numEEM = data.EEMMatrix._next_eem_id

# --- Python Data Structures

def create_scene():
    materials = [
        { # 0: Air (Required for volumes)
            "refractive_index": [1.0] * SPECTRAL_BANDS, 
            "extinction_coefficient": [0.0] * SPECTRAL_BANDS,
            "roughness": [0.0] * SPECTRAL_BANDS, 
            "is_true_volume": [1] * SPECTRAL_BANDS,
            "scattering_coefficient": [0.0]* SPECTRAL_BANDS,
            "anistropy_factor":[.8] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 0
        },
        { # 1: White Diffuse
            "refractive_index": [1.4] * SPECTRAL_BANDS, 
            "extinction_coefficient": [0.0] * SPECTRAL_BANDS,
            "roughness": [1.0] * SPECTRAL_BANDS, 
            "is_true_volume": [0] * SPECTRAL_BANDS,
            "scattering_coefficient": [1.0] * SPECTRAL_BANDS,
            "anistropy_factor":[0.0] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 0
        },
        { # 2: Red Diffuse
            "refractive_index": [1.4] * SPECTRAL_BANDS, 
            "extinction_coefficient": k_1.vec,
            "roughness": [0.001] * SPECTRAL_BANDS, 
            "is_true_volume": [0] * SPECTRAL_BANDS,
            "scattering_coefficient": [1.0] * SPECTRAL_BANDS,
            "anistropy_factor":[0.0] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 0
        },
        { # 3: Green Diffuse
            "refractive_index": [1.4] * SPECTRAL_BANDS, 
            "extinction_coefficient": [0.0] * SPECTRAL_BANDS,
            "roughness": [1.0] * SPECTRAL_BANDS, 
            "is_true_volume": [0] * SPECTRAL_BANDS,
            "scattering_coefficient": [1.0] * SPECTRAL_BANDS,
            "anistropy_factor":[0.0] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 0
        },
        { # 4: Light Emission
            "refractive_index": [1.0] * SPECTRAL_BANDS, 
            "extinction_coefficient": [0.0] * SPECTRAL_BANDS,
            "roughness": [0.0] * SPECTRAL_BANDS, 
            "is_true_volume": [0] * SPECTRAL_BANDS,
            "scattering_coefficient": [0.0] * SPECTRAL_BANDS,
            "surface_emission": [1] * SPECTRAL_BANDS, 
            "emission": [50.0]*8 + [15.0] * (SPECTRAL_BANDS - 8),
            "eem_id": 0
        },
        { # 5: Gold Metal
            "refractive_index": gold_ior.n, 
            "extinction_coefficient": gold_ior.k,
            "roughness": [0.001] * SPECTRAL_BANDS, 
            "is_true_volume": [1] * SPECTRAL_BANDS,
            "scattering_coefficient": [0.0] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 0
        },
        { # 6: Fluorescent Glass Dielectric
            "refractive_index": [3.0] * SPECTRAL_BANDS, 
            "extinction_coefficient": [1e-2]*8 + [0.0]*(SPECTRAL_BANDS-8),
            "roughness": [0.001] * SPECTRAL_BANDS, 
            "is_true_volume": [1] * SPECTRAL_BANDS,
            "scattering_coefficient": [0.0] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 1
        },
        { # 7: Glass Dielectric
            "refractive_index": [3.0] * SPECTRAL_BANDS, 
            "extinction_coefficient": [0.0]*(SPECTRAL_BANDS),
            "roughness": [0.001] * SPECTRAL_BANDS, 
            "is_true_volume": [1] * SPECTRAL_BANDS,
            "scattering_coefficient": [0.0] * SPECTRAL_BANDS,
            "surface_emission": [0] * SPECTRAL_BANDS, 
            "emission": [0.0] * SPECTRAL_BANDS,
            "eem_id": 0
        }
    ]
    # --- Geometry ---
    meshes = []

    # Cornell Box Walls (size 2x2x2, centered at origin)
    # Floor
    meshes.append({'vertices': [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], 'faces': [[0, 2, 1], [0, 3, 2]], 'material_id': 1})
    # Ceiling
    meshes.append({'vertices': [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]], 'faces': [[0, 2, 1], [0, 3, 2]], 'material_id': 1})
    # Back Wall
    meshes.append({'vertices': [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]], 'faces': [[0, 2, 1], [0, 3, 2]], 'material_id': 1})
    # Right Wall 
    meshes.append({'vertices': [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]], 'faces': [[0, 2, 1], [0, 3, 2]], 'material_id': 2})
    # Left Wall 
    meshes.append({'vertices': [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]], 'faces': [[0, 2, 1], [0, 3, 2]], 'material_id': 2})

    # Light
    meshes.append({'vertices': [[-0.3, 0.99, -0.3], [-0.3, 0.99, 0.3], [0.3, 0.99, 0.3], [0.3, 0.99, -0.3]], 'faces': [[0, 2, 1], [0, 3, 2]], 'material_id': 4})

    # Add a sphere (approximated by triangles)
    def create_sphere(center, radius, material_id, subdivisions=2):
        # Create a base icosahedron
        t = (1.0 + 5.0**0.5) / 2.0
        vertices = [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ]
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        # Normalize vertices to form a sphere
        vertices = np.array(vertices, dtype=np.float32)
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

        # Subdivide faces
        for _ in range(subdivisions):
            new_faces = []
            mid_points = {}
            for face in faces:
                v1, v2, v3 = face
                m12 = tuple(sorted((v1, v2)))
                m23 = tuple(sorted((v2, v3)))
                m31 = tuple(sorted((v3, v1)))

                for m in [m12, m23, m31]:
                    if m not in mid_points:
                        mid_point = (vertices[m[0]] + vertices[m[1]]) / 2.0
                        mid_points[m] = len(vertices)
                        vertices = np.vstack([vertices, mid_point / np.linalg.norm(mid_point)])

                new_faces.append([v1, mid_points[m12], mid_points[m31]])
                new_faces.append([v2, mid_points[m23], mid_points[m12]])
                new_faces.append([v3, mid_points[m31], mid_points[m23]])
                new_faces.append([mid_points[m12], mid_points[m23], mid_points[m31]])
            faces = new_faces

        final_vertices = (vertices * radius) + np.array(center)
        vertex_normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        return {'vertices': final_vertices.tolist(), 'faces': faces, 'vertex_normals':vertex_normals.tolist(), 'material_id': material_id}

    meshes.append(create_sphere(center=[-0.4, -0.7, -0.3], radius=0.3, material_id=5)) # Gold
    meshes.append(create_sphere(center=[0.4, -0.7, 0.3], radius=0.3, material_id=6)) # Glass

    camera = {
        "origin": [0.0, 0.0, 3.5],
        "lookat": [0.0, 0.0, -1.0],
        "up": [0.0, 1.0, 0.0]
    }
    return {"meshes": meshes, "materials": materials, "camera": camera}

# --- Data Structures ---

SpectralVector = ti.types.vector(SPECTRAL_BANDS, ti.f32)
SpectralBool = ti.types.vector(SPECTRAL_BANDS, ti.i32)
AABB = ti.types.struct(min=tm.vec3, max=tm.vec3)

@ti.dataclass
class Triangle:
    v0: tm.vec3
    v1: tm.vec3
    v2: tm.vec3
    n0: tm.vec3
    n1: tm.vec3
    n2: tm.vec3
    normal: tm.vec3
    material_id: ti.i32
    area: ti.f32

@ti.dataclass
class BVHNode:
    aabb: AABB
    # if num_tris > 0, it's a leaf node, left_child is the start index of triangles
    # if num_tris == 0, it's an internal node, left_child is the index of the left child node
    left_child: ti.i32
    right_child: ti.i32
    num_tris: ti.i32
    tri_start_idx: ti.i32

@ti.dataclass
class MaterialSample:
    refractive_index: ti.f32
    extinction_coefficient: ti.f32
    roughness: ti.f32

    scattering_coefficient: ti.f32
    anistropy_factor: ti.f32
    is_true_volume:ti.i32

    surface_emission: ti.i32
    emission: ti.f32
    eem_id: ti.i32

@ti.dataclass
class Ray:
    position: tm.vec3
    direction: tm.vec3
    throughput: ti.f32
    active_wavelength_idx: ti.i32
    transport_wavelength_idx: ti.i32
    path_length: ti.i32 #Track bounces, 
    has_terminated: ti.i32
    vol_mat_id: ti.i32
    has_escaped: ti.i32
    surface_pdf: ti.f32
    penetrated_tri_idx: ti.i32
    is_reflected: ti.i32

@ti.dataclass
class GeoRay:
    position: tm.vec3
    direction: tm.vec3
    BVH_depth: ti.i32
    depth_to_hit: ti.i32

@ti.dataclass
class ShadowRay:
    position: tm.vec3
    direction: tm.vec3

@ti.dataclass
class HitRecord:
    t: ti.f32 #TODO, rename to "distance_traveled"
    normal: tm.vec3
    shading_normal: tm.vec3
    material_id: ti.i32
    u: ti.f32 # Barycentric coord
    v: ti.f32 # Barycentric coord
    is_hit: ti.i32
    is_front_face: ti.i32
    tri_idx: ti.i32

@ti.dataclass
class VolumeEvent:
    hit_action: ti.i32 #1:hit surface, 2:scattered, 3:fluoresce
    travel_distance: ti.f32
    estimator: ti.f32
    
    normal: tm.vec3
    shading_normal: tm.vec3
    is_front_face: ti.i32
    hit_mat_id: ti.i32
    tri_idx: ti.i32
    
    direction: tm.vec3
    new_wavelength_idx: ti.i32
    penetrated_tri_idx: ti.i32

@ti.dataclass
class SurfaceEvent:
    did_reflect: ti.i32
    position: tm.vec3
    direction: tm.vec3
    estimator: ti.f32
    pdf: ti.f32
    new_wavelength: ti.f32

@ti.dataclass
class NEE_Sample:
    contribution_factor: ti.f32 # This is the main term: (BSDF * G_Term / PDF_light)
    emission: ti.f32
    weight: ti.f32

# --- Global Taichi Fields ---

# Pixel Space
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(IMAGE_WIDTH, IMAGE_HEIGHT))
spectral_accumulator = ti.field(ti.f32,shape=(RENDER_WIDTH, RENDER_HEIGHT,SPECTRAL_BANDS))
block_accumulator = ti.field(ti.f32,shape=(RENDER_WIDTH, RENDER_HEIGHT,SPECTRAL_BANDS)) 
pixels_geometry = ti.Vector.field(3, dtype=ti.f32, shape=(RENDER_WIDTH, RENDER_HEIGHT))

# Scene Data
transport_wavelength_idx = ti.field(ti.i32,shape=(RENDER_WIDTH, RENDER_HEIGHT))
active_wavelength_idx = ti.field(ti.i32,shape=(RENDER_WIDTH, RENDER_HEIGHT))
reemission_matrix = ti.Matrix.field(SPECTRAL_BANDS, SPECTRAL_BANDS, dtype=ti.f32, shape=(numEEM))
num_scene_objects = ti.field(ti.i32, shape=())

# Camera
cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_up = ti.Vector.field(3, dtype=ti.f32, shape=())

cam_u = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_v = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_w = ti.Vector.field(3, dtype=ti.f32, shape=())

#Quantifiers
total_light_area_field = ti.field(ti.f32, shape=SPECTRAL_BANDS)
num_lights_per_band = ti.field(ti.i32, shape=SPECTRAL_BANDS)

# Color Conversion Data
cmf_data = ti.Vector.field(3, dtype=ti.f32, shape=SPECTRAL_BANDS)
xyz_to_rgb_matrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
normalization_factor = ti.field(dtype=ti.f32, shape=())
xyz_to_rgb_matrix.from_numpy(cs_srgb.T)
cmf_data.from_numpy(cs_srgb.cmf)
normalization_factor[None] = np.sum(cs_srgb.cmf[:, 1]) * WAVELENGTH_STEP

# --- Scene Compilation ---

def get_scene_dimensions(scene_dict):
    """
    Parses the scene dictionary to determine the required sizes for Taichi fields.
    """
    num_materials = len(scene_dict.get('materials', []))
    
    num_triangles = 0
    for mesh in scene_dict.get('meshes', []):
        num_triangles += len(mesh.get('faces', []))
        
    num_bvh_nodes = max(0, 2 * num_triangles - 1)
    
    num_lights = 0
    py_materials = scene_dict["materials"]
    lights_per_band = [0] * SPECTRAL_BANDS
    for mesh in scene_dict.get('meshes', []):
        mat_id = mesh['material_id']
        is_emissive_at_any_band = any(py_materials[mat_id].get("surface_emission", [False]*SPECTRAL_BANDS))
        if is_emissive_at_any_band:
            num_tris_in_mesh = len(mesh.get('faces', []))
            for i in range(SPECTRAL_BANDS):
                if py_materials[mat_id]["surface_emission"][i]:
                    lights_per_band[i] += num_tris_in_mesh
    max_num_lights_per_band = max(lights_per_band) if lights_per_band else 0

    print("--- Scene Dimensions ---")
    print(f"Materials: {num_materials}")
    print(f"Triangles: {num_triangles}")
    print(f"Max BVH Nodes: {num_bvh_nodes}")
    print(f"Max Light Tris in any band: {max_num_lights_per_band}")
    print("------------------------")
    
    return {
        "num_materials": num_materials,
        "num_triangles": num_triangles,
        "num_bvh_nodes": num_bvh_nodes,
        "max_num_lights_per_band": max_num_lights_per_band
    }

def build_bvh(py_triangles):
    """
    Builds the BVH and returns reordered triangles and the BVH node list.
    """
    num_tris = len(py_triangles)
    if num_tris == 0:
        return [], []
        
    tri_indices = np.arange(num_tris)
    centroids = np.array([(t['v0'] + t['v1'] + t['v2']) / 3.0 for t in py_triangles])
    aabbs = np.array([[np.min([t['v0'], t['v1'], t['v2']], axis=0), np.max([t['v0'], t['v1'], t['v2']], axis=0)] for t in py_triangles])
    
    bvh_list = []
    triangles_reordered = [None] * num_tris
    reorder_idx = 0

    def get_aabb(indices):
        mins = np.min(aabbs[indices, 0, :], axis=0)
        maxs = np.max(aabbs[indices, 1, :], axis=0)
        thickness_epsilon = 1e-6
        for axis in range(3): #make sure AABBs actually have volume
            if mins[axis] == maxs[axis]:
                mins[axis] -= thickness_epsilon
                maxs[axis] += thickness_epsilon
        return mins, maxs

    def surface_area(mins, maxs):
        d = maxs - mins
        return 2 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def subdivide(indices):
        nonlocal reorder_idx
        node_idx = len(bvh_list)
        bvh_list.append({})
        aabb_min, aabb_max = get_aabb(indices)

        if len(indices) <= 10:
            bvh_list[node_idx] = {'aabb_min': aabb_min, 'aabb_max': aabb_max, 'tri_start_idx': reorder_idx, 'num_tris': len(indices)}
            for i in indices:
                triangles_reordered[reorder_idx] = py_triangles[i]
                reorder_idx += 1
            return node_idx

        best_axis, best_split_pos, best_cost = -1, -1, np.inf
        parent_sa = surface_area(aabb_min, aabb_max)
        for axis in range(3):
            sorted_indices = indices[np.argsort(centroids[indices, axis])]
            for i in range(1, len(sorted_indices)):
                left_indices, right_indices = sorted_indices[:i], sorted_indices[i:]
                left_aabb_min, left_aabb_max = get_aabb(left_indices)
                right_aabb_min, right_aabb_max = get_aabb(right_indices)
                cost = (surface_area(left_aabb_min, left_aabb_max) * len(left_indices) + surface_area(right_aabb_min, right_aabb_max) * len(right_indices)) / parent_sa
                if cost < best_cost:
                    best_cost, best_axis, best_split_pos = cost, axis, i
        
        if best_cost >= len(indices):
            bvh_list[node_idx] = {'aabb_min': aabb_min, 'aabb_max': aabb_max, 'tri_start_idx': reorder_idx, 'num_tris': len(indices)}
            for i in indices:
                triangles_reordered[reorder_idx] = py_triangles[i]
                reorder_idx += 1
            return node_idx

        sorted_indices = indices[np.argsort(centroids[indices, best_axis])]
        left_indices, right_indices = sorted_indices[:best_split_pos], sorted_indices[best_split_pos:]
        
        left_child_idx, right_child_idx = subdivide(left_indices), subdivide(right_indices)
        bvh_list[node_idx] = {'aabb_min': aabb_min, 'aabb_max': aabb_max, 'left_child': left_child_idx, 'right_child': right_child_idx, 'num_tris': 0}
        return node_idx

    subdivide(tri_indices)
    return triangles_reordered, bvh_list

def setup_scene(scene_dict):
    """
    Populates the already-placed Taichi fields with scene data.
    """
    # --- 1. Process Python scene data ---
    py_materials = scene_dict["materials"]
    py_triangles = []
    light_tris_info = []
    
    for mesh in scene_dict["meshes"]:
        verts = np.array(mesh["vertices"], dtype=np.float32)
        mat_id = mesh["material_id"]
        has_vertex_normals = 'vertex_normals' in mesh
        if has_vertex_normals:
            v_norms = np.array(mesh["vertex_normals"], dtype=np.float32)
        for face in mesh["faces"]:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            if has_vertex_normals:
                n0 = v_norms[face[0]]
                n1 = v_norms[face[1]]
                n2 = v_norms[face[2]]
            else:
                n0 = normal
                n1 = normal
                n2 = normal
            area = np.linalg.norm(normal) / 2.0
            if area > 1e-6: normal /= (area * 2.0)
            py_triangles.append({'v0': v0, 'v1': v1, 'v2': v2, 'n0': n0, 'n1': n1, 'n2': n2, 'normal':normal, 'area': area, 'mat_id': mat_id})

    # --- 2. Build BVH and get reordered triangles ---
    reordered_tris, bvh_list = build_bvh(py_triangles)
    #reordered_tris_light, bvh_list_light = build_bvh(py_triangles_light), TODO: light BVH
    
    # --- 3. Populate Taichi fields with data ---
    # Materials
    for i, mat_dict in enumerate(py_materials):
        # Provide default values for missing keys to prevent errors
        # For spectral vectors, default to a list of zeros
        zeros_vec = [0.0] * SPECTRAL_BANDS
        zeros_vec_bool = [0] * SPECTRAL_BANDS
        for j in range(SPECTRAL_BANDS):
            material_props_f32[i, PROP_REFRACTIVE_INDEX,j] = mat_dict.get('refractive_index', zeros_vec)[j]
            material_props_f32[i, PROP_EXTINCTION_COEFFICIENT,j] = mat_dict.get('extinction_coefficient', zeros_vec)[j]
            material_props_f32[i, PROP_ROUGHNESS,j] = mat_dict.get('roughness', zeros_vec)[j]
            material_props_f32[i, PROP_SCATTERING_COEFFICIENT,j] = mat_dict.get('scattering_coefficient', zeros_vec)[j]
            material_props_f32[i, PROP_EMISSION,j] = mat_dict.get('emission', zeros_vec)[j]
            material_props_f32[i, PROP_ANISTROPY_FACTOR,j] = mat_dict.get('anistropy_factor',zeros_vec)[j]

            material_props_int[i, PROP_IS_TRUE_VOLUME,j] = mat_dict.get('is_true_volume', zeros_vec_bool)[j]
            material_props_int[i, PROP_SURFACE_EMISSION,j] = mat_dict.get('surface_emission', zeros_vec_bool)[j]

        material_props_single_int[i] = mat_dict.get('eem_id', 0)
        
    # Triangles and BVH
    if len(reordered_tris) > 0:
        for i, t in enumerate(reordered_tris):
            triangles[i].v0 = t['v0']
            triangles[i].v1 = t['v1']
            triangles[i].v2 = t['v2']
            triangles[i].n0 = t['n0']
            triangles[i].n1 = t['n1']
            triangles[i].n2 = t['n2']
            triangles[i].normal = t['normal']
            triangles[i].area = t['area']
            triangles[i].material_id = t['mat_id']

    if len(bvh_list) > 0:
        for i, node in enumerate(bvh_list):
            bvh_nodes[i].aabb.min = node['aabb_min']
            bvh_nodes[i].aabb.max = node['aabb_max']
            if node.get('num_tris', 0) > 0:
                bvh_nodes[i].tri_start_idx = node['tri_start_idx']
                bvh_nodes[i].num_tris = node['num_tris']
                bvh_nodes[i].left_child = -1 # Mark as leaf
                bvh_nodes[i].right_child = -1 # Mark as leaf
            else:
                bvh_nodes[i].left_child = node['left_child']
                bvh_nodes[i].right_child = node['right_child']
                bvh_nodes[i].num_tris = 0 # Mark as internal

    # Lights    
    max_lights_in_band = scene_dims["max_num_lights_per_band"]
    for band_idx in range(SPECTRAL_BANDS):
        light_tris_for_this_band = []
        for i, t in enumerate(reordered_tris):
            mat_id = t['mat_id']
            if py_materials[mat_id]["surface_emission"][band_idx]:
                light_tris_for_this_band.append({'index': i, 'area': t['area']})

        num_lights = len(light_tris_for_this_band)
        num_lights_per_band[band_idx] = num_lights
        
        if num_lights > 0:
            total_light_area = sum(lt['area'] for lt in light_tris_for_this_band)
            total_light_area_field[band_idx] = total_light_area

            # Create numpy arrays for this band's data
            # Pad with zeros to fit the Taichi field's dimension
            np_indices = np.zeros(max_lights_in_band, dtype=np.int32)
            np_cdf = np.zeros(max_lights_in_band, dtype=np.float32)

            cdf = 0.0
            for i, lt in enumerate(light_tris_for_this_band):
                np_indices[i] = lt['index']
                cdf += lt['area'] / total_light_area
                np_cdf[i] = cdf
                light_source_tri_indices[band_idx,i] = lt['index']
                light_source_cdf[band_idx, i] = cdf
            
            # Ensure the last element is exactly 1.0 to catch all random numbers
            #if num_lights > 0:
                #np_cdf[num_lights - 1] = 1.0
            
            #light_source_tri_indices[band_idx, :].from_numpy(np_indices)
            #light_source_cdf[band_idx, :].from_numpy(np_cdf)
            
        else:
            total_light_area_field[band_idx] = 0.0

    # Camera
    cam_origin[None], cam_lookat[None], cam_up[None] = scene_dict["camera"]["origin"], scene_dict["camera"]["lookat"], scene_dict["camera"]["up"]
    update_camera()

@ti.kernel
def update_camera():
    w = tm.normalize(cam_origin[None] - cam_lookat[None])
    u = tm.normalize(tm.cross(cam_up[None], w))
    v = tm.cross(w, u)
    cam_w[None]=w
    cam_u[None]=u
    cam_v[None]=v

@ti.kernel
def create_eem(excite_wave:int, emit_wave:int, dist:float, rel_amp:float, amp:float, amp2:float):
    '''
    Taichi Kernel that generates a Excitation Emission Matrix using a Gaussian Normal Distribution

    Inputs:
        excite_wave: the excitation wavelength, nm
        emit_wave: the emission wavelength, nm
        amp: The maximum height of the curve at the center for the excitation wave
        amp2: The maximum height of the curve at the center for the emission wave
        dist: The absolute distance from the center where the curve reaches a specific drop-off height. Must be greater than 0
        rel_amp: The relative amplitude drop-off at dist. This is a multiplier of amp and must be strictly in the range (0, 1)
    '''
    
    #Excitation Emission Matrices
    eem_id = 1
    ti.loop_config(serialize=True)
    temp_eem = np.empty((SPECTRAL_BANDS,SPECTRAL_BANDS))
    for i in ti.ndrange(SPECTRAL_BANDS):
        i_wav = (WAVELENGTH_MIN + i * WAVELENGTH_STEP)
        # Excitations
        exitation = normal_dist(i_wav, excite_wave, dist, rel_amp, amp)
        total = 0.0
        for j in ti.ndrange(SPECTRAL_BANDS):
            j_wav = (WAVELENGTH_MIN + j * WAVELENGTH_STEP)
            # Emissions
            emission = normal_dist(j_wav, emit_wave, dist, rel_amp, amp2) 
            temp_eem[i, j] = emission * exitation
            total+= temp_eem[i, j]
    
    eem_object = data.EEMMatrix(
        matrix=temp_eem, 
        wavelength_min=WAVELENGTH_MIN,
        wavelength_max=WAVELENGTH_MAX,
        spectral_bands=SPECTRAL_BANDS
    )
    reemission_matrix[eem_object.id] = eem_object.matrix

# --- Utility Functions

@ti.func
def random_in_unit_sphere() -> tm.vec3:
    z = ti.random() * 2.0 - 1.0
    r_xy = ti.sqrt(tm.max(0.0, 1.0 - z * z))
    phi = ti.random() * 2.0 * tm.pi
    return tm.vec3(r_xy * tm.cos(phi), r_xy * tm.sin(phi), z)

@ti.func
def cosine_weighted_hemisphere_direction(normal: tm.vec3) -> tm.vec3:
    """
    Generates a random direction on a hemisphere with a cosine distribution.
    """
    r1 = ti.random()
    r2 = ti.random()
    phi = 2.0 * tm.pi * r1
    cos_theta_sq = 1.0 - r2
    cos_theta = ti.sqrt(cos_theta_sq)
    sin_theta = ti.sqrt(r2)
    
    up = tm.vec3(0, 1, 0) if abs(normal.y) < 0.999 else tm.vec3(1, 0, 0)
    tangent = tm.normalize(tm.cross(up, normal))
    bitangent = tm.cross(normal, tangent)
    
    return tangent * tm.cos(phi) * sin_theta + bitangent * tm.sin(phi) * sin_theta + normal * cos_theta

@ti.func
def random_in_unit_cone(direction: tm.vec3, half_angle: ti.f32) -> tm.vec3:
    # Get a random direction in the cone's local space
    phi = 2.0 * tm.pi * ti.random()
    cos_theta = 1.0 - ti.random() * (1.0 - tm.cos(half_angle))
    sin_theta = tm.sqrt(1.0 - cos_theta * cos_theta)
    
    local_dir = tm.vec3(tm.cos(phi) * sin_theta, tm.sin(phi) * sin_theta, cos_theta)
    
    # Transform back to world space
    # (using an orthonormal basis for the cone's axis)
    up = tm.vec3(0, 1, 0) if abs(direction.y) < 0.999 else tm.vec3(1, 0, 0)
    tangent = tm.normalize(tm.cross(up, direction))
    bitangent = tm.cross(direction, tangent)
    return tm.normalize(tangent * local_dir.x + bitangent * local_dir.y + direction * local_dir.z)

@ti.func
def random_rotation_in_unit_cone(direction: tm.vec3, half_angle: ti.f32) -> tm.vec3:
    # Get a random direction in the cone's local space
    phi = 2.0 * tm.pi * ti.random()
    cos_theta = tm.cos(half_angle)
    sin_theta = tm.sin(half_angle)
    
    local_dir = tm.vec3(tm.cos(phi) * sin_theta, tm.sin(phi) * sin_theta, cos_theta)
    
    # Transform back to world space
    # (using an orthonormal basis for the cone's axis)
    up = tm.vec3(0, 1, 0) if abs(direction.y) < 0.999 else tm.vec3(1, 0, 0)
    tangent = tm.normalize(tm.cross(up, direction))
    bitangent = tm.cross(direction, tangent)
    return tm.normalize(tangent * local_dir.x + bitangent * local_dir.y + direction * local_dir.z)

@ti.func
def normal_dist(x:ti.f32, center:ti.f32, dist:ti.f32, rel_amp:ti.f32, amp:ti.f32):
    '''
    Calculates a height from a coordinate in a Gaussian Normal Distribution

    Inputs:
        x: The coordinate at which to evaluate the function
        center: The coordinate of the curve's peak
        amp: The maximum height of the curve at the center 
        dist: The absolute distance from the center where the curve reaches a specific drop-off height. Must be greater than 0
        rel_amp: The relative amplitude drop-off at dist. This is a multiplier of amp and must be strictly in the range (0, 1)
    Outputs:
        The evaluated height of the Gaussian curve at coordinate x
    '''
    o_term = dist / ti.sqrt(-2 * tm.log(rel_amp))
    exponent = ti.exp( -( (x - center)**2 / (2*o_term**2) ) )
    scale = amp
    return scale * exponent

@ti.func
def evaluate_henyey_greenstein(cos_theta: ti.f32, g: ti.f32) -> ti.f32:
    """
    Evaluates the Henyey-Greenstein phase function.
    """
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    # Add epsilon to prevent division by zero with certain g and cos_theta values
    return (1.0 - g2) / (4.0 * tm.pi * denom * ti.sqrt(denom) + 1e-6)

@ti.func
def sample_henyey_greenstein(incoming_dir: tm.vec3, g: ti.f32) -> tm.vec3:
    """
    Samples a direction based on the Henyey-Greenstein phase function.
    """
    cos_theta = 0.0
    # Special case for isotropic scattering
    if abs(g) < 1e-4:
        cos_theta = 2.0 * ti.random() - 1.0
    else:
        # Inverse transform sampling
        rand_sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * ti.random())
        cos_theta = (1.0 + g * g - rand_sq * rand_sq) / (2.0 * g)

    sin_theta = ti.sqrt(tm.max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * tm.pi * ti.random()
    
    # Create an orthonormal basis around the incoming direction
    w = incoming_dir
    up = tm.vec3(0, 1, 0) if abs(w.y) < 0.999 else tm.vec3(1, 0, 0)
    u = tm.normalize(tm.cross(up, w))
    v = tm.cross(w, u)

    # Convert spherical coordinates to world direction
    return u * tm.cos(phi) * sin_theta + v * tm.sin(phi) * sin_theta + w * cos_theta


# --- Ray-Geometry Intersection

@ti.func
def intersect_aabb(position, inv_dir, aabb: AABB) -> ti.i32:
    t_min = (aabb.min - position) * inv_dir
    t_max = (aabb.max - position) * inv_dir
    t0 = tm.min(t_min, t_max)
    t1 = tm.max(t_min, t_max)
    t_enter = tm.max(t0.x, tm.max(t0.y, t0.z))
    t_exit = tm.min(t1.x, tm.min(t1.y, t1.z))
    return t_enter < t_exit and t_exit > 0.0

@ti.func
def intersect_triangle(ray_pos: tm.vec3, ray_dir: tm.vec3, tri: Triangle) -> ti.f32:
    e1 = tri.v1 - tri.v0
    e2 = tri.v2 - tri.v0
    pvec = tm.cross(ray_dir, e2)
    det = tm.dot(e1, pvec)
    t = -1.0
    u = -1.0
    v = -1.0
    if abs(det) > 1e-8:
        inv_det = 1.0 / det
        tvec = ray_pos - tri.v0
        u_temp = tm.dot(tvec, pvec) * inv_det
        if u_temp >= 0.0 and u_temp <= 1.0:
            qvec = tm.cross(tvec, e1)
            v_temp = tm.dot(ray_dir, qvec) * inv_det
            if v_temp >= 0.0 and u_temp + v_temp <= 1.0:
                t_temp = tm.dot(e2, qvec) * inv_det
                if t_temp > EPSILON:
                    t = t_temp
                    u = u_temp
                    v = v_temp
    return tm.vec3(t, u, v)

@ti.func
def trace(position: tm.vec3, direction:tm.vec3) -> HitRecord:
    hit = HitRecord(t=1e9, is_hit=0, tri_idx=-1)
    inv_dir = 1.0 / (direction)
    stack = ti.Vector([0] * 64, dt=ti.i32)
    stack_ptr = 1
    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]
        node = bvh_nodes[node_idx]
        if not intersect_aabb(position, inv_dir, node.aabb):
            continue
        if node.num_tris > 0:
            for i in range(node.num_tris):
                tri_idx = node.tri_start_idx + i
                hit_data = intersect_triangle(position, direction, triangles[tri_idx])
                t = hit_data.x
                if t > EPSILON and t < hit.t:
                    hit.is_hit = 1
                    hit.t = t
                    hit.material_id = triangles[tri_idx].material_id
                    hit.tri_idx = tri_idx
                    hit.u = hit_data.y
                    hit.v = hit_data.z
        else:
            stack[stack_ptr] = node.left_child
            stack_ptr = stack_ptr + 1
            stack[stack_ptr] = node.right_child
            stack_ptr = stack_ptr + 1
    if hit.is_hit:
        hit.normal = triangles[hit.tri_idx].normal 
        hit.shading_normal = tm.normalize((1.0 - hit.u - hit.v) * triangles[hit.tri_idx].n0 + hit.u * triangles[hit.tri_idx].n1 + hit.v * triangles[hit.tri_idx].n2)
        hit.is_front_face = ti.cast(tm.dot(direction, hit.normal) < 0.0, ti.i32)
    return hit

# --- Complex arythmetic

@ti.func
def complex_mul(a, b):
    """Multiplies two complex numbers (represented as 2D vectors)."""
    # (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
    return ti.Vector([a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]])

@ti.func
def complex_div(a, b):
    """Divides two complex numbers (a / b)."""
    # (a * b_conjugate) / |b|^2
    b_conjugate = ti.Vector([b[0], -b[1]])
    mod_sq = b[0]**2 + b[1]**2
    return complex_mul(a, b_conjugate) / mod_sq

@ti.func
def complex_mod_sq(a):
    """Calculates the modulus squared |a|^2 of a complex number."""
    return a[0]**2 + a[1]**2

@ti.func
def complex_sqrt(z):
    """Complex square root: sqrt(z)"""
    mod_z = ti.sqrt(z[0]**2 + z[1]**2)
    real_part = ti.sqrt((mod_z + z[0]) / 2.0)
    imag_part = ti.math.sign(z[1]) * ti.sqrt((mod_z - z[0]) / 2.0)
    # Handle the case where the imaginary part is zero
    if z[1] == 0:
        imag_part = 0.0
    return ti.Vector([real_part, imag_part])

# --- Ray Scattering Functions ---

@ti.func
def normaldistrobution_ggx(NdotH: ti.f32, roughness: ti.f32) -> ti.f32:
    alpha = roughness * roughness
    #alpha = tm.max(alpha, 1e-5)
    a2 = alpha * alpha
    NdotH2 = NdotH * NdotH
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    return a2 / (tm.pi * denom * denom)

@ti.func
def geometry_smith_ggx(normVec: tm.vec3, viewVec: tm.vec3, lightVec: tm.vec3, roughness: ti.f32):
    NdotV = tm.max(tm.dot(normVec, viewVec), 0.0)
    alpha = roughness * roughness
    alpha = tm.max(alpha, 1e-4)
    alpha_sq = alpha * alpha
    k_v = alpha_sq / 2.0
    g1_view = NdotV / (NdotV * (1.0 - k_v) + k_v + 1e-7)

    # Calculate G1 for the light vector (shadowing)
    NdotL = tm.max(tm.dot(normVec, lightVec), 0.0)
    k_l = alpha_sq / 2.0
    g1_light = NdotL / (NdotL * (1.0 - k_l) + k_l + 1e-7)
    
    return g1_view * g1_light

@ti.func
def fresnel_spectral(cos_theta_i: ti.f32, ni: ti.f32, ki: ti.f32, nt:ti.f32, kt:ti.f32) -> ti.f32:
    reflectance = 0.0
    cos_theta_i_clamped = tm.clamp(cos_theta_i, -1.0, 1.0)
    income_angle = tm.acos(cos_theta_i_clamped)

    # Complex IOR of Medium
    e_real = ni**2 - ki**2
    e_imag = 2 * ni * ki
    epsilon_in = ti.Vector([e_real, e_imag])
    # Complex IOR of substance
    e_real = nt**2 - kt**2
    e_imag = 2 * nt * kt
    epsilon_tran = ti.Vector([e_real, e_imag])

    # 2. Calculate angular terms
    cos_theta = ti.cos(income_angle)
    cos_theta_comp = ti.Vector([cos_theta, 0.0])
    sin_sq_theta = ti.sin(income_angle)**2
    sin_sq_theta_comp = ti.Vector([sin_sq_theta, 0.0])

    # 4. Calculate r_s (s-polarization) using complex arithmetic
    leftterm = complex_mul(complex_sqrt(epsilon_in),cos_theta_comp)
    rightterm = complex_sqrt(epsilon_tran - complex_mul(epsilon_in,sin_sq_theta_comp))
    rs_num = leftterm - rightterm
    rs_den = leftterm + rightterm
    rs = complex_div(rs_num, rs_den)

    # 5. Calculate r_p (p-polarization) using complex arithmetic
    leftterm = complex_mul(epsilon_tran, leftterm)
    rightterm = complex_mul(epsilon_in, rightterm)
    rp_num = leftterm - rightterm
    rp_den = leftterm + rightterm
    rp = complex_div(rp_num, rp_den)
    
    # 6. Calculate reflectance by averaging the modulus squared of r_s and r_p
    Rs = complex_mod_sq(rs)
    Rp = complex_mod_sq(rp)
    
    reflectance = 0.5 * (Rs + Rp)
        
    return reflectance

@ti.func
def importance_sample_ggx(xi: tm.vec2, normVec: tm.vec3, alpha: ti.f32) -> tm.vec3:
    a2 = alpha * alpha
    phi = 2.0 * tm.pi * xi.x
    cos_theta = ti.sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y))
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    h_local = tm.vec3(tm.cos(phi) * sin_theta, tm.sin(phi) * sin_theta, cos_theta)
    up = tm.vec3(0, 1, 0) if abs(normVec.y) < 0.999 else tm.vec3(1, 0, 0)
    tangent = tm.normalize(tm.cross(up, normVec))
    bitangent = tm.cross(normVec, tangent)
    return tm.normalize(tangent * h_local.x + bitangent * h_local.y + normVec * h_local.z)

@ti.func
def get_mat(mat_id:ti.i32, wavelength_idx:ti.i32) -> MaterialSample:
    mat_send = MaterialSample(
        refractive_index = material_props_f32[mat_id, PROP_REFRACTIVE_INDEX, wavelength_idx],
        extinction_coefficient = material_props_f32[mat_id, PROP_EXTINCTION_COEFFICIENT, wavelength_idx],
        roughness = material_props_f32[mat_id, PROP_ROUGHNESS, wavelength_idx],
        
        scattering_coefficient = material_props_f32[mat_id, PROP_SCATTERING_COEFFICIENT, wavelength_idx],
        emission = material_props_f32[mat_id, PROP_EMISSION, wavelength_idx],
                
        anistropy_factor = material_props_f32[mat_id, PROP_ANISTROPY_FACTOR, wavelength_idx],
        is_true_volume = material_props_int[mat_id, PROP_IS_TRUE_VOLUME, wavelength_idx],
        surface_emission = material_props_int[mat_id, PROP_SURFACE_EMISSION, wavelength_idx],
        
        eem_id = material_props_single_int[mat_id]
    )
    return mat_send

# --- Indirect Lighting ---

@ti.func
def scatter_surface(ray: Ray, vol_event:VolumeEvent) -> SurfaceEvent:
    """
    Handles Reflection of Transmission
    Physically based unified BSDF
    """
    result = SurfaceEvent(did_scatter=1)
    
    surface_material = get_mat(vol_event.hit_mat_id, ray.active_wavelength_idx)
    volume_material = get_mat(ray.vol_mat_id, ray.active_wavelength_idx)
    
    view_surface = -ray.direction
    normal_surface = vol_event.shading_normal

    refraction_index_incoming = volume_material.refractive_index
    extinction_coefficient_incoming = volume_material.extinction_coefficient
    refraction_index_surface = surface_material.refractive_index
    extinction_coefficient_surface = surface_material.extinction_coefficient
    
    halfVec = importance_sample_ggx(tm.vec2(ti.random(), ti.random()), normal_surface, surface_material.roughness)
    imcoming_angle = tm.max(1e-4,tm.dot(view_surface, halfVec))
    
    # --- Physical Characteristicss
    #Specular Prob
    reflactance = fresnel_spectral(imcoming_angle, refraction_index_incoming, extinction_coefficient_incoming, refraction_index_surface, extinction_coefficient_surface)
    
    # --- Path Selection
    rand = tm.clamp(ti.random(), 0.0, 0.99)
    if rand < reflactance:
        # --- PATH 1: SPECULAR REFLECTION (GGX) ---
        reflection_direction = tm.reflect(ray.direction, halfVec)
        if surface_material.roughness < 0.008: ##TODO, find value of roughness where this is a seemless transition
            #GGX microfacet model is numerically unstable at low roughness
            result.direction = reflection_direction
            result.did_reflect = 1
            # For a delta distribution, D, G, and PDF cancel out.
            result.estimator = reflactance 
            result.pdf = 1e9 
        else:
            NdotL = tm.max(1e-4,tm.dot(normal_surface, reflection_direction))
            NdotH = tm.max(1e-4, tm.dot(normal_surface, halfVec))
            NdotV = tm.max(1e-4, tm.dot(normal_surface, view_surface))
            
            D = normaldistrobution_ggx(NdotH, surface_material.roughness)
            G = geometry_smith_ggx(normal_surface, view_surface, reflection_direction, surface_material.roughness)

            bsdf = (D * G * reflactance) / (4.0 * NdotL * NdotV)
            pdf = (D * NdotH)/ (4.0 * imcoming_angle)

            result.direction = reflection_direction
            result.did_reflect = 1
            cos_theta = tm.max(0.0, tm.dot(normal_surface, reflection_direction))
            result.estimator = (bsdf * cos_theta)/(pdf)
            result.pdf = (pdf)

    else: 
        # --- Penetration  ---
        absorb_coeff = (4.0 * tm.pi * extinction_coefficient_surface)/(WAVELENGTH_MIN * 1e-9 + ray.active_wavelength_idx * WAVELENGTH_STEP * 1e-9)
        dist_to_event = - ti.log(-ti.random() + 1 + 1e-7)/(surface_material.scattering_coefficient + absorb_coeff + 1e-6) 
        if not surface_material.is_true_volume:
            result.did_reflect = 1 
            scatter_direction = cosine_weighted_hemisphere_direction(normal_surface)
            
            NdotL = tm.max(1e-4,tm.dot(normal_surface, scatter_direction))
            NdotV = tm.max(1e-4, tm.dot(normal_surface, view_surface))

            #TODO, random walk subsurface scattering

            # -- Single Scatter, Oren-Nayer --
            #absorb_coeff = (4.0 * tm.pi * extinction_coefficient_surface * 1e9) / (WAVELENGTH_MIN + ray.active_wavelength_idx * WAVELENGTH_STEP)
            diffuse_albedo = surface_material.scattering_coefficient / (surface_material.scattering_coefficient + absorb_coeff + 1e-6)
            sigma = surface_material.roughness * tm.pi / 2
            sigma2 = sigma * sigma
            A = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33))
            B = 0.45 * sigma2 / (sigma2 + 0.09)
            #Compute Angles
            theta_i = tm.acos(NdotL)
            theta_o = tm.acos(NdotV)
            alpha = tm.max(theta_i, theta_o)
            beta = tm.min(theta_i, theta_o)
            #Compute azimuthal difference. cos(phi_i - phi_o)
            cos_phi_diff=0.0
            if NdotL > 1e-4 and NdotV > 1e-4:
                proj_i = tm.normalize(scatter_direction - normal_surface * NdotL)
                proj_o = tm.normalize(view_surface - normal_surface * NdotV)
                cos_phi_diff = tm.max(0.0, tm.dot(proj_i, proj_o))
            
            bsdf = (1.0 - reflactance) * (diffuse_albedo / tm.pi) * (A + B * cos_phi_diff * tm.sin(alpha) * tm.tan(beta))
            pdf = 0.318309886184 * NdotL

            # Subsurface Fluorescence
            quantum_yield = 0.0
            eem_id =  surface_material.eem_id
            result.new_wavelength = ray.active_wavelength_idx
            if eem_id > 0: 
                quantum_yield_vec = SpectralVector(0.0)
                for j in ti.ndrange(SPECTRAL_BANDS):
                    quantum_yield_vec[j] = reemission_matrix[eem_id][ray.active_wavelength_idx, j]
                    if quantum_yield_vec[j] > quantum_yield:
                        quantum_yield = quantum_yield_vec[j]
                
                rand3 = ti.random()
                if quantum_yield > 0.0 and rand3 < quantum_yield:
                    normalizer = 0.0
                    for j in ti.ndrange(SPECTRAL_BANDS):
                        normalizer += reemission_matrix[eem_id][ray.active_wavelength_idx, j]
                    
                    rand4 = ti.random()
                    cumulative_prob = 0.0
                    strength = 0.0
                    new_wavelength = ray.active_wavelength_idx
                    for j in ti.ndrange(SPECTRAL_BANDS):
                        strength = reemission_matrix[eem_id][ray.active_wavelength_idx, j]
                        cumulative_prob += strength/normalizer
                        if rand4 < cumulative_prob:
                            new_wavelength = j
                            break
                    result.new_wavelength = new_wavelength
                    surface_material_fluorescence = get_mat(vol_event.hit_mat_id, new_wavelength)
                    absorb_coeff = (4.0 * tm.pi * surface_material_fluorescence.extinction_coefficient)/(WAVELENGTH_MIN * 1e-9 + new_wavelength * WAVELENGTH_STEP * 1e-9)
                    fluorescence_transmittance = ti.exp(-absorb_coeff * dist_to_event)
                    bsdf *= fluorescence_transmittance #Switch to absorbed re-emitted light
            
            result.estimator = (bsdf * NdotL)/(pdf)
            result.pdf = (pdf)
            result.direction = scatter_direction

        else:
            # -- Transmission --
            ior_ratio = refraction_index_incoming / refraction_index_surface
            refraction_direction = tm.refract(ray.direction, halfVec, ior_ratio)
            if refraction_direction.norm_sqr() < 1e-4:
                # Total Internal Reflection
                reflection_direction = tm.reflect(ray.direction, halfVec)
                if surface_material.roughness < 0.008: ##TODO, find value of roughness where this is a seemless transition
                    #GGX microfacet model is numerically unstable at low roughness
                    result.direction = reflection_direction
                    result.did_reflect = 1
                    # For a delta distribution, D, G, and PDF cancel out.
                    result.estimator = reflactance 
                    result.pdf = 1e9 
                else:
                    NdotL = tm.max(1e-4,tm.dot(normal_surface, reflection_direction))
                    NdotH = tm.max(1e-4, tm.dot(normal_surface, halfVec))
                    NdotV = tm.max(1e-4, tm.dot(normal_surface, view_surface))
                    
                    D = normaldistrobution_ggx(NdotH, surface_material.roughness)
                    G = geometry_smith_ggx(normal_surface, view_surface, reflection_direction, surface_material.roughness)

                    bsdf = (D * G * reflactance) / (4.0 * NdotL * NdotV)
                    pdf = (D * NdotH)/ (4.0 * imcoming_angle) 

                    result.direction = reflection_direction
                    result.did_reflect = 1
                    cos_theta = ti.abs(tm.dot(normal_surface, reflection_direction))
                    result.estimator = (bsdf * cos_theta)/(pdf)
                    result.pdf = (pdf)    
            else: 
                # Refraction
                result.did_reflect = 0
                result.direction = refraction_direction
                
                if surface_material.roughness < 0.0018:
                    pdf = 1e9
                    result.estimator = (1.0 - reflactance)
                    result.pdf = 1e9
                else:
                    LdotH = tm.dot(refraction_direction, halfVec)
                    VdotH = tm.dot(view_surface, halfVec)
                    NdotH = tm.max(1e-4, tm.dot(normal_surface, halfVec))
                    NdotL = tm.max(1e-4, ti.abs(tm.dot(normal_surface, refraction_direction)))
                    NdotV = tm.max(1e-4, ti.abs(tm.dot(normal_surface, view_surface)))

                    G = geometry_smith_ggx(normal_surface, view_surface, refraction_direction, surface_material.roughness)
                    D = normaldistrobution_ggx(NdotH, surface_material.roughness)
                    
                    denom = (refraction_index_incoming * VdotH + refraction_index_surface * LdotH)
                    denom_sq = denom * denom
                    
                    bsdf_term1 = (ti.abs(LdotH * VdotH)) / (NdotL * NdotV)
                    bsdf_term2 = (refraction_index_surface**2 * (1.0 - reflactance) * D * G) / denom_sq
                    bsdf = bsdf_term1 * bsdf_term2
                    
                    pdf = (D * NdotH * (1.0 - reflactance) * refraction_index_surface**2 * ti.abs(LdotH)) / denom_sq
                    
                    result.estimator = (bsdf * NdotL) / (pdf)
                    result.pdf = pdf

    return result

@ti.func
def scatter_volume(ray: Ray) -> VolumeEvent:
    """
    Handles Volumetric Effects
    """
    # --- Fetch ---
    result = VolumeEvent()
    vol_mat = get_mat(ray.vol_mat_id, ray.active_wavelength_idx)
    result.new_wavelength_idx = ray.active_wavelength_idx
    result.direction = ray.direction
    result.penetrated_tri_idx = ray.penetrated_tri_idx

    # --- Distance to Volume Event  ---
    absorption_coefficient = (4.0 * tm.pi * vol_mat.extinction_coefficient)/(WAVELENGTH_MIN * 1e-9 + ray.active_wavelength_idx * WAVELENGTH_STEP * 1e-9)
    distance_to_volume_event = - ti.log(-ti.random() + 1 - 1e-7)/(vol_mat.scattering_coefficient + absorption_coefficient + 1e-6) 

    #--- Distance to Surface Event ---
    hit = trace(ray.position, ray.direction)
    distance_to_surface = hit.t

    #--- Determine Closest Event ---
    travel_distance = -1.0
    if distance_to_surface < ESCAPE:
        travel_distance = distance_to_surface
        result.penetrated_tri_idx = hit.tri_idx
        if distance_to_volume_event + EPSILON < travel_distance:
            travel_distance = distance_to_volume_event
    #elif distance_to_volume_event < ESCAPE:
        #travel_distance = distance_to_volume_event

    result.travel_distance=travel_distance
    result.is_front_face = hit.is_front_face
    result.normal = hit.normal * ti.cast(hit.is_front_face, ti.f32) - hit.normal * (1 - ti.cast(hit.is_front_face,ti.f32))
    result.shading_normal = hit.shading_normal * ti.cast(hit.is_front_face, ti.f32) - hit.shading_normal * (1 - ti.cast(hit.is_front_face,ti.f32))

    # --- Attenuation ---
    transmittance  = ti.exp(-(absorption_coefficient+vol_mat.scattering_coefficient) * travel_distance)
    emittance = ((-transmittance + 1)*vol_mat.emission)/(absorption_coefficient + 1e-6)
    attenuation = transmittance + emittance

    # TODO: optimize this selection structure, or just make it looke better 
    # --- Determine Event ---
    if travel_distance == distance_to_surface:
        # -- Surface Hit --
        bsdf = attenuation
        pdf = 1.0
        result.estimator = bsdf/(pdf + 1e-6)
        result.hit_action = 1
        result.hit_mat_id = hit.material_id
        result.tri_idx = hit.tri_idx
    elif travel_distance >= 0.0:
        # -- True Volume Path -- 
        result.hit_mat_id = -1
        result.tri_idx = -1
        rand_event = ti.random()
        scatter_type = absorption_coefficient/(vol_mat.scattering_coefficient + absorption_coefficient + 1e-6)
        if rand_event <= scatter_type:
            # -- Fluorescence --
            quantum_yield = 0.0
            eem_id =  vol_mat.eem_id
            if eem_id > 0: 
                quantum_yield_vec = SpectralVector(0.0)
                for j in ti.ndrange(SPECTRAL_BANDS):
                    quantum_yield_vec[j] = reemission_matrix[eem_id][ray.active_wavelength_idx, j]
                    if quantum_yield_vec[j] > quantum_yield:
                        quantum_yield = quantum_yield_vec[j]

                rand_fluoresce = ti.random()
                if quantum_yield > 0.0 and rand_fluoresce < quantum_yield:
                    result.hit_action = 3
                    normalizer = 0.0
                    for j in ti.ndrange(SPECTRAL_BANDS):
                        normalizer += reemission_matrix[eem_id][ray.active_wavelength_idx, j]
                    
                    rand2 = ti.random()
                    cumulative_prob = 0.0
                    strength = 0.0
                    new_active_wavelength_idx = ray.active_wavelength_idx
                    for j in ti.ndrange(SPECTRAL_BANDS):
                        strength = reemission_matrix[eem_id][ray.active_wavelength_idx, j]
                        cumulative_prob += strength/normalizer
                        if rand2 < cumulative_prob:
                            new_active_wavelength_idx = j
                            break
                    #Fluorescence is generally isotropic
                    result.new_wavelength_idx = new_active_wavelength_idx
                    result.direction = random_in_unit_sphere()
                    emittance = ((-(1.0 - transmittance) + 1)*get_mat(ray.vol_mat_id,new_active_wavelength_idx).emission)/(absorption_coefficient + 1e-6)
                    result.estimator = (1.0 - transmittance) + emittance
                else:
                    # -- Failed Fluorescence --
                    result.estimator = attenuation
            else:
                pdf = 1.0
                bsdf = attenuation
                result.estimator = bsdf/(pdf + 1e-6)
        else:
            # -- Scattering --
            result.hit_action = 2
            result.direction = sample_henyey_greenstein(ray.direction, vol_mat.anistropy_factor)
            result.estimator = attenuation

    return result

# --- Direct Lighting

@ti.func
def sample_direct_light(ray:Ray, normVec: tm.vec3, surf_mat_id: ti.i32) -> NEE_Sample:
    result = NEE_Sample()

    p = ray.position
    vol_mat_id = ray.vol_mat_id
    viewVec = -ray.direction
    wavelength_idx = ray.active_wavelength_idx
    is_reflected = ray.is_reflected

    num_lights_for_band = num_lights_per_band[wavelength_idx]
    if num_lights_for_band > 0:
        rand = ti.random()
        light_tri_idx = 0 
        for i in ti.ndrange(num_lights_for_band):
            if rand < light_source_cdf[wavelength_idx, i]:
                light_tri_idx = light_source_tri_indices[wavelength_idx, i]
                break

        light_tri = triangles[light_tri_idx]

        # Sample a random point on that light's surface
        rand_bary = tm.vec2(ti.random(), ti.random())
        if rand_bary.x + rand_bary.y > 1.0: 
            rand_bary = 1.0 - rand_bary
        point_on_light = light_tri.v0 + rand_bary.x * (light_tri.v1 - light_tri.v0) + rand_bary.y * (light_tri.v2 - light_tri.v0)
        # Probability of choosing point on light: 1 / Triangle Area

        direction_to_light = tm.normalize(point_on_light - p)
        distance_to_light = tm.length(point_on_light - p)

        #ShadowTraansmittanceInput=(origin=, direction_to_light=,distance_to_light=,vol_mat_id=,wavelength_idx=)
        transmittance = shadow_transmittance(p, direction_to_light, distance_to_light, light_tri_idx, vol_mat_id, wavelength_idx)
        if transmittance > 1e-6: 
            light_mat = get_mat(light_tri.material_id, wavelength_idx)
            light_normal = light_tri.normal

            cos_theta_light = tm.max(0.0, tm.dot(light_normal, -direction_to_light)) # Visibility
            distance_sq = distance_to_light * distance_to_light # light spread
            geometry_term = cos_theta_light / (distance_sq + 1e-4)

            # BSDF (or Phase Function) Evaluation
            factor = ti.cast(0.0, ti.f32)
            bsdf = 0.0 # default to Volume Phase Function
            cos_theta = 1.0
            pdf = 1.0
            if surf_mat_id != -1: # We are on a surface
                #EvaluateBSDFInput = (vol_mat_id=, surf_mat_id=, viewVec=, direction_to_light=,normVec=,wavelength_ids=)
                cos_theta = tm.max(0.0, tm.dot(normVec, direction_to_light)) # surface angle
                bsdf, pdf = evaluate_bsdf(vol_mat_id, surf_mat_id, viewVec, direction_to_light, normVec, wavelength_idx, is_reflected)
            else:
                cos_theta = tm.max(0.0, tm.dot(ray.direction, direction_to_light))
                bsdf = evaluate_henyey_greenstein(cos_theta, get_mat(vol_mat_id, wavelength_idx).anistropy_factor)
                pdf = bsdf

            factor = bsdf * cos_theta * transmittance * geometry_term * total_light_area_field[wavelength_idx]
            result.contribution_factor = factor
            result.emission = get_mat(light_tri.material_id, ray.transport_wavelength_idx).emission
            
            pdf_light = (distance_sq) / (tm.max(1e-6, cos_theta_light) * total_light_area_field[wavelength_idx])
            result.weight = (pdf_light * pdf_light) / (pdf_light * pdf_light + pdf * pdf + 1e-6)

    return result

@ti.func
def evaluate_bsdf(vol_mat_id: ti.i32, surf_mat_id:ti.i32, viewVec: tm.vec3, lightVec: tm.vec3, normVec: tm.vec3, wavelength_idx:ti.i32, is_reflected):
    vol_mat = get_mat(vol_mat_id, wavelength_idx)
    surf_mat = get_mat(surf_mat_id, wavelength_idx)

    n_i = vol_mat.refractive_index
    k_i = vol_mat.extinction_coefficient
    n_t = surf_mat.refractive_index
    k_t = surf_mat.extinction_coefficient
    
    NdotL = tm.max(1e-4,tm.dot(normVec, lightVec))
    NdotV = tm.max(1e-4, tm.dot(normVec, viewVec))
    
    is_reflected = ti.cast(NdotV * NdotL > 0.0, ti.i32)
    
    bsdf = 0.0
    pdf = 0.0
    
    if is_reflected:
        halfVec = tm.normalize(viewVec + lightVec)
        if tm.dot(halfVec, normVec) < 0.0: halfVec = -halfVec
        VdotH = tm.max(1e-4,tm.dot(viewVec, halfVec))
        F = fresnel_spectral(VdotH, n_i, k_i, n_t, k_t)
        
        if surf_mat.roughness > 0.008:
            # Specular Term
            NdotH = tm.max(1e-4, tm.dot(normVec, halfVec))
            
            D = normaldistrobution_ggx(NdotH, surf_mat.roughness)
            G = geometry_smith_ggx(normVec, viewVec, lightVec, surf_mat.roughness)

            bsdf += (D * G * F) / (4.0 * NdotL * NdotV)
            pdf += (D * NdotH * F)/ (4.0 * VdotH)
        
        absorb_coeff = (4.0 * tm.pi * k_t * 1e9) / (WAVELENGTH_MIN + wavelength_idx * WAVELENGTH_STEP)
        diffuse_albedo = surf_mat.scattering_coefficient / (surf_mat.scattering_coefficient + absorb_coeff + 1e-6)
        sigma = surf_mat.roughness * tm.pi / 2
        sigma2 = sigma * sigma
        A = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33))
        B = 0.45 * sigma2 / (sigma2 + 0.09)
        #Compute Angles
        theta_i = tm.acos(NdotL)
        theta_o = tm.acos(NdotV)
        alpha = tm.max(theta_i, theta_o)
        beta = tm.min(theta_i, theta_o)
        #Compute azimuthal difference. cos(phi_i - phi_o)
        cos_phi_diff=0.0
        if NdotL > 1e-4 and NdotV > 1e-4:
            proj_i = tm.normalize(lightVec - normVec * NdotL)
            proj_o = tm.normalize(viewVec - normVec * NdotV)
            cos_phi_diff = tm.max(0.0, tm.dot(proj_i, proj_o))
        
        bsdf += (1.0 - F) * (1.0 - surf_mat.is_true_volume) * (diffuse_albedo / tm.pi) * (A + B * cos_phi_diff * tm.sin(alpha) * tm.tan(beta))
        pdf += NdotL * 0.318309886184 * (1.0 - F) * (1.0 - surf_mat.is_true_volume)
    else:
        # Transmission
        # Generalized halfway vector for refraction
        if surf_mat.roughness > 0.008:
            halfVec = -tm.normalize(viewVec * n_i + lightVec * n_t)
            if tm.dot(halfVec, normVec) < 0.0: halfVec = -halfVec
            
            VdotH = tm.dot(viewVec, halfVec)
            LdotH = tm.dot(lightVec, halfVec)
            NdotH = tm.max(1e-4, tm.dot(normVec, halfVec))
            
            # Only valid if light and view are on correct sides of the microfacet
            if VdotH * LdotH < 0.0: 
                F = fresnel_spectral(ti.abs(VdotH), n_i, k_i, n_t, k_t)
                D = normaldistrobution_ggx(NdotH, surf_mat.roughness)
                G = geometry_smith_ggx(normVec, viewVec, lightVec, surf_mat.roughness)
                
                # The Jacobian denominator
                denom = (n_i * VdotH + n_t * LdotH)
                denom_sq = denom * denom
                
                # Transmission BSDF
                bsdf_term1 = (ti.abs(LdotH * VdotH)) / (ti.abs(NdotL * NdotV))
                bsdf_term2 = (n_t * n_t * (1.0 - F) * D * G) / denom_sq
                bsdf += bsdf_term1 * bsdf_term2
                
                # Transmission PDF (incorporating the Jacobian)
                pdf += (D * NdotH * (1.0 - F) * n_t * n_t * ti.abs(LdotH)) / denom_sq
    
    #EvaluateBSDFOutput = (bsdf=,pdf=)
    return bsdf, pdf

@ti.func
def shadow_transmittance(origin: tm.vec3, direction: tm.vec3, max_dist: ti.f32, tri_idx:ti.i32, start_vol_mat_id: ti.i32, wavelength_idx: ti.i32):

    total_transmittance = 1.0
    distance_traveled = 0.0
    current_pos = origin
    current_vol_id = start_vol_mat_id

    shadow_ray = ShadowRay(position=current_pos, direction=direction)

    for i in ti.ndrange(SCATTER_BOUNCES):
        hit = trace(shadow_ray.position, shadow_ray.direction)
        dist_to_surface = hit.t
        
        # Volume Interaction
        vol_mat = get_mat(current_vol_id, wavelength_idx)
        absorb_coeff = (4.0 * tm.pi * vol_mat.extinction_coefficient) / (WAVELENGTH_MIN * 1e-9 + wavelength_idx * WAVELENGTH_STEP * 1e-9)
        total_transmittance *= ti.exp(-(absorb_coeff+vol_mat.scattering_coefficient) * (hit.t))

        # Hit Light
        if hit.t >= (max_dist - distance_traveled) or hit.tri_idx==tri_idx:
            vol_mat = get_mat(current_vol_id, wavelength_idx)
            total_transmittance *= ti.exp(-(absorb_coeff + vol_mat.scattering_coefficient) * (max_dist - distance_traveled))
            break
        
        # Surface Interaction
        surface_mat = get_mat(hit.material_id, wavelength_idx)
        n_i = vol_mat.refractive_index
        k_i = vol_mat.extinction_coefficient
        n_t = surface_mat.refractive_index
        k_t = surface_mat.extinction_coefficient 
        if not hit.is_front_face:
            n_t = get_mat(0, wavelength_idx).refractive_index  # Air
            k_t = get_mat(0, wavelength_idx).extinction_coefficient
        incident_normal = hit.shading_normal * hit.is_front_face - hit.shading_normal * (1.0 - hit.is_front_face)
        
        ior_ratio = n_i / (n_t)
        refracted_direction = tm.refract(shadow_ray.direction, incident_normal, ior_ratio)
                
        refract_term = ((tm.dot(shadow_ray.direction, refracted_direction)) - 1)*37.9743 # (dot-1)/(1-cos(.23 rad))
        angle_factor = 1/(1 + refract_term**4 * (3-2*refract_term))
        total_transmittance *= angle_factor # Attentuate ray by refraction
        
        if total_transmittance < 1e-6 or not surface_mat.is_true_volume:
            total_transmittance = 0.0
            break

        # Update
        #shadow_ray.direction = tm.normalize(refracted_direction) #ray should point to light
        distance_traveled += dist_to_surface + EPSILON
        shadow_ray.position = current_pos + shadow_ray.direction * (dist_to_surface + EPSILON)
        new_vol_id = hit.material_id * hit.is_front_face # 0 if backface,
        current_vol_id = new_vol_id
    return total_transmittance

# --- Render Kernels ---
@ti.kernel
def cast_rays(current_sample_in_block: int, block_index:int):
    # Camera setup
    for i, j in pixels_geometry:
        u = (i + ti.random()) / RENDER_WIDTH
        v = (j + ti.random()) / RENDER_HEIGHT

        active_wavelength_idx[i,j] = current_sample_in_block
        transport_wavelength_idx[i,j] = current_sample_in_block
        
        ray_direction = tm.normalize(
            cam_u[None] * (u * 2.0 - 1.0) * ASPECT_RATIO * 1.0 +
            cam_v[None] * (v * 2.0 - 1.0) * 1.0 -
            cam_w[None] * 2.0
        )

        ray = Ray(
            position=cam_origin[None], 
            direction=ray_direction, 
            active_wavelength_idx=active_wavelength_idx[i,j], 
            transport_wavelength_idx=transport_wavelength_idx[i,j],
            vol_mat_id = 0,
            throughput = 1.0,
            path_length = 0,
            has_terminated = 0,
            has_escaped = 0,
            surface_pdf = 1.0,
            )
        #path_contribution = SpectralVector(0.0)
        #direct_path_contribution = 0.0
        #indirect_path_contribution = 0.0

        for k in ti.ndrange(MAX_PATH_LENGTH):
            if ray.has_terminated:
                break
            direct_path_contribution = 0.0
            indirect_path_contribution = 0.0
            NEE_sample = NEE_Sample()
            event = scatter_volume(ray)
            if event.travel_distance < 0.0:
                ray.has_escaped = 1
                break
            ray.position += ray.direction * event.travel_distance  
            ray.throughput *= event.estimator  
            ray.penetrated_tri_idx = event.penetrated_tri_idx

            if event.hit_action == 1:
                surface_material =  get_mat(event.hit_mat_id, ray.transport_wavelength_idx)
                if surface_material.surface_emission:
                    #-- Direct Hit with Emissive
                    light_tri = triangles[event.tri_idx]
                    distance_sq = event.travel_distance * event.travel_distance
                    cos_theta_light = tm.max(0.0, tm.dot(light_tri.normal, -ray.direction))
                    
                    pdf_light = (distance_sq * tm.min(1.0, ray.path_length)) / (cos_theta_light * total_light_area_field[ray.active_wavelength_idx] + 1e-6)
                    weight =  ((ray.surface_pdf * ray.surface_pdf) / (pdf_light * pdf_light + ray.surface_pdf * ray.surface_pdf + 1e-6))
                    indirect_path_contribution = ray.throughput * surface_material.emission * weight
                    ray.has_terminated = 1
                    #break
                else:
                    NEE_sample = sample_direct_light(ray, event.shading_normal, event.hit_mat_id)
                    direct_path_contribution = ray.throughput * NEE_sample.emission * NEE_sample.contribution_factor * NEE_sample.weight

                    ray.path_length += 1
                    #-- Hit Surface
                    surf_event = scatter_surface(ray, event)

                    ray.surface_pdf = surf_event.pdf
                    ray.direction = surf_event.direction
                    ray.throughput *= surf_event.estimator
                    ray.position += event.normal * EPSILON * (surf_event.did_reflect - (1.0-surf_event.did_reflect))
                    ray.vol_mat_id = event.hit_mat_id * event.is_front_face * (1 - surf_event.did_reflect) + ray.vol_mat_id * surf_event.did_reflect # 0 (air) on backface
                    ray.is_reflected = surf_event.did_reflect

            else:
                NEE_sample = sample_direct_light(ray, event.shading_normal, event.hit_mat_id)
                direct_path_contribution = ray.throughput * NEE_sample.emission * NEE_sample.contribution_factor * NEE_sample.weight
                if event.hit_action==2:
                    ray.direction = event.direction
                    ray.path_length += 1
                #Volume Event, Fluoresce
                elif event.hit_action==3:
                    ray.direction = event.direction
                    ray.active_wavelength_idx = event.new_wavelength_idx
                    active_wavelength_idx[i,j] = ray.active_wavelength_idx
                    ray.path_length += 1

            block_accumulator[i, j,active_wavelength_idx[i,j]] += indirect_path_contribution
            block_accumulator[i,j,active_wavelength_idx[i,j]] += direct_path_contribution

@ti.kernel
def average_and_reset_block(num_blocks_completed: int):
    for i, j,k in spectral_accumulator:
        # Get the previous average from the main accumulator
        previous_avg = spectral_accumulator[i, j, k]
        
        # Get the newly completed block's total contribution
        block_contribution = block_accumulator[i, j,k]
        
        # Perform a running average on the blocks
        # This gives equal weight to every completed block
        new_avg = (previous_avg * ti.cast(num_blocks_completed - 1, ti.f32) + block_contribution) / ti.cast(num_blocks_completed, ti.f32)
        
        # Store the new average and reset the block accumulator for the next run
        spectral_accumulator[i, j,k] = new_avg
        block_accumulator[i, j,k] = 0.0

@ti.kernel
def color_rays():
    for i,j in pixels:
        # Convert to final pixel color

        render_i = ti.cast(i / IMAGE_WIDTH * RENDER_WIDTH, ti.i32)
        render_j = ti.cast(j / IMAGE_HEIGHT * RENDER_HEIGHT, ti.i32)

        xyz = tm.vec3(0.0)
        for k in ti.ndrange(SPECTRAL_BANDS):
            xyz += spectral_accumulator[render_i, render_j, k] * cmf_data[k]
        xyz *= WAVELENGTH_STEP / normalization_factor[None]

        hdr_rgb = xyz_to_rgb_matrix[None] @ xyz
        
        exposed_color = EXPOSURE * hdr_rgb
        mapped_color = exposed_color / (exposed_color + 1.0)
        ldr_rgb = tm.pow(mapped_color, 1.0 / GAMMA)
        
        #ldr_rgb = post_process(hdr_rgb)
        pixels[i, j] = ldr_rgb

@ti.kernel
def visualize_normals():
    for i, j in pixels_geometry:
        u = (i + 0.5) / RENDER_WIDTH
        v = (j + 0.5) / RENDER_HEIGHT

        ray_direction = tm.normalize(
            cam_u[None] * (u * 2.0 - 1.0) * ASPECT_RATIO * 1.0 +
            cam_v[None] * (v * 2.0 - 1.0) * 1.0 -
            cam_w[None] * 2.0
        )
        ray = GeoRay(
            position=cam_origin[None], 
            direction=ray_direction, 
            BVH_depth=0,
            depth_to_hit=-1
            )

        hit = trace(ray.position, ray.direction)
        if hit.is_hit:
            color = hit.shading_normal * 0.5 + 0.5
            pixels_geometry[i, j] = color
        else:
            pixels_geometry[i, j] = tm.vec3(0.0) # Black for no hit
    
    for i, j in pixels:  
        render_i = ti.cast(i / IMAGE_WIDTH * RENDER_WIDTH, ti.i32)
        render_j = ti.cast(j / IMAGE_HEIGHT * RENDER_HEIGHT, ti.i32)
        pixels[i, j] = pixels_geometry[render_i, render_j]

@ti.kernel
def clear_pixel_space():
    """
    Resets all pixel-space accumulators and the final pixel buffer to zero.
    """
    for i, j in pixels:
        pixels[i, j] = tm.vec3(0.0)

    for i, j, k in spectral_accumulator:
        spectral_accumulator[i, j,k] = 0.0
        block_accumulator[i, j,k] = 0.0
    
    for i, j in pixels_geometry:
        pixels_geometry[i, j] = tm.vec3(0.0)

# --- Main Execution ---
if __name__ == "__main__":

    # --- Procedural Structures and Fields ---
    py_scene = create_scene()
    scene_dims = get_scene_dimensions(py_scene)    

    # Consolidated field for all f32 spectral properties
    reemission_matrix[fluorescent_glass.id] = fluorescent_glass.matrix
    material_props_f32 = ti.field(dtype=ti.f32, shape=(
        scene_dims["num_materials"],
        NUM_F32_PROPERTIES,
        SPECTRAL_BANDS
    ))

    # Consolidated field for all integer properties
    material_props_int = ti.field(dtype=ti.i32, shape=(
        scene_dims["num_materials"],
        NUM_INT_PROPERTIES,
        SPECTRAL_BANDS
    ))
    material_props_single_int = ti.field(dtype=ti.i32, shape=(
        scene_dims["num_materials"],
    ))

    triangles = Triangle.field(shape=scene_dims["num_triangles"])
    bvh_nodes = BVHNode.field(shape=scene_dims["num_bvh_nodes"])
    max_lights = scene_dims["max_num_lights_per_band"]
    if max_lights > 0:
        light_source_tri_indices = ti.field(ti.i32, shape=(SPECTRAL_BANDS, max_lights))
        light_source_cdf = ti.field(ti.f32, shape=(SPECTRAL_BANDS, max_lights))
    else: # Handle case with no lights to avoid Taichi errors
        light_source_tri_indices = ti.field(ti.i32, shape=(SPECTRAL_BANDS, 1))
        light_source_cdf = ti.field(ti.f32, shape=(SPECTRAL_BANDS, 1))

    print("Setting Scene...")
    setup_scene(py_scene)

    gui = ti.GUI("Hyperspectral Path Tracer", res=(IMAGE_WIDTH, IMAGE_HEIGHT), fast_gui=True)

    render_mode = 'render'  # Modes: 'render', 'geometry'
    total_samples = 0
    num_blocks_completed = 0
    rendering_finished = False
    start_time = time.time()

    clear_pixel_space()
    print("Starting interactive renderer...")
    print("Press 's' to switch between Render and Geometry modes.")
    print("Press 'c' to simulate camera movement (resets render).")
    print("Press ESC to exit.")

    cam_pos = np.array(py_scene["camera"]["origin"], dtype=np.float32)
    # Start facing down the -Z axis (standard for your setup)
    cam_yaw = -np.pi / 2.0  
    cam_pitch = 0.0
    move_speed = 0.05
    rot_speed = 0.03

    while gui.running:
        # --- Event Handling ---
        # Check for key presses to change state
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            
            camera_changed = False
        
            # Pull current forward (-w) and right (u) vectors from Taichi to guide WASD translation
            # We flatten the Y component so we don't fly up/down when looking up/down (FPS style)
            forward = -cam_w[None].to_numpy()
            forward[1] = 0.0 
            if np.linalg.norm(forward) > 1e-6:
                forward = forward / np.linalg.norm(forward)
                
            right = cam_u[None].to_numpy()
            right[1] = 0.0
            if np.linalg.norm(right) > 1e-6:
                right = right / np.linalg.norm(right)

            if gui.is_pressed('w'):
                cam_pos += forward * move_speed
                camera_changed = True
            if gui.is_pressed('s'):
                cam_pos -= forward * move_speed
                camera_changed = True
            if gui.is_pressed('a'):
                cam_pos -= right * move_speed
                camera_changed = True
            if gui.is_pressed('d'):
                cam_pos += right * move_speed
                camera_changed = True
            if gui.is_pressed('e'):
                cam_pos[1] += move_speed
                camera_changed = True
            if gui.is_pressed('q'):
                cam_pos[1] -= move_speed
                camera_changed = True
            
            if gui.is_pressed(ti.GUI.LEFT):
                cam_yaw -= rot_speed
                camera_changed = True
            if gui.is_pressed(ti.GUI.RIGHT):
                cam_yaw += rot_speed
                camera_changed = True
            if gui.is_pressed(ti.GUI.UP):
                cam_pitch += rot_speed
                camera_changed = True
            if gui.is_pressed(ti.GUI.DOWN):
                cam_pitch -= rot_speed
                camera_changed = True
            
            if e.key == 'r':  # Switch mode
                if render_mode == 'render':
                    render_mode = 'geometry'
                    print("Switched to Geometry Mode (Normals)")
                else:
                    render_mode = 'render'
                    print("Switched to Path Tracing Mode")
                
                # Reset state for the new mode
                clear_pixel_space()
                total_samples = 0
                num_blocks_completed = 0
                rendering_finished = False
                start_time = time.time()
            
            cam_pitch = max(-np.pi/2.0 + 0.01, min(np.pi/2.0 - 0.01, cam_pitch))
            if camera_changed:
                # Calculate the new look direction based on spherical coordinates
                dir_x = np.cos(cam_yaw) * np.cos(cam_pitch)
                dir_y = np.sin(cam_pitch)
                dir_z = np.sin(cam_yaw) * np.cos(cam_pitch)
                look_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float32)

                # Update the Taichi fields
                cam_origin[None] = cam_pos
                cam_lookat[None] = cam_pos + look_dir
                
                # Recalculate camera basis vectors (u, v, w) in Taichi
                update_camera()

                # Flush rendering state so the image doesn't smear
                clear_pixel_space()
                total_samples = 0
                num_blocks_completed = 0
                rendering_finished = False
                start_time = time.time()

        # --- Rendering Logic ---
        if render_mode == 'render':
            if not rendering_finished:
                if total_samples < SAMPLES_PER_PIXEL:
                    total_samples += 1
                    current_sample_in_block = (total_samples - 1) % SPECTRAL_BANDS
                    cast_rays(current_sample_in_block, num_blocks_completed)

                    # A block is one full spectral pass
                    if total_samples % SPECTRAL_BANDS == 0:
                        num_blocks_completed += 1
                        average_and_reset_block(num_blocks_completed)
                        color_rays() 
                else:
                    print(f"Finished rendering {SAMPLES_PER_PIXEL} samples in {time.time() - start_time:.2f} seconds.")
                    rendering_finished = True 
        elif render_mode == 'geometry':
            visualize_normals()

        # --- Display Update ---
        gui.set_image(pixels)
        gui.show()