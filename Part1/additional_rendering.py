import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy, numpy as np, argparse, os, json, random, glob
from mathutils import Vector

# ---------- Categories ----------
CATEGORIES = {
    "tweezers": {
        "id": 2, "name": "tweezers",
        "kp_order": ["handle_end","left_arm","left_tip","right_arm","right_tip"],
        "skeleton": [[0,1],[0,3],[1,2],[3,4]],
        "swap_pairs": [[1,3],[2,4]],
        "front_axis": "Z",
    },
    "needle_holder": {
        "id": 1, "name": "needle_holder",
        "kp_order": ["joint","left_handle","left_tip","right_handle","right_tip"],
        "skeleton": [[0,1],[0,2],[0,3],[0,4]],
        "swap_pairs": [[1,3],[2,4]],
        "front_axis": "Z",
    }
}

# ---------- Occluder controls ----------
OCCLUDER_MIN, OCCLUDER_MAX = 2, 5
OCCLUDER_SIZE_RANGE = (0.15, 0.60)    # relative to tool bbox diagonal
OCCLUDER_DIST_SCALE = 0.7
OCCLUDER_TYPES = ("CUBE", "PLANE")

# ---------- Args ----------
p = argparse.ArgumentParser()
p.add_argument('--dataset_type', default="train")
p.add_argument('--tools_root', default="tools_blend")
p.add_argument('--camera_params', default="camera.json")
p.add_argument('--output_dir', default="data")
p.add_argument('--num_frames_per_tool', type=int, default=2)
p.add_argument('--radius_min', type=float, default=3.0)
p.add_argument('--radius_max', type=float, default=12.0)
p.add_argument('--target_bbox_frac', type=float, default=0.4)
p.add_argument('--max_tries', type=int, default=10000)
p.add_argument('--front_angle_deg', type=float, default=85.0)
args = p.parse_args()

bproc.init()

# ---------- Output paths ----------
split_dir = os.path.join(args.output_dir, args.dataset_type)
images_dir = os.path.join(split_dir, "images")
os.makedirs(split_dir, exist_ok=True)
kp_json_path = os.path.join(split_dir, "coco_keypoints.json")

# ---------- Camera intrinsics ----------
fx, fy, cx, cy, W, H = (lambda d: (d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"]))(
    json.load(open(args.camera_params))
)
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]], float)
CameraUtility.set_intrinsics_from_K_matrix(K, W, H)

# ---------- Small utils ----------
def project_world(pts_xyz, cam2world, K):
    M = np.linalg.inv(cam2world)
    Pc = (M @ np.c_[pts_xyz, np.ones(len(pts_xyz))].T).T[:, :3]
    X, Y, Z = Pc[:,0], Pc[:,1], Pc[:,2]
    Zc = -Z; valid = Zc > 1e-6
    Zs = np.where(valid, Zc, 1.0)
    x = K[0,0]*(X/Zs) + K[0,2]
    y = -K[1,1]*(Y/Zs) + K[1,2]
    return np.stack([x,y], 1), valid

def clamp_box(x0,y0,x1,y1,w,h):
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    x0c, y0c = np.clip([x0,y0], [0,0], [w-1,h-1])
    x1c, y1c = np.clip([x1,y1], [0,0], [w-1,h-1])
    bw, bh = max(1.0, x1c-x0c+1), max(1.0, y1c-y0c+1)
    return float(x0c), float(y0c), float(bw), float(bh)

def in_image(x,y,w,h): return 0.0 <= x <= w-1 and 0.0 <= y <= h-1
def count_pngs(folder): return sum(f.lower().endswith(".png") for f in os.listdir(folder)) if os.path.isdir(folder) else 0
def set_world_bg(strength):
    node = bpy.data.worlds["World"].node_tree.nodes.get("Background")
    if node: node.inputs[1].default_value = float(strength)

def mesh_aabb_volume(obj):
    try:
        bb = np.array(obj.get_bound_box()); ext = bb.max(0)-bb.min(0)
        return float(ext[0]*ext[1]*ext[2])
    except: return -1.0

def axis_vec(s):
    s = s.strip().upper(); sign = -1.0 if s.startswith('-') else 1.0; a = s.replace('-','')
    return sign * np.array({'X':[1,0,0],'Y':[0,1,0]}.get(a,[0,0,1]), float)

def front_dir_world(tool_obj, axis_str):
    R = tool_obj.get_local2world_mat()[:3,:3]
    v = axis_vec(axis_str)
    u = R @ v
    return u / (np.linalg.norm(u)+1e-9)

def cos_front(cam2world, tool_obj, front_axis):
    cam, ctr = cam2world[:3,3], tool_obj.get_location()
    tool2cam = cam - ctr; tool2cam /= (np.linalg.norm(tool2cam)+1e-9)
    return float(np.dot(front_dir_world(tool_obj, front_axis), tool2cam))

def ray_visible(cam_loc, pt_world, allowed_names, max_offset=1e-3, eps=1e-4):
    cam_loc = np.asarray(cam_loc).ravel(); pt_world = np.asarray(pt_world).ravel()
    direction = Vector((pt_world - cam_loc).tolist()); dist = direction.length
    if dist <= 1e-6: return True
    direction.normalize(); origin = Vector(cam_loc.tolist()) + direction * max_offset
    hit, hit_loc, _, _, hit_obj, _ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(), origin, direction, distance=float(dist+1e-3)
    )
    if not hit: return True
    if (hit_loc - origin).length + eps >= dist: return True
    try:
        name = (hit_obj.original.name if hit_obj.original else hit_obj.name)
    except: name = hit_obj.name
    return (name in allowed_names)

def add_occluders(tool_obj, bbox_world):
    """
    Create occluders around the tool.
    Returns a list of dicts: [{"name": <obj_name>, "type": <"CUBE"/"PLANE">}, ...]
    """
    entries, bb = [], np.array(bbox_world)
    ctr, diag = tool_obj.get_location(), float(np.linalg.norm(bb.max(0)-bb.min(0)))
    for _ in range(random.randint(OCCLUDER_MIN, OCCLUDER_MAX)):
        t = random.choice(OCCLUDER_TYPES)
        size = random.uniform(*OCCLUDER_SIZE_RANGE) * max(0.05, diag)
        loc = (ctr + np.random.uniform(-OCCLUDER_DIST_SCALE*diag, OCCLUDER_DIST_SCALE*diag, 3)).tolist()
        if t == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc); obj = bpy.context.active_object
            obj.scale = (size, size, size*random.uniform(0.4,1.2))
        else:  # "PLANE"
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc); obj = bpy.context.active_object
            obj.scale = (size*random.uniform(0.7,1.4), size*random.uniform(0.7,1.4), 1.0)
            obj.rotation_euler = tuple(random.uniform(0, np.pi) for _ in range(3))
        if not obj.data.materials:
            m = bpy.data.materials.new("OccMat"); m.use_nodes = True
            bsdf = m.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (random.random(),random.random(),random.random(),1.0)
                bsdf.inputs["Roughness"].default_value = random.uniform(0.2,0.9)
            obj.data.materials.append(m)
        entries.append({"name": obj.name, "type": t})
    return entries

def thin_occluder_planes(created_occluders, drop_frac=0.5, min_keep=1):
    """
    Delete a random subset of PLANE occluders.
    Mutates `created_occluders` in place to remove deleted entries.
    Returns how many planes were deleted.
    """
    planes = [e for e in created_occluders if isinstance(e, dict) and e.get("type") == "PLANE"]
    if not planes:
        return 0
    can_drop = max(0, len(planes) - int(min_keep))
    k = min(can_drop, max(0, int(round(len(planes) * float(drop_frac)))))
    if k <= 0:
        return 0
    to_drop = random.sample(planes, k)
    for e in to_drop:
        name = e["name"]
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
        # remove from our bookkeeping list
        created_occluders.remove(e)
    return k


def _resolve_bpy_obj(x):
    """Return bpy.types.Object from a BlenderProc object or a bpy object."""
    if hasattr(x, "blender_obj") and x.blender_obj is not None:
        return x.blender_obj
    if hasattr(x, "get_name"):
        return bpy.data.objects.get(x.get_name())
    return x if isinstance(x, bpy.types.Object) else None

def _get_principled_bsdf(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if not bsdf:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (0, 0)
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if not out:
            out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (240, 0)
        nt.links.new(bsdf.outputs.get("BSDF"), out.inputs.get("Surface"))
    return bsdf

def _broadcast_value(sock, value):
    """Assign value to a socket, auto-broadcasting scalars to vec3/vec4 when needed."""
    try:
        # simple case
        sock.default_value = value
        return
    except Exception:
        pass
    # Try broadcasting scalar -> vector
    try:
        cur = sock.default_value  # has a length (e.g., (r,g,b) or (r,g,b,a))
        n = len(cur)
        if isinstance(value, (int, float)):
            if n == 3:
                sock.default_value = (float(value),) * 3
            elif n == 4:
                # keep alpha from existing if present
                sock.default_value = (float(value), float(value), float(value), cur[3] if n==4 else 1.0)
        else:
            # last resort: if tuple length mismatches, slice/pad
            if isinstance(value, (tuple, list)):
                vv = list(value)[:n] + [cur[i] for i in range(len(value), n)]
                sock.default_value = type(cur)(vv)
    except Exception:
        # ignore if we still can't set; some sockets are link-only
        pass

def _set_input(bsdf, labels, value):
    """
    Set a Principled input by trying multiple label aliases.
    labels: str or sequence of str (aliases).
    """
    if isinstance(labels, str):
        labels = [labels]
    for lbl in labels:
        s = bsdf.inputs.get(lbl)
        if s is not None and hasattr(s, "default_value"):
            _broadcast_value(s, value)
            return  # success

def _get_or_create_node(nt, ntype, name):
    node = next((n for n in nt.nodes if n.type == ntype and n.label == name), None)
    if node:
        return node
    node = nt.nodes.new(ntype)
    node.label = name
    return node

def apply_tool_metal_material(obj_like):
    """
    Apply a brushed/clearcoated metal to a MeshObject (BlenderProc) or bpy object.
    Creates/reuses the first material slot and keeps all slots consistent.
    """
    bo = _resolve_bpy_obj(obj_like)
    if bo is None or not hasattr(bo, "data"):
        return

    # Create/reuse first material
    if bo.data.materials and bo.data.materials[0] is not None:
        mat = bo.data.materials[0]
    else:
        mat = bpy.data.materials.new(name="ToolMetal")
        if len(bo.data.materials) == 0:
            bo.data.materials.append(mat)
        else:
            bo.data.materials[0] = mat

    bsdf = _get_principled_bsdf(mat)

    # Stainless-ish base
    _set_input(bsdf, "Base Color", (0.62, 0.62, 0.62, 1.0))
    _set_input(bsdf, "Metallic", random.uniform(0.85, 1.0))
    _set_input(bsdf, ["Roughness", "Specular Roughness"], random.uniform(0.06, 0.22))

    # Specular control: some builds expose "Specular IOR Level" instead of "Specular"
    _set_input(bsdf, ["Specular IOR Level", "Specular"], 0.5)
    _set_input(bsdf, "Specular Tint", random.uniform(0.0, 0.25))  # will auto-broadcast if vector

    # Brushed look
    _set_input(bsdf, "Anisotropic", random.uniform(0.25, 0.75))
    _set_input(bsdf, "Anisotropic Rotation", random.uniform(0.0, 1.0))

    # Thin clearcoat (alias across Blender versions)
    _set_input(bsdf, ["Coat Weight", "Clearcoat"], random.uniform(0.05, 0.25))
    _set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], random.uniform(0.03, 0.15))

    _set_input(bsdf, "IOR", 1.45)

    # Subtle normal detail; reuse nodes if already present
    nt = mat.node_tree
    noise = _get_or_create_node(nt, "ShaderNodeTexNoise", "TM_Noise")
    noise.location = (-600, -100)
    noise.inputs["Scale"].default_value = random.uniform(800.0, 2000.0)
    noise.inputs["Detail"].default_value = 2.0

    bump = _get_or_create_node(nt, "ShaderNodeBump", "TM_Bump")
    bump.location = (-300, -80)
    bump.inputs["Strength"].default_value = random.uniform(0.02, 0.06)

    # Ensure links (avoid duplicate links)
    def _link(o, oi, t, ti):
        if not any(l.from_node is o and l.to_node is t and l.from_socket == o.outputs[oi] and l.to_socket == t.inputs[ti]
                   for l in nt.links):
            nt.links.new(o.outputs[oi], t.inputs[ti])

    if bsdf.inputs.get("Normal"):
        _link(noise, "Fac", bump, "Height")
        _link(bump, "Normal", bsdf, "Normal")

    # Keep all material slots consistent
    for i in range(len(bo.data.materials)):
        bo.data.materials[i] = mat
        
# quick KP vis flags for a candidate pose (pre-accept test) ---
def kp_flags_for_pose(kp_world, cam2world, K, w, h, mesh_name_set):
    """Return (v_flags, pts2d, valid) for all KPs for a given pose."""
    pts2d, valid = project_world(kp_world, cam2world, K)
    cam_loc = cam2world[:3,3]
    v_flags = []
    for i, ok in enumerate(valid):
        if not ok:
            v_flags.append(0); continue
        x, y = float(pts2d[i,0]), float(pts2d[i,1])
        if not in_image(x, y, w, h):
            v_flags.append(0); continue
        is_vis = ray_visible(cam_loc, kp_world[i], mesh_name_set)
        v_flags.append(2 if is_vis else 1)
    return v_flags, pts2d, valid

def inject_inline_occluder(cam2world, kp_world_point, approx_size):
    """
    Place a small plane along the ray from camera to the keypoint to force occlusion.
    approx_size controls plane size in world units.
    Returns the created object's name.
    """
    cam = cam2world[:3,3]
    frac = random.uniform(0.3, 0.8)
    p = cam + frac * (kp_world_point - cam)
    
    bpy.ops.mesh.primitive_plane_add(size=float(approx_size), location=p.tolist())
    obj = bpy.context.active_object
    obj.rotation_euler = tuple(random.uniform(0, np.pi) for _ in range(3))

    # --- Add dark blood-like material (brown, red, black) ---
    if not obj.data.materials:
        m = bpy.data.materials.new("OccInline")
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            dark_colors = [
                (0.05, 0.01, 0.01, 1.0),   # deep blood red
                (0.1, 0.05, 0.02, 1.0),    # dark brown
                (0.02, 0.02, 0.02, 1.0)    # near black
            ]
            color = random.choice(dark_colors)
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = random.uniform(0.5, 0.9)
            # bsdf.inputs["Specular"].default_value = 0.2
        obj.data.materials.append(m)

    obj.name = f"OccInline_{random.randint(0,1_000_000)}"
    return obj.name


# ---------- COCO accumulators ----------
if os.path.exists(kp_json_path):
    kp_data = json.load(open(kp_json_path))
    images, annotations = kp_data.get("images", []), kp_data.get("annotations", [])
    categories = kp_data.get("categories", [])
    next_ann_id = max((an["id"] for an in annotations), default=-1) + 1
else:
    categories = [{
        "id": spec["id"], "name": spec["name"], "supercategory": "tool",
        "keypoints": spec["kp_order"], "skeleton": spec["skeleton"]
    } for spec in CATEGORIES.values()]
    images, annotations, next_ann_id = [], [], 0

# ---------- Gather blend files ----------
blend_jobs = [(cls, p) for cls in CATEGORIES for p in sorted(glob.glob(os.path.join(args.tools_root, cls, "*.blend")))]

# ---------- Render loop ----------
bproc.renderer.set_max_amount_of_samples(96)
bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.enable_segmentation_output(map_by=["category_id","instance","name"], default_values={"category_id": None})

img_diag = (W*W + H*H)**0.5
target_diag = args.target_bbox_frac * img_diag
cos_thresh = float(np.cos(np.deg2rad(args.front_angle_deg)))

for cls_name, blend_path in blend_jobs:
    if cls_name == "tweezers":
        continue
    spec = CATEGORIES[cls_name]
    kp_names = spec["kp_order"]
    swap_pairs = spec.get("swap_pairs", [])
    front_axis = spec.get("front_axis","Y")
    cat_id = int(spec["id"])

    print(f"\n=== {cls_name} :: {os.path.basename(blend_path)} ===")
    bproc.utility.reset_keyframes()
    loaded = bproc.loader.load_blend(blend_path)
    name2obj = {o.get_name(): o for o in loaded}

    kp_objs = [name2obj[k] for k in kp_names if k in name2obj]
    if len(kp_objs) != len(kp_names):
        print(f"[WARN] Missing keypoint empties in {blend_path}, skipping.")
        bproc.object.delete_multiple(loaded, remove_all_offspring=True)
        continue

    mesh_objs = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
    if not mesh_objs:
        candidates = [o for o in loaded if o.get_name() not in set(kp_names) and o.get_name() != "Camera"]
        candidates.sort(key=mesh_aabb_volume, reverse=True)
        if not candidates or mesh_aabb_volume(candidates[0]) <= 0:
            print(f"[WARN] No valid tool mesh in {blend_path}, skipping.")
            bproc.object.delete_multiple(loaded, remove_all_offspring=True)
            continue
        tool = candidates[0]
    else:
        mesh_objs.sort(key=mesh_aabb_volume, reverse=True); tool = mesh_objs[0]

    tool_mesh_names = {m.get_name() for m in mesh_objs}
    for m in mesh_objs: 
        m.set_cp("category_id", cat_id)
        apply_tool_metal_material(m)

    
    # Simple light
    L = bproc.types.Light(); L.set_type("POINT")
    L.set_energy(random.uniform(150, 800))
    L.set_location(tool.get_location() + np.array([0.5, 0.5, 0.8]))

    # Precompute KP world + bbox
    kp_world = np.array([(o.get_local2world_mat() @ np.array([0,0,0,1.]))[:3] for o in kp_objs])
    M_tool = tool.get_local2world_mat()
    bbox_world = (M_tool @ np.c_[tool.get_bound_box(), np.ones(8)].T).T[:, :3]
    # handy extent-based scale for rescue plane
    bbox_extent = np.ptp(bbox_world, axis=0)  # (ex,ey,ez)
    approx_plane_size = 0.1 * float(np.linalg.norm(bbox_extent))  # small, relative

    # Add occluders (once per tool)
    created_occluders = add_occluders(tool, bbox_world)

    frames_target = max(1, args.num_frames_per_tool)
    cam_poses, tries, poses = [], 0, 0
    radius_min_cur = args.radius_min

    while tries < args.max_tries and poses < frames_target:
        set_world_bg(np.random.uniform(0.2, 1.2))
        r = np.random.uniform(radius_min_cur, args.radius_max)
        center = tool.get_location()
        loc = bproc.sampler.shell(center=center, radius_min=r, radius_max=r, elevation_min=-65, elevation_max=75)
        look_at = center + np.random.uniform(-0.2,0.2,3)
        R = bproc.camera.rotation_from_forward_vec(look_at - loc, inplane_rot=np.random.uniform(-0.6,0.6))
        cam2world = bproc.math.build_transformation_mat(loc, R)

        # tool must be visible & front-ish
        if tool not in bproc.camera.visible_objects(cam2world): tries += 1; continue
        if cos_front(cam2world, tool, front_axis) < cos_thresh: tries += 1; continue

        # avoid fully-occluded center
        if not ray_visible(cam2world[:3,3], center, tool_mesh_names): tries += 1; continue

        # size control via projected bbox
        pts2d_bb, valid_bb = project_world(bbox_world, cam2world, K)
        vb = pts2d_bb[valid_bb]
        if vb.shape[0] < 2: tries += 1; continue
        x0,y0 = vb.min(0); x1,y1 = vb.max(0)
        diag = float(np.hypot(x1-x0, y1-y0))

        if diag > 1.20*target_diag:
            radius_min_cur = min(args.radius_max, radius_min_cur*1.08+0.05); tries += 1; continue
        if diag < 0.15*target_diag:
            radius_min_cur = max(1.5, radius_min_cur*0.92); tries += 1; continue

        # --- NEW: require at least one occluded KP and one visible KP ---
        v_flags_test, _, _ = kp_flags_for_pose(kp_world, cam2world, K, W, H, tool_mesh_names)
        has_visible = any(v == 2 for v in v_flags_test)
        has_occluded = any(v == 1 for v in v_flags_test)

        # NEW: if nothing is visible, thin planes once and re-check
        if not has_visible:
            dropped = thin_occluder_planes(created_occluders, drop_frac=0.5, min_keep=1)
            if dropped > 0:
                # re-evaluate with fewer planes
                v_flags_test, _, _ = kp_flags_for_pose(kp_world, cam2world, K, W, H, tool_mesh_names)
                has_visible = any(v == 2 for v in v_flags_test)
                has_occluded = any(v == 1 for v in v_flags_test)
            if not has_visible:
                tries += 1
                continue

        rescue_name = None
        if has_visible and not has_occluded:
            # Optional rescue: try to occlude a random currently-visible KP
            vis_idxs = [i for i,v in enumerate(v_flags_test) if v == 2]
            if vis_idxs:
                i = random.choice(vis_idxs)
                rescue_name = inject_inline_occluder(cam2world, kp_world[i], approx_size=approx_plane_size)
                v_flags2, _, _ = kp_flags_for_pose(kp_world, cam2world, K, W, H, tool_mesh_names)
                has_visible, has_occluded = any(v==2 for v in v_flags2), any(v==1 for v in v_flags2)
                if not (has_visible and has_occluded):
                    if rescue_name in bpy.data.objects:
                        bpy.data.objects.remove(bpy.data.objects[rescue_name], do_unlink=True)
                    rescue_name = None
                    tries += 1
                    continue
            else:
                tries += 1
                continue


        # Accept pose
        # Store rescue_name (if any) so we can sweep later just in case
        bproc.camera.add_camera_pose(cam2world, frame=poses)
        cam_poses.append((cam2world.copy(), rescue_name) if rescue_name else cam2world.copy())
        poses += 1; tries += 1

    data = bproc.renderer.render()

    bproc.writer.write_coco_annotations(
        output_dir=split_dir,
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        mask_encoding_format="rle",
        append_to_existing_output=True
    )

    frames_rendered = len(data["colors"])
    total_after = count_pngs(images_dir)
    start_idx = total_after - frames_rendered
    print(f"[COCO-KP] frames={frames_rendered}, images_after={total_after}, start_idx={start_idx}")

    tool_bpy = bpy.data.objects.get(tool.get_name())
    for frame_idx, cam_pose in enumerate(cam_poses):
        # support both tuple (with rescue) and plain matrix
        if isinstance(cam_pose, tuple):
            cam2world, _rescue = cam_pose
        else:
            cam2world = cam_pose

        h, w = H, W
        pts2d, valid = project_world(kp_world, cam2world, K)

        vis_pts = pts2d[valid]
        if vis_pts.size:
            x0,y0 = vis_pts.min(0); x1,y1 = vis_pts.max(0)
        else:
            pts2d_bb, valid_bb = project_world(bbox_world, cam2world, K)
            vb = pts2d_bb[valid_bb]
            if not vb.size: continue
            x0,y0 = vb.min(0); x1,y1 = vb.max(0)

        x0c, y0c, bw, bh = clamp_box(x0,y0,x1,y1,w,h)

        cam_loc = cam2world[:3,3]
        v_flags = []
        for i, ok in enumerate(valid):
            if not ok: v_flags.append(0); continue
            x, y = map(float, pts2d[i])
            if not in_image(x,y,w,h): v_flags.append(0); continue
            v_flags.append(2 if ray_visible(cam_loc, kp_world[i], tool_mesh_names) else 1)

        # back-view swap (symmetry)
        if cos_front(cam2world, tool, front_axis) < 0.0:
            pts2d_sw, v_sw = pts2d.copy(), v_flags.copy()
            for a,b in swap_pairs:
                pts2d_sw[a], pts2d_sw[b] = pts2d_sw[b].copy(), pts2d_sw[a].copy()
                v_sw[a], v_sw[b] = v_sw[b], v_sw[a]
            pts2d, v_flags = pts2d_sw, v_sw

        keypoints = [coord for i in range(len(kp_names))
                     for coord in (float(pts2d[i,0]), float(pts2d[i,1]), int(v_flags[i]))]

        img_id = start_idx + frame_idx
        if not any(im.get("id")==img_id for im in images):
            images.append({"id": img_id, "width": w, "height": h, "file_name": f"images/{img_id:06d}.png"})

        annotations.append({
            "id": next_ann_id, "image_id": img_id, "category_id": cat_id,
            "iscrowd": 0, "area": int(round(bw*bh)),
            "bbox": [int(round(x0c)), int(round(y0c)), int(round(bw)), int(round(bh))],
            "num_keypoints": len(kp_names), "keypoints": keypoints
        })
        next_ann_id += 1

    json.dump({"images": images, "annotations": annotations, "categories": categories},
              open(kp_json_path, "w"))

    # cleanup
    for e in created_occluders:
        n = e["name"] if isinstance(e, dict) else e
        if n in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
    # sweep any leftover OccInline planes (defensive)
    for obj in list(bpy.data.objects):
        if obj.name.startswith("OccInline"):
            bpy.data.objects.remove(obj, do_unlink=True)

    bproc.object.delete_multiple(loaded, remove_all_offspring=True)