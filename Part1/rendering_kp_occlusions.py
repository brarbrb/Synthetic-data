import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
from mathutils import Vector
import argparse, os, json, random, glob

# in this file we render - non depth aware and with occlusions!???

# -------- Categories, KP order, symmetry, skeletons --------
# IMPORTANT:
# - swap_pairs are *index* pairs into kp_order that must swap together when viewing the back side
# - front_axis defines the local axis that points "out of" the front face of your tool
CATEGORIES = {
    "tweezers": {
        "id": 2,
        "name": "tweezers",
        "kp_order": ["handle_end", "left_arm", "left_tip", "right_arm", "right_tip"],
        "skeleton": [[0,1],[0,3],[1,2],[3,4]],
        "swap_pairs": [[1,3],[2,4]],
        "front_axis": "Z",   
    },
    "needle_holder": {
        "id": 1,
        "name": "needle_holder",
        "kp_order": ["joint", "left_handle", "left_tip", "right_handle", "right_tip"],
        "skeleton": [[0,1],[0,2],[0,3],[0,4]],
        "swap_pairs": [[1,3],[2,4]],
        "front_axis": "Z",
    }
}

# -------- Occluder controls --------
OCCLUDER_MIN = 2         # min occluders per tool
OCCLUDER_MAX = 5         # max occluders per tool
OCCLUDER_SIZE_RANGE = (0.15, 0.60)   # relative to tool bbox diagonal
OCCLUDER_DIST_SCALE = 0.9  # occluders placed ~within this many tool-diagonals from center
OCCLUDER_TYPES = ("CUBE", "PLANE")   # choose from: CUBE / PLANE / CYLINDER (cylinder via ops too)

# -------- Args --------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', default="val", help="train/val/test")
parser.add_argument('--tools_root', default="tools", help="Folder containing class subfolders with .blend files")
parser.add_argument('--camera_params', default="camera.json", help="Intrinsics JSON")
parser.add_argument('--output_dir', default="out", help="Output root")
parser.add_argument('--num_frames_per_tool', type=int, default=5, help="Desired frames per tool (min 1 enforced)")
parser.add_argument('--radius_min', type=float, default=3.0, help="Min camera radius (meters)")
parser.add_argument('--radius_max', type=float, default=12.0, help="Max camera radius (meters)")
parser.add_argument('--target_bbox_frac', type=float, default=0.33, help="Target bbox diag as fraction of image diag")
parser.add_argument('--max_tries', type=int, default=8000, help="Pose sampling guard")
parser.add_argument('--front_angle_deg', type=float, default=85.0, help="Max angle from front to accept (<= this = OK; > this = treat as back)")

args = parser.parse_args()

bproc.init()

# -------- Output paths --------
split_dir = os.path.join(args.output_dir, args.dataset_type)
images_dir = os.path.join(split_dir, "images")
os.makedirs(split_dir, exist_ok=True)
kp_json_path = os.path.join(split_dir, "coco_keypoints.json")

# -------- Camera intrinsics --------
with open(args.camera_params, "r") as f:
    cam = json.load(f)
fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
W, H = cam["width"], cam["height"]
K = np.array([[fx, 0,  cx],
              [0,  fy, cy],
              [0,  0,  1]], dtype=float)
CameraUtility.set_intrinsics_from_K_matrix(K, W, H)

# -------- Helpers --------

def project_world(points_xyz, cam2world, K):
    """Return (N,2) pixel coords and boolean validity (in front of camera)."""
    world2cam = np.linalg.inv(cam2world)
    Pw = np.concatenate([points_xyz, np.ones((len(points_xyz),1))], axis=1)  # (N,4)
    Pc = (world2cam @ Pw.T).T[:, :3]
    X, Y, Z = Pc[:,0], Pc[:,1], Pc[:,2]
    Zc = -Z
    valid = Zc > 1e-6
    Zs = np.where(valid, Zc, 1.0)
    x = K[0,0]*(X/Zs) + K[0,2]
    y = -K[1,1] * (Y / Zs) + K[1,2]
    return np.stack([x,y], axis=1), valid

def clamp_box(x0,y0,x1,y1,w,h):
    x0c = float(np.clip(x0, 0, w-1)); y0c = float(np.clip(y0, 0, h-1))
    x1c = float(np.clip(x1, 0, w-1)); y1c = float(np.clip(y1, 0, h-1))
    bw = max(1.0, x1c - x0c + 1.0)
    bh = max(1.0, y1c - y0c + 1.0)
    return x0c, y0c, bw, bh

def in_image(x, y, w, h):
    return (0.0 <= x <= w-1) and (0.0 <= y <= h-1)

def count_pngs(folder):
    try:
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(".png"))
    except FileNotFoundError:
        return 0

def ensure_world_bg_strength(val):
    bg = bpy.data.worlds["World"].node_tree.nodes.get("Background", None)
    if bg is not None:
        bg.inputs[1].default_value = float(val)

def mesh_aabb_volume(obj):
    try:
        bb = np.array(obj.get_bound_box())
        ext = bb.max(axis=0) - bb.min(axis=0)
        return float(ext[0]*ext[1]*ext[2])
    except Exception:
        return -1.0

def axis_to_vec(axis_str):
    s = axis_str.strip().upper()
    sign = -1.0 if s.startswith('-') else 1.0
    a = s.replace('-', '')
    if a == 'X': v = np.array([1,0,0], dtype=float)
    elif a == 'Y': v = np.array([0,1,0], dtype=float)
    else: v = np.array([0,0,1], dtype=float)
    return sign * v

def tool_front_dir_world(tool_obj, axis_str):
    """Return world-space direction vector corresponding to the tool's 'front_axis'."""
    M = tool_obj.get_local2world_mat()
    # Take rotation part (first 3 columns of 3x3) applied to a unit local axis
    local = axis_to_vec(axis_str)
    R = M[:3,:3]
    d = (R @ local.reshape(3,1)).flatten()
    n = d / (np.linalg.norm(d) + 1e-9)
    return n

def is_back_view(cam2world, tool_obj, front_axis="Y"):
    """True if the camera is in the tool's back hemisphere (angle > 90° from front)."""
    cam_loc = cam2world[:3, 3]
    tool_center = tool_obj.get_location()
    tool_to_cam = cam_loc - tool_center                 # <-- tool -> camera
    tool_to_cam = tool_to_cam / (np.linalg.norm(tool_to_cam) + 1e-9)
    front_world = tool_front_dir_world(tool_obj, front_axis)
    return np.dot(front_world, tool_to_cam) < 0.0       # cos(theta) < 0 => back

def accept_front_view(cam2world, tool_obj, front_axis, front_angle_deg):
    cam_loc = cam2world[:3, 3]
    tool_center = tool_obj.get_location()
    tool_to_cam = cam_loc - tool_center
    tool_to_cam = tool_to_cam / (np.linalg.norm(tool_to_cam) + 1e-9)
    front_world = tool_front_dir_world(tool_obj, front_axis)
    cosang = float(np.dot(front_world, tool_to_cam))
    return cosang >= np.cos(np.deg2rad(front_angle_deg))

def ray_visible_to_point(cam_loc, pt_world, allowed_names: set[str],
                         max_offset=1e-3, eps=1e-4) -> bool:
    """
    Visible iff there is no hit closer than the point, or the closest hit is the tool itself.
    We DO NOT require hitting the tool; free-space up to the keypoint is also visible.
    """
    cam_loc = np.asarray(cam_loc).reshape(3,)
    pt_world = np.asarray(pt_world).reshape(3,)

    direction = Vector((pt_world - cam_loc).tolist())
    dist_to_pt = direction.length
    if dist_to_pt <= 1e-6:
        return True  # point at camera → treat as visible

    direction.normalize()
    origin = Vector(cam_loc.tolist()) + direction * max_offset

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Cast up to slightly PAST the point to avoid “just-on-surface” misses
    cast_dist = float(dist_to_pt + 1e-3)

    hit, hit_loc, hit_norm, hit_face_idx, hit_obj, _ = scene.ray_cast(
        depsgraph,
        origin,
        direction,
        distance=cast_dist
    )

    if not hit:
        # No geometry before/at the point → nothing occluding
        return True

    # Distance from origin to first hit
    hit_distance = (hit_loc - origin).length

    # If first hit is *behind* the point (numerically), not occluded
    if hit_distance + eps >= dist_to_pt:
        return True

    # Otherwise something is in front; visible only if that something is the tool
    try:
        original = hit_obj.original
        hit_name = (original.name if original else hit_obj.name)
    except Exception:
        hit_name = hit_obj.name

    return (hit_name in allowed_names)


def add_random_occluders_around(tool_obj, bbox_world_pts, n_min=OCCLUDER_MIN, n_max=OCCLUDER_MAX):
    """
    Create simple mesh occluders (cubes/planes) around the tool.
    They do NOT get 'category_id' so the COCO writer will ignore them,
    but they will occlude in the render and ray_cast.
    Returns list of created bpy object names for cleanup.
    """
    created = []
    bb = np.array(bbox_world_pts)
    center = tool_obj.get_location()
    diag = float(np.linalg.norm(bb.max(axis=0) - bb.min(axis=0)))

    n = random.randint(n_min, n_max)
    for _ in range(n):
        t = random.choice(OCCLUDER_TYPES)
        size = random.uniform(*OCCLUDER_SIZE_RANGE) * max(0.05, diag)
        offset = np.random.uniform(-OCCLUDER_DIST_SCALE*diag, OCCLUDER_DIST_SCALE*diag, size=3)
        loc = center + offset

        if t == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc.tolist())
            obj = bpy.context.active_object
            obj.scale = (size, size, size * random.uniform(0.4, 1.2))
        elif t == "PLANE":
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc.tolist())
            obj = bpy.context.active_object
            # random anisotropic plane size
            obj.scale = (size * random.uniform(0.7,1.4), size * random.uniform(0.7,1.4), 1.0)
            obj.rotation_euler = (
                random.uniform(0, np.pi),
                random.uniform(0, np.pi),
                random.uniform(0, np.pi)
            )
        else:
            # extend if you want cylinders etc.
            continue

        # Give a simple diffuse material (optional)
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="OccMat")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (random.random(), random.random(), random.random(), 1.0)
                bsdf.inputs["Roughness"].default_value = random.uniform(0.2, 0.9)
            obj.data.materials.append(mat)

        created.append(obj.name)

    return created

# -------- COCO accumulator (append if exists) --------
if os.path.exists(kp_json_path):
    with open(kp_json_path, "r") as f:
        kp_data = json.load(f)
    images = kp_data.get("images", [])
    annotations = kp_data.get("annotations", [])
    categories = kp_data.get("categories", [])
    next_ann_id = (max((an["id"] for an in annotations), default=-1) + 1)
else:
    categories = []
    for cls_name, spec in CATEGORIES.items():
        categories.append({
            "id": spec["id"],
            "name": spec["name"],
            "supercategory": "tool",
            "keypoints": list(spec["kp_order"]),
            "skeleton": list(spec["skeleton"])
        })
    images, annotations, next_ann_id = [], [], 0

# -------- Gather blend files --------
blend_jobs = []
for cls_name in CATEGORIES.keys():
    cls_dir = os.path.join(args.tools_root, cls_name)
    for path in sorted(glob.glob(os.path.join(cls_dir, "*.blend"))):
        blend_jobs.append((cls_name, path))

# -------- Main loop over tools --------
for cls_name, blend_path in blend_jobs:
    spec = CATEGORIES[cls_name]
    kp_names = spec["kp_order"]
    swap_pairs = spec.get("swap_pairs", [])
    front_axis = spec.get("front_axis", "Y")
    cat_id = int(spec["id"])

    print(f"\n=== {cls_name} :: {os.path.basename(blend_path)} ===")
    bproc.utility.reset_keyframes()

    # Load and upcast to concrete types
    loaded = bproc.loader.load_blend(blend_path)

    # Map by name
    name2obj = {o.get_name(): o for o in loaded}

    # Collect KP empties by exact names
    kp_objs, missing = [], []
    for kpn in kp_names:
        if kpn in name2obj:
            kp_objs.append(name2obj[kpn])
        else:
            missing.append(kpn)
    if missing:
        print(f"[WARN] Missing keypoint empties in {blend_path}: {missing}. Skipping.")
        bproc.object.delete_multiple(loaded, remove_all_offspring=True)
        continue

    # Choose tool mesh
    mesh_objects = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
    if not mesh_objects:
        candidates = [o for o in loaded if o.get_name() not in set(kp_names) and o.get_name() != "Camera"]
        candidates.sort(key=mesh_aabb_volume, reverse=True)
        if not candidates or mesh_aabb_volume(candidates[0]) <= 0:
            print(f"[WARN] No valid tool mesh-like object in {blend_path}, skipping.")
            bproc.object.delete_multiple(loaded, remove_all_offspring=True)
            continue
        tool = candidates[0]
    else:
        mesh_objects.sort(key=mesh_aabb_volume, reverse=True)
        tool = mesh_objects[0]

    tool_mesh_name_set = {m.get_name() for m in mesh_objects}
    for m in mesh_objects:
        m.set_cp("category_id", cat_id)

    # Light
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(random.uniform(150, 800))
    light.set_location(tool.get_location() + np.array([0.5, 0.5, 0.8]))

    # Render controls
    bproc.renderer.set_max_amount_of_samples(96)
    bproc.renderer.set_output_format(enable_transparency=True)

    # NEW: provide a default for objects without category_id (occluders, etc.)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance", "name"],
        default_values={"category_id": None}
    )
    # Frames target
    frames_target = max(args.num_frames_per_tool, 1)
    cam_poses = []
    tries, poses = 0, 0

    # Auto-zoom heuristics
    img_diag = np.sqrt(W*W + H*H)
    target_diag = args.target_bbox_frac * img_diag
    radius_min_cur = args.radius_min

    # Precompute static world data (KP world coords + mesh AABB)
    kp_world_static = []
    for o in kp_objs:
        M = o.get_local2world_mat()
        p = (M @ np.array([0, 0, 0, 1.0]))[:3]
        kp_world_static.append(p)
    kp_world_static = np.array(kp_world_static)  # (K,3)

    M_tool = tool.get_local2world_mat()
    bbox_local = np.array(tool.get_bound_box())  # 8x3
    bbox_world = (M_tool @ np.c_[bbox_local, np.ones(8)].T).T[:, :3]

    # --- Add occluders once per tool (static for all frames of this tool) ---
    created_occluders = add_random_occluders_around(tool, bbox_world)

    # Pose sampling
    while tries < args.max_tries and poses < frames_target:
        ensure_world_bg_strength(np.random.uniform(0.2, 1.2))

        radius = np.random.uniform(radius_min_cur, args.radius_max)
        center = tool.get_location()
        loc = bproc.sampler.shell(center=center,
                                  radius_min=radius, radius_max=radius,
                                  elevation_min=-65, elevation_max=75)

        look_at = center + np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
        R = bproc.camera.rotation_from_forward_vec(look_at - loc,
                                                   inplane_rot=np.random.uniform(-0.6, 0.6))
        cam2world = bproc.math.build_transformation_mat(loc, R)

        # Must be visible
        if tool not in bproc.camera.visible_objects(cam2world):
            tries += 1
            continue
        
        # Reject back views (keep fronts & sides only)
        if not accept_front_view(cam2world, tool, front_axis, args.front_angle_deg):
            tries += 1
            continue

        # (Optional but recommended) Guard against fully-occluded tool:
        tool_bpy_for_guard = bpy.data.objects.get(tool.get_name())
        cam_loc = cam2world[:3, 3]
        if not ray_visible_to_point(cam_loc, tool.get_location(), tool_mesh_name_set):
            tries += 1
            continue            # An occluder sits between the camera and tool center → skip


        # Size control via projected mesh AABB
        pts2d_bb, valid_bb = project_world(bbox_world, cam2world, K)
        vb = pts2d_bb[valid_bb]
        if vb.shape[0] < 2:
            tries += 1
            continue

        x0, y0 = vb[:,0].min(), vb[:,1].min()
        x1, y1 = vb[:,0].max(), vb[:,1].max()
        box_diag = np.sqrt((x1-x0)**2 + (y1-y0)**2)

        if box_diag > (1.20 * target_diag):
            tries += 1
            radius_min_cur = min(args.radius_max, radius_min_cur * 1.08 + 0.05)
            continue
        if box_diag < (0.15 * target_diag):
            tries += 1
            radius_min_cur = max(1.5, radius_min_cur * 0.92)
            continue

        # Accept pose
        bproc.camera.add_camera_pose(cam2world, frame=poses)
        cam_poses.append(cam2world.copy())
        poses += 1
        tries += 1

    # Render
    data = bproc.renderer.render()

    # Writer: det/seg + images
    bproc.writer.write_coco_annotations(
        output_dir=split_dir,
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        mask_encoding_format="rle",
        append_to_existing_output=True
    )

    # Index new images
    frames_rendered = len(data["colors"])
    total_after = count_pngs(images_dir)
    start_idx = total_after - frames_rendered
    print(f"[COCO-KP] frames={frames_rendered}, images_after={total_after}, start_idx={start_idx}")

    # Build KP annotations per frame (with occlusion + symmetry-aware swaps)
    tool_bpy = bpy.data.objects.get(tool.get_name())

    for frame_idx, (rgba, cam2world) in enumerate(zip(data["colors"], cam_poses)):
        h, w = rgba.shape[0], rgba.shape[1]

        # Project KPs
        pts2d, valid = project_world(kp_world_static, cam2world, K)

        # Compute a tight bbox around *visible-to-camera* (not occlusion-tested) KPs,
        # else fallback to projected mesh bbox
        vis_pts = pts2d[valid]
        if vis_pts.shape[0] >= 1:
            x0, y0 = vis_pts[:,0].min(), vis_pts[:,1].min()
            x1, y1 = vis_pts[:,0].max(), vis_pts[:,1].max()
        else:
            pts2d_bb, valid_bb = project_world(bbox_world, cam2world, K)
            vb = pts2d_bb[valid_bb]
            if vb.shape[0] == 0:
                # nothing visible -> skip this frame
                continue
            x0, y0 = vb[:,0].min(), vb[:,1].min()
            x1, y1 = vb[:,0].max(), vb[:,1].max()

        x0c, y0c, bw, bh = clamp_box(x0, y0, x1, y1, w, h)

        # Visibility flags via ray casting
        cam_loc = cam2world[:3,3]
        v_flags = []
        for i in range(len(kp_names)):
            if not valid[i]:
                v_flags.append(0)  # behind camera -> unlabeled
                continue
            x, y = float(pts2d[i,0]), float(pts2d[i,1])
            if not in_image(x, y, w, h):
                v_flags.append(0)  # off image -> unlabeled
                continue

            # true occlusion test
            is_visible = ray_visible_to_point(cam_loc, kp_world_static[i], tool_mesh_name_set)
            v_flags.append(2 if is_visible else 1)

        # Apply left/right coupled swaps if viewing from back
        if is_back_view(cam2world, tool, front_axis=front_axis):
            pts2d_swapped = pts2d.copy()
            v_swapped = v_flags.copy()
            for (a, b) in swap_pairs:
                pts2d_swapped[a], pts2d_swapped[b] = pts2d_swapped[b].copy(), pts2d_swapped[a].copy()
                v_swapped[a],    v_swapped[b]    = v_swapped[b], v_swapped[a]
            pts2d, v_flags = pts2d_swapped, v_swapped

        # Pack COCO keypoints [x,y,v] in your specified order
        keypoints = []
        for i in range(len(kp_names)):
            x, y = float(pts2d[i,0]), float(pts2d[i,1])
            keypoints += [x, y, int(v_flags[i])]

        img_id = start_idx + frame_idx
        file_name = f"images/{img_id:06d}.png"

        if not any(im.get("id") == img_id for im in images):
            images.append({"id": img_id, "width": w, "height": h, "file_name": file_name})

        annotations.append({
            "id": next_ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "iscrowd": 0,
            "area": int(round(bw * bh)),
            "bbox": [int(round(x0c)), int(round(y0c)), int(round(bw)), int(round(bh))],  # xywh
            "num_keypoints": len(kp_names),
            "keypoints": keypoints
        })
        next_ann_id += 1

    # Persist COCO-KP JSON for this tool
    with open(kp_json_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    # Cleanup: occluders + loaded tool
    for name in created_occluders:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    bproc.object.delete_multiple(loaded, remove_all_offspring=True)
