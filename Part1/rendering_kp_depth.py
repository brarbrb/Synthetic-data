import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
import argparse, os, json, random, glob

# In this file we render with depth aware and clear background!

# -------- Categories, KP order, skeletons --------
CATEGORIES = {
    "tweezers": {
        "id": 2,
        "name": "tweezers",
        "kp_order": ["handle_end", "left_arm", "left_tip", "right_arm", "right_tip"],
        "skeleton": [[0,1],[0,3],[1,2],[3,4]],
        "swap_pairs": [[1,3],[2,4]],
        "front_axis": "Z"   # which local axis is "front" (+X,+Y,+Z or -X,-Y,-Z) 
                            # from trying different ones "Z" is the correct one!!!

    },
    "needle_holder": {
        "id": 1,
        "name": "needle_holder",
        "kp_order": ["joint", "left_handle", "left_tip", "right_handle", "right_tip"],
        "skeleton": [[0,1],[0,2],[0,3],[0,4]],
        "swap_pairs": [[1,3],[2,4]],
        "front_axis": "Z"
    }
}
# -------- Args --------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', default="train", help="train/val/test")
parser.add_argument('--tools_root', default="tools_blend", help="Folder containing class subfolders with .blend files")
parser.add_argument('--camera_params', default="camera.json", help="Intrinsics JSON")
parser.add_argument('--output_dir', default="out", help="Output root")
parser.add_argument('--num_frames_per_tool', type=int, default=30, help="Desired frames per tool (min 30 enforced)")
parser.add_argument('--radius_min', type=float, default=3.0, help="Min camera radius (meters)")
parser.add_argument('--radius_max', type=float, default=12.0, help="Max camera radius (meters)")
parser.add_argument('--target_bbox_frac', type=float, default=0.33, help="Target bbox diag as fraction of image diag")
parser.add_argument('--max_tries', type=int, default=8000, help="Pose sampling guard")
parser.add_argument('--front_angle_deg', type=float, default=85.0, help="Max angle from front to accept (<= this = OK; > this = treat as back)")

args = parser.parse_args()

bproc.init(clean_up_scene=True)
# Render controls - needed once per running :) 
bproc.renderer.set_max_amount_of_samples(96)
bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
bproc.renderer.enable_depth_output(activate_antialiasing=False)  # we need depth maps for occlusion flags


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
def _axis_vec(axis_str: str) -> np.ndarray:
    # axis_str like "X","Y","Z","-X","-Y","-Z"
    s = axis_str.strip().upper()
    sign = -1.0 if s.startswith('-') else 1.0
    a = s[-1]
    base = {'X': np.array([1.0,0.0,0.0]),
            'Y': np.array([0.0,1.0,0.0]),
            'Z': np.array([0.0,0.0,1.0])}[a]
    return sign * base

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v

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

def project_world_with_Z(points_xyz, cam2world, K):
    world2cam = np.linalg.inv(cam2world)
    Pw = np.c_[points_xyz, np.ones(len(points_xyz))]
    Pc = (world2cam @ Pw.T).T[:, :3]   # camera space (X,Y,Z)
    X, Y, Z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    Zc = -Z  # positive when in front of camera
    valid = Zc > 1e-6
    x = K[0, 0] * (X / np.where(valid, Zc, 1.0)) + K[0, 2]
    y = -K[1, 1] * (Y / np.where(valid, Zc, 1.0)) + K[1, 2]
    return np.stack([x, y], axis=1), Zc, valid

def clamp_box(x0,y0,x1,y1,w,h):
    x0c = float(np.clip(x0, 0, w-1)); y0c = float(np.clip(y0, 0, h-1))
    x1c = float(np.clip(x1, 0, w-1)); y1c = float(np.clip(y1, 0, h-1))
    bw = max(1.0, x1c - x0c + 1.0)
    bh = max(1.0, y1c - y0c + 1.0)
    return x0c, y0c, bw, bh

def vflag(x, y, w, h):
    return 2 if (0.0 <= x <= w-1 and 0.0 <= y <= h-1) else 1

def vflag_occlusion(x, y, w, h, Zc, depth_map, tol=0.01):
    # If outside image → 0
    if not (0 <= x < w and 0 <= y < h):
        return 0
    # If projected behind camera → 0
    if Zc <= 0:
        return 0
    # Compare depth at pixel to keypoint depth (smaller = closer)
    di = depth_map[int(round(y)), int(round(x))]
    if not np.isfinite(di):
        # No depth written (background). Keypoint not actually visible.
        return 1  # mark as occluded but present
    # If our 3D point is the front-most within tolerance → visible (2)
    return 2 if (Zc <= di + tol) else 1

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
    cat_id = int(spec["id"])

    print(f"\n=== {cls_name} :: {os.path.basename(blend_path)} ===")
    bproc.utility.reset_keyframes()

    # Load and upcast to concrete types
    loaded = bproc.loader.load_blend(blend_path)
    # loaded = bproc.object.convert_to_entities(loaded, convert_to_subclasses=True)

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

    # Choose tool mesh: prefer MeshObject, else largest non-KP, non-Camera entity
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

    tool.set_cp("category_id", cat_id)

    # Light
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(random.uniform(150, 800))
    light.set_location(tool.get_location() + np.array([0.5, 0.5, 0.8]))

    # Frames target: at least 30
    frames_target = max(args.num_frames_per_tool, 1) # at least 1
    cam_poses = []
    tries, poses = 0, 0

    # Auto-zoom heuristics
    img_diag = np.sqrt(W*W + H*H)
    target_diag = args.target_bbox_frac * img_diag
    radius_min_cur = args.radius_min  # local copy so we don't pollute next tools

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
    front_axis = spec.get("front_axis", "Y")
    front_local = _axis_vec(front_axis)
    R_tool = M_tool[:3, :3]                 # rotation part
    front_world = _normalize(R_tool @ front_local)  # tool's "front" in world coords


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
        
        # Camera → tool direction (what the camera is looking toward)
        view_dir = _normalize(tool.get_location() - loc)

        # Angle between tool front and camera view direction
        view_dir = _normalize(tool.get_location() - loc)  
        cos_thresh = np.cos(np.deg2rad(args.front_angle_deg))
        cosang = -float(np.dot(front_world, view_dir))    # negate to turn "front" positive

        if cosang < cos_thresh:
            # Too far around the back → reject pose
            tries += 1
            continue

        # Must be visible
        if tool not in bproc.camera.visible_objects(cam2world):
            tries += 1
            continue

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
    
    depth_maps = data["depth"]  # list of HxW float32, distance from camera


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

    # Build KP annotations per frame (using cached kp_world_static)
    for frame_idx, (rgba, cam2world) in enumerate(zip(data["colors"], cam_poses)):
        h, w = rgba.shape[0], rgba.shape[1]

        # --- NEW: project with Z and use depth for occlusion-aware visibility ---
        pts2d, Zc_all, valid = project_world_with_Z(kp_world_static, cam2world, K)
        depth_map = depth_maps[frame_idx]

        vis_pts = pts2d[valid]
        if vis_pts.shape[0] >= 1:
            x0, y0 = vis_pts[:, 0].min(), vis_pts[:, 1].min()
            x1, y1 = vis_pts[:, 0].max(), vis_pts[:, 1].max()
        else:
            # fallback on mesh bbox if no KP is in front
            pts2d_bb, valid_bb = project_world(bbox_world, cam2world, K)
            vb = pts2d_bb[valid_bb]
            if vb.shape[0] == 0:
                continue
            x0, y0 = vb[:, 0].min(), vb[:, 1].min()
            x1, y1 = vb[:, 0].max(), vb[:, 1].max()

        x0c, y0c, bw, bh = clamp_box(x0, y0, x1, y1, w, h)

        # --- NEW: occlusion-aware [x,y,v] using depth ---
        keypoints = []
        for i in range(len(kp_names)):
            x, y = float(pts2d[i, 0]), float(pts2d[i, 1])
            v = vflag_occlusion(x, y, w, h, float(Zc_all[i]), depth_map, tol=0.01)
            keypoints += [x, y, v]

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
            "bbox": [int(round(x0c)), int(round(y0c)), int(round(bw)), int(round(bh))],
            "num_keypoints": len(kp_names),
            "keypoints": keypoints
        })
        next_ann_id += 1

    # Persist COCO-KP JSON for this tool
    with open(kp_json_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    # Now it's safe to delete everything we loaded from this .blend
    bproc.object.delete_multiple(loaded, remove_all_offspring=True)