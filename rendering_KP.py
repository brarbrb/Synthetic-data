import blenderproc as bproc
import os, sys, json, math, argparse, re
import bpy
import mathutils

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True, help="Path to .blend")
    p.add_argument("--camera_json", required=True, help="Path to camera intrinsics JSON (fx, fy, cx, cy, width, height)")
    p.add_argument("--out_renders", required=True, help="Folder for rendered PNGs")
    p.add_argument("--out_coco", required=True, help="Path to output COCO JSON")
    p.add_argument("--category", required=True, choices=["tweezers", "needle_holder"], help="Object category for KP names/skeleton")
    p.add_argument("--img_prefix", default="000000", help="Image file prefix (e.g., 000000 -> 000000.png)")
    p.add_argument("--sensor_width_mm", type=float, default=36.0, help="Assumed horizontal sensor width in mm")
    p.add_argument("--sensor_height_mm", type=float, default=24.0, help="Assumed vertical sensor height in mm")
    p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"], help="Render engine")
    return p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:])

# Category configuration: names in a fixed order + skeleton edges (1-based indexing for COCO)
CATEGORIES = {
    "tweezers": {
        "id": 2,
        "name": "tweezers",
        "kp_order": ["handle_end","left_arm","left_tip","right_arm","right_tip"],
        "skeleton": [[1,2],[2,3],[1,4],[4,5]]
    },
    "needle_holder": {
        "id": 1,
        "name": "needle_holder",
        "kp_order": ["left_tip","right_tip","joint","left_handle","right_handle"],
        "skeleton": [[1,3],[2,3],[3,4],[3,5]]
    }
}

# Name normalization helpers (tolerate underscores/spaces/case)
def norm(s):
    return re.sub(r"[\s_]+", "", s).lower()

def find_empty_by_names(scene_objs, candidate_names):
    """
    Return dict name->object for given candidate names, matched case/underscore-insensitively.
    """
    wanted = {norm(n): n for n in candidate_names}
    found_map = {}
    index = {norm(o.name): o for o in scene_objs if o.type == "EMPTY"}
    for k_norm, original in wanted.items():
        o = index.get(k_norm)
        if o is None:
            # Try a looser search (starts/endswith) just in case
            for key, obj in index.items():
                if key == k_norm or key.endswith(k_norm) or key.startswith(k_norm):
                    o = obj; break
        if o is None:
            print(f"[WARN] Empty not found for keypoint '{original}' (searched as '{k_norm}').")
        found_map[original] = o
    return found_map

def compute_object_bbox_world(mesh_objs):
    """Return (center, size_xyz, 8 corners) in world space across all given meshes."""
    if not mesh_objs:
        return None, None, None
    corners = []
    for obj in mesh_objs:
        for c in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(c))
    xs = [v.x for v in corners]; ys = [v.y for v in corners]; zs = [v.z for v in corners]
    minv = mathutils.Vector((min(xs), min(ys), min(zs)))
    maxv = mathutils.Vector((max(xs), max(ys), max(zs)))
    center = (minv + maxv) * 0.5
    size = maxv - minv
    # 8 corners axis-aligned
    cx, cy, cz = center
    sx, sy, sz = size * 0.5
    bb_corners = [
        mathutils.Vector((cx - sx, cy - sy, cz - sz)),
        mathutils.Vector((cx - sx, cy - sy, cz + sz)),
        mathutils.Vector((cx - sx, cy + sy, cz - sz)),
        mathutils.Vector((cx - sx, cy + sy, cz + sz)),
        mathutils.Vector((cx + sx, cy - sy, cz - sz)),
        mathutils.Vector((cx + sx, cy - sy, cz + sz)),
        mathutils.Vector((cx + sx, cy + sy, cz - sz)),
        mathutils.Vector((cx + sx, cy + sy, cz + sz)),
    ]
    return center, size, bb_corners

def set_render_basics(engine, width, height):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True  # transparent background
    if scene.world:
        scene.world.color = (1.0, 1.0, 1.0)  # white if not using alpha
    # A little nicer lighting: one area light if none
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light = bpy.data.lights.new(name="temp_key_light", type="AREA")
        light.energy = 1000.0
        light_obj = bpy.data.objects.new(name="temp_key_light", object_data=light)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (2.0, 2.0, 2.0)

def make_or_get_camera():
    cam = None
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            cam = o; break
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
    return cam

def set_camera_intrinsics(cam_obj, fx, fy, cx, cy, width, height, sensor_w, sensor_h):
    """
    Set Blender camera so that its intrinsics match as closely as possible to the given K.
    We'll pick sensor_fit to minimize fx/fy mismatch.
    """
    cam = cam_obj.data
    cam.type = 'PERSP'
    # Choose sensor_fit to better match fx vs fy
    # Compute implied lens for HORIZONTAL and VERTICAL fits and pick closer match
    lens_h = fx * sensor_w / width
    lens_v = fy * sensor_h / height
    # Choose based on which yields closer fx ~ fy
    if abs(lens_h - lens_v) <= 1e-3:
        cam.sensor_fit = 'HORIZONTAL'
        cam.lens = lens_h
    else:
        # pick the one whose implied other focal is closer to given
        # Try HORIZONTAL
        fy_from_h = (cam.lens / sensor_h) * height if sensor_h != 0 else float('inf')
        # but cam.lens is not set yet; set temporarily to lens_h
        fy_from_h = (lens_h / sensor_h) * height if sensor_h != 0 else float('inf')
        # Try VERTICAL
        fx_from_v = (lens_v / sensor_w) * width if sensor_w != 0 else float('inf')
        # Compare errors
        err_h = abs(fy_from_h - fy)
        err_v = abs(fx_from_v - fx)
        if err_h <= err_v:
            cam.sensor_fit = 'HORIZONTAL'
            cam.lens = lens_h
        else:
            cam.sensor_fit = 'VERTICAL'
            cam.lens = lens_v
    cam.sensor_width = sensor_w
    cam.sensor_height = sensor_h

    # set render resolution (cx,cy are used only in projection math; Blender always uses center principal point)
    bpy.context.scene.render.resolution_x = int(width)
    bpy.context.scene.render.resolution_y = int(height)
    bpy.context.scene.render.resolution_percentage = 100

def look_at(cam_obj, target, up=mathutils.Vector((0,0,1))):
    """
    Orient camera so that its -Z looks at target and +Y is 'up' as much as possible.
    """
    direction = (target - cam_obj.location).normalized()
    # Camera looks along its -Z axis
    z_axis = -direction
    x_axis = up.cross(z_axis).normalized()
    y_axis = z_axis.cross(x_axis).normalized()
    rot = mathutils.Matrix((
        (x_axis.x, y_axis.x, z_axis.x, 0.0),
        (x_axis.y, y_axis.y, z_axis.y, 0.0),
        (x_axis.z, y_axis.z, z_axis.z, 0.0),
        (0.0,      0.0,      0.0,      1.0)
    ))
    cam_obj.matrix_world = mathutils.Matrix.Translation(cam_obj.location) @ rot

def fov_from_fx(fx, width):
    return 2.0 * math.atan(width / (2.0 * fx))

def auto_place_camera(cam_obj, center, size, fx, fy, width, height):
    """
    Place camera on +Y axis, looking at center, at distance so the object fits the frame.
    Assuming camera local -Z is forward direction; we align accordingly.
    """
    sx, sy, sz = float(size.x), float(size.y), float(size.z)
    fov_x = fov_from_fx(fx, width)
    fov_y = fov_from_fx(fy, height)

    # Required half extents in camera's image axes (we'll align so X=world X, Y=world Z)
    # We'll view from +Y towards center, so image axes ~ world X (horizontal) and Z (vertical)
    req_dx = (sx * 0.5) / math.tan(max(1e-6, fov_x * 0.5))
    req_dz = (sz * 0.5) / math.tan(max(1e-6, fov_y * 0.5))
    dist = max(req_dx, req_dz) * 1.2  # 20% margin

    cam_loc = mathutils.Vector((center.x, center.y + dist, center.z))  # on +Y
    cam_obj.location = cam_loc
    look_at(cam_obj, center)

def world_to_cam_pixels(cam_obj, P_world, fx, fy, cx, cy):
    """
    Project a world point to pixel coordinates using Blender camera extrinsics + given intrinsics.
    Blender camera space: -Z is forward. So Zc = -(R^T (P - C)).z must be > 0 to be in front.
    """
    cam_inv = cam_obj.matrix_world.inverted()
    Pc = cam_inv @ P_world
    Xc, Yc, Zc_blender = Pc.x, Pc.y, Pc.z
    Zc = -Zc_blender  # forward
    if Zc <= 1e-6:
        return (None, None, 0.0)  # behind camera
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy
    return (u, v, Zc)

def project_keypoints(cam_obj, kp_objs, fx, fy, cx, cy, width, height):
    keypoints_xyv = []
    num_vis = 0
    for o in kp_objs:
        if o is None:
            keypoints_xyv.extend([0.0, 0.0, 0])  # not present
            continue
        u, v, Zc = world_to_cam_pixels(cam_obj, o.matrix_world.translation, fx, fy, cx, cy)
        if u is None:
            keypoints_xyv.extend([0.0, 0.0, 0])
        else:
            vflag = 2 if (0 <= u < width and 0 <= v < height) else 1
            num_vis += (1 if vflag == 2 else 0)
            keypoints_xyv.extend([float(u), float(v), int(vflag)])
    return keypoints_xyv, num_vis

def project_bbox_from_world_corners(cam_obj, corners_world, fx, fy, cx, cy, width, height):
    us, vs = [], []
    for Pw in corners_world:
        u, v, Zc = world_to_cam_pixels(cam_obj, Pw, fx, fy, cx, cy)
        if u is not None:
            us.append(u); vs.append(v)
    if not us:
        # fallback: zero bbox
        return [0.0, 0.0, 0.0, 0.0]
    xmin = max(0.0, min(us))
    ymin = max(0.0, min(vs))
    xmax = min(float(width-1), max(us))
    ymax = min(float(height-1), max(vs))
    return [float(xmin), float(ymin), float(max(0.0, xmax - xmin)), float(max(0.0, ymax - ymin))]

def main():
    args = parse_args()
    os.makedirs(args.out_renders, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_coco), exist_ok=True)

    # Load intrinsics
    with open(args.camera_json, "r") as f:
        K = json.load(f)
    fx, fy, cx, cy = float(K["fx"]), float(K["fy"]), float(K["cx"]), float(K["cy"])
    width, height = int(K["width"]), int(K["height"])

    # Init BProc / load scene
    bproc.init()
    bproc.loader.load_blend(args.blend)

    # Prepare render settings
    set_render_basics(args.engine, width, height)

    # Build category + keypoint config
    cfg = CATEGORIES[args.category]
    kp_names = cfg["kp_order"]
    skeleton = cfg["skeleton"]
    cat_id = cfg["id"]
    cat_name = cfg["name"]

    # Gather objects
    all_objs = list(bpy.data.objects)
    mesh_objs = [o for o in all_objs if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("No MESH objects found in the .blend")
    center, size, bb_corners = compute_object_bbox_world(mesh_objs)
    if center is None:
        raise RuntimeError("Failed to compute object bounds")

    # Make/find camera and set intrinsics
    cam_obj = make_or_get_camera()
    set_camera_intrinsics(cam_obj, fx, fy, cx, cy, width, height, args.sensor_width_mm, args.sensor_height_mm)
    auto_place_camera(cam_obj, center, size, fx, fy, width, height)

    # Find keypoint empties by exact/normalized names
    kp_map = find_empty_by_names(all_objs, kp_names)
    kp_objs_ordered = [kp_map.get(n, None) for n in kp_names]

    # Safety check
    missing = [n for n, o in kp_map.items() if o is None]
    if missing:
        print("[WARN] Missing keypoint empties:", missing)

    # Render a single frame to PNG
    img_id = 1
    file_name = f"{args.img_prefix}.png"
    out_path = os.path.join(args.out_renders, file_name)
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"[OK] Rendered {out_path}")

    # Project keypoints + bbox
    keypoints_xyv, num_vis = project_keypoints(cam_obj, kp_objs_ordered, fx, fy, cx, cy, width, height)
    bbox_xywh = project_bbox_from_world_corners(cam_obj, bb_corners, fx, fy, cx, cy, width, height)
    area = bbox_xywh[2] * bbox_xywh[3]

    # COCO output
    coco = {
        "images": [{
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        }],
        "annotations": [{
            "id": 1,
            "image_id": img_id,
            "category_id": cat_id,
            "iscrowd": 0,
            "area": float(area),
            "bbox": bbox_xywh,
            "num_keypoints": int(sum(1 for i in range(2, len(keypoints_xyv), 3) if keypoints_xyv[i] > 0)),
            "keypoints": keypoints_xyv
        }],
        "categories": [{
            "id": cat_id,
            "name": cat_name,
            "keypoints": kp_names,
            "skeleton": skeleton
        }]
    }
    with open(args.out_coco, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"[OK] Wrote COCO annotations to {args.out_coco}")

if __name__ == "__main__":
    main()
