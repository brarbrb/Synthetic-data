import blenderproc as bproc
import os, sys, re, json, argparse
import bpy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default="tools/needle_holder/NH4.blend" , help="Path to .blend file")
    p.add_argument("--out", default="out", help="Output directory for JSON")
    p.add_argument("--obj-regex", default=".*", help="Regex to include certain objects by name")
    p.add_argument("--kp-prefix", default="kp_", help="Prefix for keypoint objects (e.g., kp_1..kp_5)")
    p.add_argument("--apply-modifiers", action="store_true",
                   help="Count mesh stats on evaluated mesh (applies modifiers for counting only)")
    return p.parse_known_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])[0]

def matrix_to_list(m):
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]

def obj_local_bbox_corners(obj):
    # 8 local-space bbox corners (same order as Blender UI)
    return [tuple(c) for c in obj.bound_box]

def transform_points(mat_world, points):
    import mathutils
    mw = mat_world if isinstance(mat_world, mathutils.Matrix) else mathutils.Matrix(mat_world)
    out = []
    for p in points:
        v = mathutils.Vector(p)
        vt = mw @ v
        out.append((float(vt.x), float(vt.y), float(vt.z)))
    return out

def get_mesh_stats(obj, apply_mods=False):
    # Use depsgraph evaluated mesh when apply_mods is requested (for accurate counts after modifiers)
    depsgraph = bpy.context.evaluated_depsgraph_get() if apply_mods else None
    mesh = None
    if depsgraph:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        verts = len(mesh.vertices)
        faces = len(mesh.polygons)
        # Cleanup temp mesh
        eval_obj.to_mesh_clear()
    else:
        mesh = obj.data
        verts = len(mesh.vertices) if mesh else 0
        faces = len(mesh.polygons) if mesh else 0
    return {"num_vertices": int(verts), "num_faces": int(faces)}

def get_materials_and_textures(obj):
    mats = []
    for slot in obj.material_slots:
        mat = slot.material
        if not mat:
            continue
        entry = {"name": mat.name, "textures": []}
        if mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and getattr(node, "image", None):
                    img = node.image
                    entry["textures"].append({
                        "node": node.name,
                        "image_name": img.name,
                        "image_path": getattr(img, "filepath_raw", "") or getattr(img, "filepath", "")
                    })
        mats.append(entry)
    return mats

def camera_intrinsics(scene, cam_obj):
    cam = cam_obj.data
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.0
    width = int(res_x * scale)
    height = int(res_y * scale)

    # Choose sensor fit similar to Blender’s AUTO behavior
    sensor_fit = cam.sensor_fit if cam.sensor_fit != 'AUTO' else ('HORIZONTAL' if width >= height else 'VERTICAL')
    sensor_width = cam.sensor_width
    sensor_height = cam.sensor_height

    if cam.type == 'PERSP':
        if sensor_fit == 'HORIZONTAL':
            fx = (cam.lens / sensor_width) * width
            fy = (cam.lens / sensor_width) * width
        else:
            fx = (cam.lens / sensor_height) * height
            fy = (cam.lens / sensor_height) * height
    elif cam.type == 'ORTHO':
        fx = fy = width / cam.ortho_scale
    else:
        fx = fy = None

    cx = width / 2.0
    cy = height / 2.0

    return {
        "width": width, "height": height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "lens_mm": cam.lens if cam.type == 'PERSP' else None,
        "sensor_width_mm": sensor_width,
        "sensor_height_mm": sensor_height,
        "camera_type": cam.type,
        "sensor_fit": sensor_fit
    }

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # BlenderProc session
    bproc.init()
    # Load your .blend (materials are embedded; .mtl not needed)
    bproc.loader.load_blend(args.blend)

    scene = bpy.context.scene
    name_regex = re.compile(args.obj_regex)

    # Collections summary
    collections = [{"name": col.name, "num_objects": len(col.objects)} for col in bpy.data.collections]

    # Objects & keypoints
    objects = []
    keypoints = []

    for obj in bpy.data.objects:
        include = (obj.type == 'CAMERA') or name_regex.match(obj.name) or obj.name.startswith(args.kp_prefix)
        if not include:
            continue

        entry = {
            "name": obj.name,
            "type": obj.type,
            "parent": obj.parent.name if obj.parent else None,
            "collections": [c.name for c in obj.users_collection],
            "matrix_world": matrix_to_list(obj.matrix_world),
            "location_world": [float(x) for x in obj.matrix_world.translation],
            "rotation_euler_xyz": [float(v) for v in getattr(obj.rotation_euler, "xyz", obj.rotation_euler)],
            "scale": [float(s) for s in obj.scale],
            "dimensions_world": [float(d) for d in obj.dimensions],
        }

        if obj.type == 'MESH' and obj.data:
            entry["local_bbox_corners"] = obj_local_bbox_corners(obj)
            entry["world_bbox_corners"] = transform_points(obj.matrix_world, entry["local_bbox_corners"])
            entry["mesh_stats"] = get_mesh_stats(obj, apply_mods=args.apply_modifiers)
            entry["materials"] = get_materials_and_textures(obj)

        if obj.type == 'CAMERA':
            entry["intrinsics"] = camera_intrinsics(scene, obj)

        if obj.name.startswith(args.kp_prefix):
            kp = {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "matrix_world": matrix_to_list(obj.matrix_world),
                "location_world": [float(x) for x in obj.matrix_world.translation],
                "location_local": [float(x) for x in obj.location],
            }
            keypoints.append(kp)

        objects.append(entry)

    # Also include any armature bones whose names start with kp-prefix
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            arm = obj
            for bone in arm.data.bones:
                if bone.name.startswith(args.kp_prefix):
                    import mathutils
                    head_world = arm.matrix_world @ mathutils.Vector(bone.head_local)
                    keypoints.append({
                        "name": bone.name,
                        "type": "BONE",
                        "armature": arm.name,
                        "parent": bone.parent.name if bone.parent else None,
                        "location_world": [float(head_world.x), float(head_world.y), float(head_world.z)],
                    })

    # Cameras list (with extrinsics)
    cameras = []
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cameras.append({
                "name": obj.name,
                "matrix_world": matrix_to_list(obj.matrix_world),
                "intrinsics": camera_intrinsics(scene, obj)
            })

    render = {
        "engine": scene.render.engine,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
        "resolution_percentage": scene.render.resolution_percentage,
        "film_transparent": scene.render.film_transparent,
        "world_color": list(getattr(scene.world, "color", (0.05, 0.05, 0.05))),
        "unit_settings": {
            "system": scene.unit_settings.system,
            "scale_length": scene.unit_settings.scale_length
        }
    }

    payload = {
        "blend_file": os.path.abspath(args.blend),
        "using_blenderproc": True,
        "filters": {"obj_regex": args.obj_regex, "kp_prefix": args.kp_prefix},
        "collections": collections,
        "objects": objects,
        "keypoints": keypoints,
        "cameras": cameras,
        "render_settings": render,
        "notes": [
            "Keypoints are detected by name prefix (default: kp_). Rename empties/bones or pass --kp-prefix.",
            "World transforms use Blender right-handed coordinates (Z up).",
            "Materials/textures are read from the .blend; external .mtl is not needed."
        ]
    }

    out_json = os.path.join(args.out, "scene_introspection.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote {out_json}")

if __name__ == "__main__":
    main()
