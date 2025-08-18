import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
import argparse
import random
import os
import glob
import json
from colorsys import hsv_to_rgb

IDS = {"needle_holder" : 1, "tweezers" : 2}

# from
material_features = ['Alpha', 'Anisotropic', 'Anisotropic Rotation', 'Base Color', 
                    'Coat IOR', 'Coat Normal', 'Coat Roughness', 'Coat Tint', 'Coat Weight',
                    'Emission Color', 'Emission Strength', 'IOR', 'Metallic', 'Normal', 'Roughness',
                    'Sheen Roughness', 'Sheen Tint', 'Sheen Weight', 'Specular IOR Level',
                    'Specular Tint', 'Subsurface Anisotropy', 'Subsurface Radius',
                    'Subsurface Scale', 'Subsurface Weight', 'Tangent', 'Thin Film IOR',
                    'Thin Film Thickness', 'Transmission Weight'] # got it from previous EDA


def get_hdr_img_paths_from_haven(data_path: str) -> str:
    """ Returns .hdr file paths from the given directory.

    :param data_path: A path pointing to a directory containing .hdr files.
    :return: .hdr file paths
    """

    if os.path.exists(data_path):
        data_path = os.path.join(data_path, "hdris")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The folder: {data_path} does not contain a folder name hdfris. "
                                    f"Please use the download script.")
    else:
        raise FileNotFoundError(f"The data path does not exists: {data_path}")

    hdr_files = glob.glob(os.path.join(data_path, "*", "*.hdr"))
    # this will be ensure that the call is deterministic
    hdr_files.sort()
    return hdr_files

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', default="train", help="for what dataset to create the images") # change to train when needed
parser.add_argument('--obj_folder', default="/datashare/project/surgical_tools_models/", help="Path to folder with objects.")
parser.add_argument('--camera_params', default="camera.json", help="Camera intrinsics in json format")
parser.add_argument('--output_dir', default="output", help="Path to where the final files, will be saved")
parser.add_argument('--haven_path', default="/datashare/project/haven/", help="Path to the haven hdri images")

args = parser.parse_args()

# --- Paths for manual keypoints COCO ---
kp_out_dir = os.path.join(args.output_dir, args.dataset_type)
os.makedirs(kp_out_dir, exist_ok=True)
kp_json_path = os.path.join(kp_out_dir, 'coco_keypoints.json')
images_dir = os.path.join(args.output_dir, args.dataset_type, 'images')  # BlenderProc writer target

# --- Helper: 3D world -> 2D pixels ---
def project_world_points_to_image(points_world_xyz, cam2world, K):
    """
    Project 3D world points into pixels for Blender/BlenderProc.
    Blender camera looks along -Z in camera space, so use Zc = -Z.
    """
    world2cam = np.linalg.inv(cam2world)
    Pw_h = np.concatenate([points_world_xyz, np.ones((points_world_xyz.shape[0], 1))], axis=1)  # (N,4)
    Pc = (world2cam @ Pw_h.T).T[:, :3]  # (N,3) camera-space
    X, Y, Z = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    Zc = -Z                           # <-- Blender camera forward
    valid = Zc > 1e-6                 # points in front of camera
    Zsafe = np.where(valid, Zc, 1.0)  # avoid div-by-zero for invalid points

    x = K[0, 0] * (X / Zsafe) + K[0, 2]
    y = K[1, 1] * (Y / Zsafe) + K[1, 2]
    return np.stack([x, y], axis=1), valid


# --- Load/create the cumulative manual keypoints COCO (zero-based ids) ---
if os.path.exists(kp_json_path):
    with open(kp_json_path, "r") as f:
        kp_data = json.load(f)
    images = kp_data.get("images", [])
    annotations = kp_data.get("annotations", [])
    categories = kp_data.get("categories", [])
    next_ann_id = (max((an["id"] for an in annotations), default=-1) + 1)
else:
    kp_names = ["bb_tl", "bb_tr", "bb_br", "bb_bl", "bb_center"]
    skeleton = [[1,2],[2,3],[3,4],[4,1],[1,5],[2,5],[3,5],[4,5]]  # 1-based pairs
    categories = [{"id": cid, "name": name, "supercategory": "tool",
                   "keypoints": kp_names, "skeleton": skeleton}
                  for name, cid in IDS.items()]
    images, annotations = [], []
    next_ann_id = 0

# --- Camera intrinsics once per run (you can move inside loop if per-object differs) ---
with open(args.camera_params, "r") as file:
    camera_params = json.load(file)
fx = camera_params["fx"]; fy = camera_params["fy"]
cx = camera_params["cx"]; cy = camera_params["cy"]
im_width = camera_params["width"]; im_height = camera_params["height"]
K = np.array([[fx, 0,  cx],
              [0,  fy, cy],
              [0,  0,  1]], dtype=float)
CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height)

def count_pngs(folder):
    try:
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(".png"))
    except FileNotFoundError:
        return 0

bproc.init()

for subfolder in os.listdir(args.obj_folder):
    full_path = os.path.join(args.obj_folder, subfolder)
    if not os.path.isdir(full_path):
        continue

    print(f"entering the {subfolder}")
    for obj_name in os.listdir(full_path):
        if obj_name.endswith('.mtl'):
            continue

        print(f"creating images for {obj_name} on random backgrounds")
        obj_path = os.path.join(full_path, obj_name)
        loaded = bproc.loader.load_obj(obj_path)   # list
        obj = loaded[0]
        obj.set_cp("category_id", IDS[subfolder])

        # Object AABB corners in WORLD coords
        obj2world_matrix = obj.get_local2world_mat()
        bbox_local = np.array(obj.get_bound_box())            # (8,3) local
        bbox_local_h = np.concatenate([bbox_local, np.ones((8, 1))], axis=1)
        bbox_world_xyz = ((obj2world_matrix @ bbox_local_h.T).T)[:, :3]  # (8,3)
        # Randomize materials
        for mat in obj.get_materials():
            for feat in random.sample(material_features, 5):
                try:
                    if feat in ["Base Color","Emission Color","Coat Tint","Sheen Tint","Specular Tint"]:
                        val = tuple(random.uniform(0, 1) for _ in range(4))
                    elif feat in ["Coat Normal","Normal","Subsurface Radius", "Tangent"]:
                        val = tuple(random.uniform(0, 1) for _ in range(3))
                    else:
                        val = random.uniform(0, 1)
                    mat.set_principled_shader_value(feat, val)
                except Exception as e:
                    print(f"Could not set {feat} for {mat.get_name()}: {e}")
        # Light
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(bproc.sampler.shell(
            center=obj.get_location(),
            radius_min=1, radius_max=5,
            elevation_min=1, elevation_max=89
        ))
        light.set_energy(random.uniform(100, 1000))

        # --- Sample camera poses (store them locally!) ---
        cam_poses = []  # <<< store matrices here
        # load HDRI
        hdr_files = get_hdr_img_paths_from_haven(args.haven_path)
        poses, tries = 0, 0
        num_images = 25 if args.dataset_type == "train" else 3
        while tries < 10000 and poses < num_images:
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)
            random_hdr_file = random.choice(hdr_files)
            bproc.world.set_world_background_hdr_img(random_hdr_file)
            
            location = bproc.sampler.shell(
                center=obj.get_location(),
                radius_min=2,
                radius_max=10,
                elevation_min=-90,
                elevation_max=90
            )
            
            lookat_point = obj.get_location() + np.random.uniform([-0.5,-0.5,-0.5],[0.5,0.5,0.5])
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                lookat_point - location,
                inplane_rot=np.random.uniform(-0.7854, 0.7854)
            )
            cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)

            if obj in bproc.camera.visible_objects(cam2world):
                bproc.camera.add_camera_pose(cam2world, frame=poses)
                cam_poses.append(cam2world.copy())  # <<< keep it for later projection
                poses += 1
            tries += 1

        bproc.renderer.set_max_amount_of_samples(100)
        bproc.renderer.set_output_format(enable_transparency=False)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
        data = bproc.renderer.render()
        
        bproc.writer.write_coco_annotations(os.path.join(args.output_dir, args.dataset_type),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    mask_encoding_format="rle",
                                    append_to_existing_output=True)
            

        # --- Map our manual JSON to the exact filenames BlenderProc just wrote ---
        frames_rendered = len(data["colors"])
        total_after = count_pngs(images_dir)                 # images in folder AFTER write
        start_idx = total_after - frames_rendered            # index of first new frame (zero-based)
        print(f"[KP] frames_rendered={frames_rendered}, images_after={total_after}, start_idx={start_idx}")
        
        # Build manual keypoints/bboxes from 3D→2D projected AABB, per frame
        for frame_idx, (rgba, cam2world) in enumerate(zip(data["colors"], cam_poses)):
            h, w = rgba.shape[0], rgba.shape[1]

            # Project 8 world-space bbox corners
            proj_pts, valid_mask = project_world_points_to_image(bbox_world_xyz, cam2world, K)
            valid_pts = proj_pts[valid_mask]
            if valid_pts.shape[0] == 0:
                continue  # nothing in front of camera

            # 2D AABB from projected points, clamped to image
            x0 = float(np.min(valid_pts[:, 0])); y0 = float(np.min(valid_pts[:, 1]))
            x1 = float(np.max(valid_pts[:, 0])); y1 = float(np.max(valid_pts[:, 1]))
            x0c = max(0.0, min(x0, w - 1.0))
            y0c = max(0.0, min(y0, h - 1.0))
            x1c = max(0.0, min(x1, w - 1.0))
            y1c = max(0.0, min(y1, h - 1.0))
            bw = max(1.0, x1c - x0c + 1.0)
            bh = max(1.0, y1c - y0c + 1.0)

            # 5 keypoints: TL, TR, BR, BL, center
            tl = (x0c, y0c); tr = (x1c, y0c); br = (x1c, y1c); bl = (x0c, y1c)
            cxp = 0.5 * (x0c + x1c); cyp = 0.5 * (y0c + y1c)
            def vflag(pt):
                x, y = pt
                return 2 if (0.0 <= x <= (w - 1.0) and 0.0 <= y <= (h - 1.0)) else 1
            keypoints_flat = [
                float(tl[0]), float(tl[1]), vflag(tl),
                float(tr[0]), float(tr[1]), vflag(tr),
                float(br[0]), float(br[1]), vflag(br),
                float(bl[0]), float(bl[1]), vflag(bl),
                float(cxp),   float(cyp),   vflag((cxp, cyp))
            ]

            # Image id/filename matched to disk (zero-based, start_idx + frame_idx)
            img_id = start_idx + frame_idx
            file_name = f"images/{img_id:06d}.png"

            # Append the image row if not already present (avoid duplicates on reruns)
            if not any(im.get("id") == img_id for im in images):
                images.append({"id": img_id, "width": w, "height": h, "file_name": file_name})

            # One annotation per instance/object (your scene has one tool per render pass)
            annotations.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": int(IDS[subfolder]),
                "iscrowd": 0,
                "area": int(round(bw * bh)),
                "bbox": [int(round(x0c)), int(round(y0c)), int(round(bw)), int(round(bh))],  # XYWH
                "num_keypoints": 5,
                "keypoints": keypoints_flat
            })
            next_ann_id += 1

        # Save cumulative manual keypoints JSON
        with open(kp_json_path, "w") as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

        print(f"Appended projected-AABB keypoints; total images now: {len(images)}")

        bproc.object.delete_multiple(loaded, remove_all_offspring=True)