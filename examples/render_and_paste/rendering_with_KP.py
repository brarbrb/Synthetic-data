import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
import argparse
import random
import os
import json
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt


IDS = {"needle_holder" : 1, "tweezers" : 2}

# from
material_features = ['Alpha', 'Anisotropic', 'Anisotropic Rotation', 'Base Color', 
                    'Coat IOR', 'Coat Normal', 'Coat Roughness', 'Coat Tint', 'Coat Weight',
                    'Emission Color', 'Emission Strength', 'IOR', 'Metallic', 'Normal', 'Roughness',
                    'Sheen Roughness', 'Sheen Tint', 'Sheen Weight', 'Specular IOR Level',
                    'Specular Tint', 'Subsurface Anisotropy', 'Subsurface IOR', 'Subsurface Radius',
                    'Subsurface Scale', 'Subsurface Weight', 'Tangent', 'Thin Film IOR',
                    'Thin Film Thickness', 'Transmission Weight', 'Weight'] # got it from previous EDA

# coco_dataset_path_synth = "output/coco_synthetic"
# labels_path = coco_dataset_path_synth + "/labels"
# images_path = coco_dataset_path_synth + "/images"
# train_set_path = "/train/"
# val_set_path = "/val/"
# test_set_path = "" # this will be for final product and estimation based on videos

# os.makedirs(coco_dataset_path_synth, exist_ok=True) 
# os.makedirs(labels_path, exist_ok=True)
# os.makedirs(images_path, exist_ok=True) 
# os.makedirs(labels_path + train_set_path, exist_ok=True) 
# os.makedirs(labels_path + val_set_path, exist_ok=True) 
# os.makedirs(images_path + train_set_path, exist_ok=True) 
# os.makedirs(images_path + val_set_path, exist_ok=True) 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', default="val", help="for what dataset to create the images")
parser.add_argument('--obj_folder', default="tools/", help="Path to folder with objects.")
parser.add_argument('--camera_params', default="camera.json", help="Camera intrinsics in json format")
parser.add_argument('--output_dir', default="output", help="Path to where the final files, will be saved")
# parser.add_argument('--num_images', type=int, default = 25, help="Number of images to generate") # for train
# parser.add_argument('--num_images', type=int, default = 1, help="Number of images to generate") # val set
args = parser.parse_args()
bproc.init()


for subfolder in os.listdir(args.obj_folder):
    full_path = os.path.join(args.obj_folder, subfolder)
    if os.path.isdir(full_path):
        print(f"entering the {subfolder}")
        for obj_name in os.listdir(full_path):
            if obj_name.endswith('.mtl'):
                continue
            print(f"creating images for {obj_name} on random backgrounds")
            obj_path = os.path.join(full_path, obj_name)
            # ### NEW: prefer .blend that matches the object’s base name
            base, _ext = os.path.splitext(obj_path) #TODO: update the assets and the path!
            blend_path = base + ".blend"
            if os.path.exists(blend_path):
                loaded = bproc.loader.load_blend(blend_path)     # returns mesh + empties you created
            else:
                loaded = bproc.loader.load_obj(obj_path)

            # ### CHANGED: split into mesh and keypoint empties
            mesh_objs = [o for o in loaded if isinstance(o, bproc.types.MeshObject)]
            assert len(mesh_objs) >= 1, f"No mesh found in {obj_path}"
            obj = mesh_objs[0]
            obj.set_cp("category_id", IDS[subfolder])

            # Heuristic: your keypoints are Empties (Entities) whose names start with 'kp_'.
            # Adjust the predicate if you used a different naming scheme.
            kp_entities = [
                o for o in loaded
                if not isinstance(o, bproc.types.MeshObject)
                # Entities are Empties/empties-with-constraints etc. We also check name:
                and hasattr(o, "get_name") and o.get_name().lower().startswith("kp_")
            ]

            # ### NEW: deterministic keypoint order by name
            kp_entities.sort(key=lambda e: e.get_name().lower())

            # If you want a specific order, define it here and reorder accordingly:
            # DESIRED_ORDER = ["kp_right_tip","kp_left_tip","kp_hinge","kp_handle_r","kp_handle_l"]
            # name2idx = {n:i for i,n in enumerate(DESIRED_ORDER)}
            # kp_entities.sort(key=lambda e: name2idx.get(e.get_name(), 10**6))

            # ### NEW: keep the keypoint names for COCO categories
            kp_names = [e.get_name() for e in kp_entities]
            # example skeleton: chain them and connect all to the first; tailor as you like
            skeleton = [[i, i+1] for i in range(1, len(kp_names))] + [[1, j] for j in range(2, len(kp_names)+1)]

            # (you can remove the old bbox block entirely)
            # --- delete the previous bbox_corners/... lines

            # ... your lighting, intrinsics, and camera pose sampling stays the same ...

            # after rendering:
            data = bproc.renderer.render()

            # ### NEW: stash the cam poses you used (in the same order you added them)
            # You already add with: bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
            # We'll reconstruct per-frame extrinsics from those.
            cam2world_list = []
            for f in range(len(bproc.camera._CameraUtility__cam2world_mats)):
                cam2world_list.append(bproc.camera._CameraUtility__cam2world_mats[f])

            # ### NEW: collect world-space coordinates of keypoint empties once (they don't move)
            # If you animated/parented them differently, you can recompute per frame, but for your current code camera moves, object is static.
            kp_world = []
            for e in kp_entities:
                # get_location() is world-space for Entities
                kp_world.append(np.array(e.get_location()))
            kp_world = np.stack(kp_world, axis=0) if len(kp_world) else np.zeros((0,3), dtype=np.float32)

            # ... keep your writer.write_coco_annotations block as-is ...

            # ### CHANGED: build COCO scaffolding with real kp names
            images = []
            annotations = []
            categories = []
            for name, cid in IDS.items():
                categories.append({
                    "id": cid,
                    "name": name,
                    "supercategory": "tool",
                    "keypoints": kp_names,
                    "skeleton": skeleton
                })

            ann_id = 1

            def project_points(K, cam2world, XYZ):
                """Project Nx3 world points using intrinsics K and cam2world (4x4)."""
                W2C = np.linalg.inv(cam2world)
                R = W2C[:3, :3]
                t = W2C[:3, 3:4]  # (3,1)
                # cam coords: Xc = R*X + t
                X = XYZ.T  # (3,N)
                Xc = (R @ X) + t  # (3,N)
                zs = Xc[2, :]
                # pinhole
                xyn = (K @ Xc)  # (3,N)
                u = xyn[0, :] / zs
                v = xyn[1, :] / zs
                return u, v, zs

            for frame_idx, (rgba, seg) in enumerate(zip(data["colors"], data["instance_segmaps"])):
                h, w = seg.shape
                file_name = f"{frame_idx:06d}.png"
                images.append({"id": frame_idx+1, "width": w, "height": h, "file_name": file_name})

                # Resolve category_id map if present
                attr_maps = data.get("instance_attribute_maps", [{}]*len(data["colors"]))
                frame_attrs = attr_maps[frame_idx] if frame_idx < len(attr_maps) else {}

                # For each visible instance in this frame
                instance_ids = np.unique(seg)
                instance_ids = instance_ids[instance_ids != 0]

                # ### NEW: project the empties for this frame
                if kp_world.shape[0] > 0:
                    cam2world = cam2world_list[frame_idx]
                    u, v, z = project_points(K, cam2world, kp_world)
                    # visibility: z>0 (in front of camera) AND inside image bounds
                    vis = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                    # COCO v-flag: 2 visible, 0 not labeled/not visible
                    vflags = np.where(vis, 2, 0)
                    # clamp to ints for JSON
                    u_int = np.clip(np.round(u).astype(int), 0, w-1)
                    v_int = np.clip(np.round(v).astype(int), 0, h-1)

                    kp_flat_this_frame = []
                    for i in range(len(kp_names)):
                        if vflags[i] == 0:
                            kp_flat_this_frame.extend([0, 0, 0])
                        else:
                            kp_flat_this_frame.extend([int(u_int[i]), int(v_int[i]), int(vflags[i])])
                else:
                    kp_flat_this_frame = []

                for iid in instance_ids:
                    ys, xs = np.where(seg == iid)
                    if ys.size == 0:
                        continue

                    x0, y0 = int(xs.min()), int(ys.min())
                    x1, y1 = int(xs.max()), int(ys.max())
                    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)

                    # category id from attribute map if available
                    cat_id = None
                    if isinstance(frame_attrs, dict) and "category_id" in frame_attrs:
                        cat_map = frame_attrs["category_id"]
                        cy = int(round((y0 + y1) / 2))
                        cx = int(round((x0 + x1) / 2))
                        cy = np.clip(cy, 0, h-1); cx = np.clip(cx, 0, w-1)
                        cat_id = int(cat_map[cy, cx])
                    if not cat_id:
                        cat_id = 1  # fallback

                    annotations.append({
                        "id": ann_id,
                        "image_id": frame_idx + 1,
                        "category_id": cat_id,
                        "iscrowd": 0,
                        "area": int(bw * bh),
                        "bbox": [int(x0), int(y0), int(bw), int(bh)],
                        "num_keypoints": len(kp_names),
                        "keypoints": kp_flat_this_frame,   # ### NEW: real projected keypoints
                    })
                    ann_id += 1