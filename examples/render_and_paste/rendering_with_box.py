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
parser.add_argument('--obj_folder', default="/datashare/project/surgical_tools_models/", help="Path to folder with objects.")
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
            loaded = bproc.loader.load_obj(obj_path)   # <- this is a list
            obj = loaded[0]
            # obj = bproc.loader.load_obj(obj_path)[0]  # load the objects into the scene
            obj.set_cp("category_id", IDS[subfolder]) 
            
            # Key Point part - here I used bounding box
            # Get transformation matrix from object (object to world)
            obj2world_matrix = obj.get_local2world_mat() 
            bbox_corners = np.array(obj.get_bound_box())  # Get the bounding box corners in local coordinates
            bbox_corners_h = np.concatenate([bbox_corners, np.ones((8, 1))], axis=1)  # Convert bbox corners to homogeneous coordinates
            
            # Transform to world coordinates
            bbox_world = (obj2world_matrix @ bbox_corners_h.T).T[:, :3]
            
            materials = obj.get_materials()
            possible_mat = material_features.copy()
            # Randomly perturbate the material of the object - !!! do it 20 times per each object - 20 poses for each object
            for i in range(len(materials)):
                mat = materials[i]
                materials_to_change = random.sample(possible_mat, 5) # choosing 5 random elements
                for material in materials_to_change:
                    if material in ["Base Color", "Emission Color", "Coat Tint", "Sheen Tint", "Specular Tint"]:
                        value = tuple(random.uniform(0, 1) for _ in range(4))
                    elif material in ["Coat Normal", "Normal", "Subsurface Radius"]:
                        value = tuple(random.uniform(0, 1) for _ in range(3))
                    else:
                        value = random.uniform(0, 1)
                    try:
                        mat.set_principled_shader_value(material, value)
                    except Exception as e:
                        print(f"Could not set {material} for {mat.get_name()}: {e}")
                    
            # Create a new light
            light = bproc.types.Light()
            light.set_type("POINT")
            # Sample its location around the object
            light.set_location(bproc.sampler.shell(
                center=obj.get_location(),
                radius_min=1,
                radius_max=5,
                elevation_min=1,
                elevation_max=89
            ))

            light.set_energy(random.uniform(100, 1000))
            
            with open(args.camera_params, "r") as file:
                camera_params = json.load(file)

            fx = camera_params["fx"]
            fy = camera_params["fy"]
            cx = camera_params["cx"]
            cy = camera_params["cy"]
            im_width = camera_params["width"]
            im_height = camera_params["height"]
            K = np.array([[fx, 0, cx], 
                          [0, fy, cy], 
                          [0, 0, 1]])
            CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height) 
            poses = 0
            tries = 0
            if args.dataset_type == "train":
                num_images = 25
            else:
                num_images = 3
            while tries < 10000 and poses < num_images:

                # Set a random world lighting strength
                bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)

                # Sample random camera location around the object
                location = bproc.sampler.shell(
                    center=obj.get_location(),
                    radius_min=2,
                    radius_max=10,
                    elevation_min=-90,
                    elevation_max=90
                )
                
                # Compute rotation based lookat point which is placed randomly around the object
                lookat_point = obj.get_location() + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
                rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
                
                # Add homog cam pose based on location an rotation
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

                # Only add camera pose if object is still visible
                if obj in bproc.camera.visible_objects(cam2world_matrix):
                    bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
                    poses += 1
                tries += 1

            bproc.renderer.set_max_amount_of_samples(100) # to speed up rendering, reduce the number of samples
            # Disable transparency so the background becomes transparent
            bproc.renderer.set_output_format(enable_transparency=True)
            # add segmentation masks (per class and per instance)
            bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

            # Render RGB images
            data = bproc.renderer.render()

            # Save annotations
            # bproc.writer.write_coco_annotations(
            #     output_dir=os.path.join(args.output_dir, 'coco_data'),
            #     instance_segmaps=data["instance_segmaps"],
            #     instance_attribute_maps=data["instance_attribute_maps"],
            #     colors=data["colors"],
            #     mask_encoding_format="polygon", - this was changed. hope it still works with models
            #     append_to_existing_output=True
            # ) - old one
            
            # suggested fix
            bproc.writer.write_coco_annotations(
            output_dir=os.path.join(args.output_dir, args.dataset_type),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            mask_encoding_format="rle",           # better as it's lighter, harder to understanf=d
            append_to_existing_output=True
            )
            
            # Where we’ll write the COCO keypoints file
            kp_out_dir = os.path.join(args.output_dir, args.dataset_type)
            os.makedirs(kp_out_dir, exist_ok=True)
            kp_json_path = os.path.join(kp_out_dir, 'coco_keypoints.json')

            # COCO scaffolding
            images = []
            annotations = []
            categories = []

            # Define your categories (from IDS)
            # Add keypoint names and a simple skeleton (optional)
            kp_names = ["bb_tl", "bb_tr", "bb_br", "bb_bl", "bb_center"]
            skeleton = [[1,2],[2,3],[3,4],[4,1],[1,5],[2,5],[3,5],[4,5]]  # edges between points (1-based)

            for name, cid in IDS.items():
                categories.append({
                    "id": cid,
                    "name": name,
                    "supercategory": "tool",
                    "keypoints": kp_names,
                    "skeleton": skeleton
                })

            ann_id = 1

            # data["colors"] is a list of images (one per frame)
            # We'll assume filenames are 000000.png, 000001.png, ... (BlenderProc’s default)
            for frame_idx, (rgba, seg) in enumerate(zip(data["colors"], data["instance_segmaps"])):
                h, w = seg.shape
                file_name = f"{frame_idx:06d}.png"

                # Register image entry
                images.append({
                    "id": frame_idx + 1,        # COCO image ids are 1-based
                    "width": w,
                    "height": h,
                    "file_name": file_name
                })

                # Look up per-instance attributes (like category_id) from the returned maps
                # BlenderProc gives a dict per frame keyed by attribute; fall back if missing
                attr_maps = data.get("instance_attribute_maps", [{}]*len(data["colors"]))
                frame_attrs = attr_maps[frame_idx] if frame_idx < len(attr_maps) else {}

                # Unique instance ids (ignore 0 = background)
                instance_ids = np.unique(seg)
                instance_ids = instance_ids[instance_ids != 0]

                for iid in instance_ids:
                    ys, xs = np.where(seg == iid)
                    if ys.size == 0:
                        continue

                    x0, y0 = int(xs.min()), int(ys.min())
                    x1, y1 = int(xs.max()), int(ys.max())
                    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)

                    # Keypoints from bbox
                    tl = (x0, y0)
                    tr = (x1, y0)
                    br = (x1, y1)
                    bl = (x0, y1)
                    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

                    # Visibility flag v: 2 = labeled & visible (COCO convention)
                    v = 2
                    keypoints_flat = [
                        tl[0], tl[1], v,
                        tr[0], tr[1], v,
                        br[0], br[1], v,
                        bl[0], bl[1], v,
                        cx,    cy,    v
                    ]

                    # Category: try to read from attribute map; fallback to a default if needed
                    # BlenderProc usually provides a per-instance "category_id" map with shape like the segmap.
                    # We’ll pick the category at the center pixel of the instance (or use a mode over the mask).
                    cat_id = None
                    if isinstance(frame_attrs, dict) and "category_id" in frame_attrs:
                        # frame_attrs["category_id"] is an array aligned with seg; sample within the instance
                        cat_map = frame_attrs["category_id"]
                        # sample at mask centroid (integer)
                        cyi, cxi = int(round(cy)), int(round(cx))
                        cyi = np.clip(cyi, 0, h-1); cxi = np.clip(cxi, 0, w-1)
                        cat_id = int(cat_map[cyi, cxi])
                    # Fallback: if missing, you can set a default or skip
                    if cat_id is None or cat_id == 0:
                        # If you rendered one object class per pass, you could infer from the subfolder name.
                        # Here we default to 1 to avoid breaking the format; adjust as needed.
                        cat_id = 1

                    annotations.append({
                        "id": ann_id,
                        "image_id": frame_idx + 1,
                        "category_id": cat_id,
                        "iscrowd": 0,
                        "area": int(bw * bh),
                        "bbox": [int(x0), int(y0), int(bw), int(bh)],   # keep normal COCO bbox too
                        "num_keypoints": len(kp_names),
                        "keypoints": keypoints_flat,
                        # segmentation can be omitted for a keypoint task; detectors/trainers accept this
                    })
                    ann_id += 1

            # Write COCO keypoints JSON
            with open(kp_json_path, "w") as f:
                json.dump({
                    "images": images,
                    "annotations": annotations,
                    "categories": categories
                }, f)

            print(f"Wrote COCO keypoints to: {kp_json_path}")
            bproc.object.delete_multiple(loaded, remove_all_offspring=True)

            # reset the internal camera pose list so poses don’t accumulate:
            CameraUtility._CameraUtility__cam2world_mats = []