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
parser.add_argument('--obj_folder', default="/datashare/project/surgical_tools_models/", help="Path to folder with objects.")
parser.add_argument('--camera_params', default="camera.json", help="Camera intrinsics in json format")
parser.add_argument('--output_dir', default="output", help="Path to where the final files, will be saved")
# parser.add_argument('--num_images', type=int, default = 25, help="Number of images to generate") # for train
parser.add_argument('--num_images', type=int, default = 1, help="Number of images to generate") # val set
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
            obj = bproc.loader.load_obj(obj_path)[0]  # load the objects into the scene
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
            while tries < 10000 and poses < args.num_images:

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
            output_dir=os.path.join(args.output_dir, 'coco_data'),
            instance_segmaps=data["instance_segmaps"],
            instance_attribute_maps=data["instance_attribute_maps"],
            colors=data["colors"],
            mask_encoding_format="rle",           # ← change from "polygon" to "rle"
            append_to_existing_output=True
            )