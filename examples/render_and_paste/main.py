import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import bpy
import numpy as np
import argparse
import random
import os
import json
from colorsys import hsv_to_rgb

parser = argparse.ArgumentParser()
parser.add_argument('--obj_folder', default="/datashare/project/surgical_tools_models/", help="Path to folder with objects.")
# parser.add_argument('--obj', default="all", help="Path to folders with objects.")
parser.add_argument('--camera_params', default="camera.json", help="Camera intrinsics in json format")
parser.add_argument('--output_dir', default="", help="Path to where the final files, will be saved")
parser.add_argument('--num_images', type=int, default=25, help="Number of images to generate")
args = parser.parse_args()

bproc.init()

IDS = {"needle_holder" : 1, "tweezers" : 2} # TODO: make sure it aligns with Gal's part 2

for subfolder in os.listdir(args.obj_folder):
    full_path = os.path.join(args.obj_folder, subfolder)
    if os.path.isdir(full_path):
        print(f"entering the {subfolder}")
        for obj_name in os.listdir(full_path):
            print(f"creating images for {obj_name} on random backgrounds")
            obj_path = os.path.join(full_path, obj_name)
            obj = bproc.loader.load_obj(obj_path)[0]  # load the objects into the scene
            obj.set_cp("category_id", IDS[subfolder]) # TODO: check that I don't need to create subcategories for each type of object
            
            
            # Randomly perturbate the material of the object
            mat = obj.get_materials()[0] # needle holder metal color
            mat.set_principled_shader_value("Specular", random.uniform(0, 1))
            mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
            mat.set_principled_shader_value("Metallic", 1)
            mat.set_principled_shader_value("Roughness", 0.2)


mat = obj.get_materials()[1] # needle holder gold color
# random_gold_hsv_color = np.random.uniform([0.03, 0.95, 48], [0.25, 1.0, 48])
# random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0] # add alpha
# mat.set_principled_shader_value("Base Color", random_gold_color)
# mat.set_principled_shader_value("Specular", random.uniform(0, 1))
# mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
# mat.set_principled_shader_value("Metallic", 1)
# mat.set_principled_shader_value("Roughness", 0.2)

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

# Set camera intrinsics parameters
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

# Sample camera poses
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

# Write data to coco file
bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                        instance_segmaps=data["instance_segmaps"],
                        instance_attribute_maps=data["instance_attribute_maps"],
                        colors=data["colors"],
                        mask_encoding_format="polygon",
                        append_to_existing_output=True)
