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
# parser.add_argument('--obj', default="surgical_tools_models/needle_holder/NH1.obj", help="Path to the object file.")
parser.add_argument('--obj_name', default="needle_holder", help="First object name.")
parser.add_argument('--obj', default="/datashare/project/surgical_tools_models/needle_holder/NH1.obj", help="Path to the first object file.")
parser.add_argument('--obj_name2', default="tweezers", help="Second object name.")
parser.add_argument('--obj2', default="/datashare/project/surgical_tools_models/tweezers/T1.obj", help="Path to the second object file.")
parser.add_argument('--camera_params', default="camera.json", help="Camera intrinsics in json format")
parser.add_argument('--output_dir', default="output/coco_synthetic", help="Path to where the final files, will be saved")
parser.add_argument('--num_images', type=int, default=25, help="Number of images to generate")
args = parser.parse_args()

bproc.init()

'''# load the objects into the scene
obj = bproc.loader.load_obj(args.obj)[0]
obj.set_cp("category_id", 1)

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
# mat.set_principled_shader_value("Roughness", 0.2)'''

# Load the first object into the scene
obj1 = bproc.loader.load_obj(args.obj)[0]
obj1.set_cp("category_id", 1)  # First object category id

if args.obj_name2 != None:
    # Load the second object into the scene
    obj2 = bproc.loader.load_obj(args.obj2)[0]
    obj2.set_cp("category_id", 2)  # Second object category id

# Randomly perturbate materials for object1 (needle holder)
mat1 = obj1.get_materials()[0]
mat1.set_principled_shader_value("Specular Tint", tuple(random.uniform(0, 1) for _ in range(4)))
mat1.set_principled_shader_value("Roughness", random.uniform(0, 1))
mat1.set_principled_shader_value("IOR", random.uniform(0,1)) 
mat1.set_principled_shader_value("Anisotropic", random.uniform(-1,1)) 
mat1.set_principled_shader_value("Metallic", 1)
mat1.set_principled_shader_value("Roughness", 0.2)

if args.obj_name == 'needle_holder':
    mat1 = obj1.get_materials()[1]  # needle holder gold color
    random_gold_hsv_color = np.random.uniform([0.03, 0.95, 48], [0.25, 1.0, 48])
    random_gold_color = list(hsv_to_rgb(*random_gold_hsv_color)) + [1.0]  # add alpha
    mat1.set_principled_shader_value("Base Color", random_gold_color)
    mat1.set_principled_shader_value("Specular Tint", tuple(random.uniform(0, 1) for _ in range(4)))
    mat1.set_principled_shader_value("Roughness", random.uniform(0, 1))
    mat1.set_principled_shader_value("IOR", random.uniform(0,1)) 
    mat1.set_principled_shader_value("Anisotropic", random.uniform(-1,1)) 
    mat1.set_principled_shader_value("Metallic", 1)
    mat1.set_principled_shader_value("Roughness", 0.2)

if args.obj_name2 != None:
    # Randomly perturbate materials for object2 (tweezers)
    mat2 = obj2.get_materials()[0]  # Tweezers material
    mat2.set_principled_shader_value("Specular Tint", tuple(random.uniform(0, 1) for _ in range(4)))
    mat2.set_principled_shader_value("Roughness", random.uniform(0, 1))
    mat2.set_principled_shader_value("IOR", random.uniform(0,1)) 
    mat2.set_principled_shader_value("Anisotropic", random.uniform(-1,1)) 
    mat2.set_principled_shader_value("Metallic", 1)
    mat2.set_principled_shader_value("Roughness", 0.2)

    # Position objects in the scene
    # You can add some random offset to their positions to avoid overlap
    obj1.set_location([0, 0, 0])  # Example position for the needle holder
    obj2.set_location([0.5, 0.5, 0])  # Example position for the tweezers

# Create a new light
light = bproc.types.Light()
light.set_type("POINT")
# Sample its location around the object
light.set_location(bproc.sampler.shell(
    center=obj1.get_location(),
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
while tries < 10000 and poses < 1:

    # Set a random world lighting strength
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = np.random.uniform(0.1, 1.5)

    # Sample random camera location around the object
    location = bproc.sampler.shell(
        center=obj1.get_location(),
        radius_min=2,
        radius_max=10,
        elevation_min=-90,
        elevation_max=90
    )
    # Compute rotation based lookat point which is placed randomly around the object
    lookat_point = obj1.get_location() + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(lookat_point - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Only add camera pose if object is still visible
    if args.obj_name2 != None:
        if obj1 in bproc.camera.visible_objects(cam2world_matrix) and obj2 in bproc.camera.visible_objects(cam2world_matrix):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
            poses += 1
        tries += 1
        print(tries)
    elif args.obj_name2 == None:
        if obj1 in bproc.camera.visible_objects(cam2world_matrix):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=poses)
            poses += 1
        tries += 1
        print(tries)
    tries += 1

bproc.renderer.set_max_amount_of_samples(100) # to speed up rendering, reduce the number of samples
# Disable transparency so the background becomes transparent
bproc.renderer.set_output_format(enable_transparency=True)
# add segmentation masks (per class and per instance)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# Render RGB images
data = bproc.renderer.render()

# Write data to coco file
valid_segmaps = []
valid_attrs = []
valid_colors = []

for i in range(len(data["instance_segmaps"])):
    mask = data["instance_segmaps"][i]
    attr = data["instance_attribute_maps"][i]
    color = data["colors"][i]

    # Check if mask is a 2D NumPy array and has at least one object
    if isinstance(mask, np.ndarray) and mask.ndim == 2 and np.any(mask > 0):
        valid_segmaps.append(mask)
        valid_attrs.append(attr)
        valid_colors.append(color)

# Then call writer
bproc.writer.write_coco_annotations(
    output_dir=os.path.join(args.output_dir, 'coco_data'),
    instance_segmaps=valid_segmaps,
    instance_attribute_maps=valid_attrs,
    colors=valid_colors,
    mask_encoding_format="polygon",
    append_to_existing_output=True
)