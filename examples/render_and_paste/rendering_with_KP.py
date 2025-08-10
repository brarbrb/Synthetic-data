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