import blenderproc as bproc
import numpy as np
import json

bproc.init()
obj_folder = "/datashare/project/surgical_tools_models/tweezers/"

# loading the object
obj = bproc.loader.load_obj(obj_folder + "T4.obj")[0]

# Get transformation matrix from object (object to world)
obj2world_matrix = obj.get_local2world_mat() # have no idea what is that

# # Get vertices in world coordinates
# vertices_world = obj.get_mesh_vertices() @ obj2world_matrix[:3, :3].T + obj2world_matrix[:3, 3]

bbox_corners = np.array(obj.get_bound_box())  # Get the bounding box corners in local coordinates
bbox_corners_h = np.concatenate([bbox_corners, np.ones((8, 1))], axis=1)  # Convert bbox corners to homogeneous coordinates

# Transform to world coordinates
bbox_world = (obj2world_matrix @ bbox_corners_h.T).T[:, :3]
# print("Bounding box corners in world coordinates:", bbox_world)


bproc.camera.add_camera_pose(np.eye(4))  # or your actual pose
bproc.camera.set_intrinsics_from_blender_params(
    lens=1.047,                # FOV in radians (example: 60 degrees ≈ 1.047 radians)
    image_width=640,
    image_height=480,
    lens_unit="FOV"
)
# camera_params = None
# with open('camera.json', "r") as file:
#     camera_params = json.load(file)

# fx = camera_params["fx"]
# fy = camera_params["fy"]
# cx = camera_params["cx"]
# cy = camera_params["cy"]
# im_width = camera_params["width"]
# im_height = camera_params["height"]
# K = np.array([[fx, 0, cx], 
#               [0, fy, cy], 
#               [0, 0, 1]])
# CameraUtility.set_intrinsics_from_K_matrix(K, im_width, im_height) 

projected_2d = []
for corner in bbox_world:
    uv = bproc.camera.project_points(np.array([corner]))  # Pass as shape (1, 3)
    if uv is not None:
        projected_2d.append((uv[0][0], uv[0][1]))  # Unpack first element

print("Projected 2D coordinates of bounding box corners:")
for i, (u, v) in enumerate(projected_2d):
    print(f"Corner {i}: (u={u:.2f}, v={v:.2f})")
# Note: The projected_2d will contain the 2D coordinates of the bounding box corners in the image plane.
# You can use these coordinates for further processing or visualization.

with open("output/pose_annotation.json", "w") as f:
    json.dump({"bbox_2d": projected_2d}, f)