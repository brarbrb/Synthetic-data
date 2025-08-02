import blenderproc as bproc
import os

# In this file we did basic overall analysis of the provided .obj files
# Including objects original positions and the materials 

obj_folder = "/datashare/project/surgical_tools_models"

bproc.init()
CATEGORY_IDS  = {"needle_holder" : 1, "tweezers" : 2} # TODO: make sure it aligns with Gal's part 2
material_features = {
    "needle_holder": set(),
    "tweezers": set()
}

for category in os.listdir(obj_folder):
    full_path = os.path.join(obj_folder, category)
    if os.path.isdir(full_path):
        print(f"=== Entering category '{category}' ===")

        for obj_name in os.listdir(full_path):
            if obj_name.endswith('.obj'):
                print(f"----- Loading: {obj_name} -----")
                obj_path = os.path.join(full_path, obj_name)

                obj = bproc.loader.load_obj(obj_path)[0]
                obj.set_cp("category_id", CATEGORY_IDS [category])

                # overall analysis of the object
                print(f"Object name: {obj.get_name()}")
                print("Object custom properties:")
                print(obj.get_all_cps())
                print(f"Object bounding box: {obj.get_bound_box()}")
                print(f"Object scale: {obj.get_scale()}")
                print(f"Object location: {obj.get_location()}")
                print(f"Object rotation: {obj.get_rotation_euler()}")
                print(f"Object local to world matrix: {obj.get_local2world_mat()}")

                # Material analysis
                materials = obj.get_materials()
                print(f"Number of materials: {len(materials)}")
                for i, material in enumerate(materials):
                    print(f"  Material {i+1}:")
                    print(f"    Name: {material.get_name()}")    
                    bsdf_node = None
                    for node in material.blender_obj.node_tree.nodes:
                        # print(f"The nodeeee is: {node.type}")
                        if node.type == "BSDF_PRINCIPLED":
                            bsdf_node = node
                            break
                    if bsdf_node:
                        print("Material Inputs:")
                        for inp in bsdf_node.inputs:
                            if inp.name not in material_features[category]:
                                print(f"{inp.name} found first in object {obj_name}")
                            material_features[category].add(inp.name)
                            # print(f"- {inp.name}")


print("=== Summary of Material Inputs ===")
for category, features in material_features.items():
    print(f"{category.capitalize()}: {sorted(features)}")
    
if material_features['needle_holder'] == material_features['tweezers']:
    print("The objects have identical blender properties!")
else: 
    print("The ojects have some different properties")