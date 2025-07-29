import blenderproc as bproc
import os

# In this file we did basic overall analysis of the provided .obj files
# Including objects original positions and the materials 

obj_folder = "/datashare/project/surgical_tools_models"

bproc.init()
IDS = {"needle_holder" : 1, "tweezers" : 2} # TODO: make sure it aligns with Gal's part 2
features_set_NH = set()
features_set_Tweezers = set()


for subfolder in os.listdir(obj_folder):
    full_path = os.path.join(obj_folder, subfolder)
    if os.path.isdir(full_path):
        print(f"\n=== Entering category '{subfolder}' ===")

        for obj_name in os.listdir(full_path):
            if obj_name.endswith('.obj'):
                print(f"\n----- Loading: {obj_name} -----")
                obj_path = os.path.join(full_path, obj_name)

                obj = bproc.loader.load_obj(obj_path)[0]
                obj.set_cp("category_id", IDS[subfolder])

                # overall analysis of the object
                print(f"Object name: {obj.get_name()}")
                print("Object custom properties:")
                print(obj.get_all_cps())
                print(f"Object bounding box: {obj.get_bound_box()}")
                print(f"Object scale: {obj.get_scale()}")
                print(f"Object location: {obj.get_location()}")
                print(f"Object rotation: {obj.get_rotation_euler()}")

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
                        # print("Material Inputs:")
                        for inp in bsdf_node.inputs:
                            if subfolder == "needle_holder":
                                features_set_NH.add(inp.name)
                            else:
                                features_set_Tweezers.add(inp.name)
                            # print(f"- {inp.name}")
print("Materilals for needle holder: ", features_set_NH)
print("materials for Tweezers: ", features_set_Tweezers)
                    