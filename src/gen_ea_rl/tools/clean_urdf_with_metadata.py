from yourdfpy import URDF
from gen_ea_rl.helpers.helpers import save_to_urdf, get_all_files_end_with
from lxml import etree
import os
import json

urdf_folders = ["/workspace/data/urdfs"]
json_files = get_all_files_end_with(urdf_folders, "meta-information.json")
num_jsons = len(json_files)
print(f"Loaded {num_jsons} JSON files.")

for file in json_files:
    with open(file) as f:
        meta_data = json.load(f)
        folder = os.path.split(file)[0]
        for robot in meta_data["robots"]:
            urdf_file = os.path.join(folder, robot["urdf"][0])
            print(f"Processing URDF: {urdf_file}")
            try:
                robot_model = URDF.load(urdf_file, load_meshes=False)
                for j, (name, link) in enumerate(robot_model.link_map.items()):
                    link.collisions = [col for col in link.collisions if col.geometry.mesh is None]
                    link.visuals = []
                for j, joint in enumerate(robot_model.joint_map.values()):
                    joint.calibration = None
                    joint.safety_controller = None
                urdf_xml = robot_model.write_xml()
                urdf_text = etree.tostring(urdf_xml, xml_declaration=True, pretty_print=True, encoding="UTF-8").decode("utf-8")
                metadata_comment = etree.Comment(json.dumps(robot, indent=2))
                urdf_text_with_metadata = str(metadata_comment) + "\n" + urdf_text
                save_to_urdf("clean_urdf_with_metadata", urdf_file, urdf_text_with_metadata)
            except Exception as e:
                print(f"Error processing {urdf_file}: {e}")
