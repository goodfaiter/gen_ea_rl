from yourdfpy import URDF
from gen_ea_rl.helpers.helpers import get_all_urdf_files, save_to_urdf
from lxml import etree

urdf_folders = ["/workspace/data/urdfs"]
urdf_files = get_all_urdf_files(urdf_folders)
# urdf_files = [urdf_files[0]]
urdf_files = urdf_files[0:10]
num_urdfs = len(urdf_files)
print(f"Loaded {num_urdfs} URDF texts.")

for urdf_index, urdf_file in enumerate(urdf_files):
    try:
        robot = URDF.load(urdf_file, load_meshes=False)
        robot.robot.name = "robot"
        link_to_old_name = dict()
        for j, (name, link) in enumerate(robot.link_map.items()):
            link_to_old_name[link.name] = f"{j:02d}_link"
            link.name = f"{j:02d}_link"
            link.collisions = [col for col in link.collisions if col.geometry.mesh is None]
            link.visuals = []
        for j, joint in enumerate(robot.joint_map.values()):
            joint.name = f"{j:02d}_joint"
            joint.parent = link_to_old_name[joint.parent]
            joint.child = link_to_old_name[joint.child]
            joint.calibration = None
            joint.safety_controller = None
        urdf_xml = robot.write_xml()
        urdf_text = etree.tostring(urdf_xml, xml_declaration=True, pretty_print=True, encoding='UTF-8').decode('utf-8')
        save_to_urdf(urdf_index, urdf_files[urdf_index], urdf_text)
    except Exception as e:
        print(f"Error processing {urdf_file}: {e}")