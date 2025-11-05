from yourdfpy import URDF
from gen_ea_rl.helpers.helpers import get_all_urdf_files, save_to_file
from gen_ea_rl.helpers.urdf_helpers import remove_link, remove_joint, add_link, generate_link, randomize_urdf, reorder_urdf
from gen_ea_rl.helpers.robot_model import RandmoizationStep, inverse_randomization_step
from lxml import etree
import os
import numpy as np
import pandas as pd


def save(robot: URDF, urdf_file: str, task_iteration:int, randomization_iteration: int, df: pd.DataFrame, random_step: RandmoizationStep):
    # reorder and update link name in the randomization step
    # reorder_urdf(robot)
    # random_step.link = link_to_old_name[random_step.link]
    urdf_xml = robot.write_xml()
    urdf_text = etree.tostring(urdf_xml, xml_declaration=True, pretty_print=True, encoding="UTF-8").decode("utf-8")
    robot_name = os.path.basename(urdf_file[:-5])
    if randomization_iteration <= 1:
        parent_urdf_file = urdf_file
    else:
        parent_urdf_file = output_path + f"/randomized_urdf/{robot_name}/{task_iteration:02d}_{randomization_iteration-1:02d}_{robot_name}_randomized.urdf"
    file_name = output_path + f"/randomized_urdf/{robot_name}/{task_iteration:02d}_{randomization_iteration:02d}_{robot_name}_randomized.urdf"
    df.loc[len(df)] = {
        "original_urdf": urdf_file,
        "robot": robot_name,
        "task_number": task_iteration,
        "randomization_step": randomization_iteration,
        "urdf_file": file_name,
        "parent_urdf_file": parent_urdf_file,
        "inverse_randomization_action": random_step.model_dump_json(),
    }
    save_to_file(file_name, urdf_text)

output_path = "/workspace/data/output"
urdf_folders = ["/workspace/data/output/clean_urdfs/anymal"]
# urdf_folders = ["/workspace/data/output/clean_urdfs/anymal_test"]
urdf_files = get_all_urdf_files(urdf_folders)
num_urdfs = len(urdf_files)
print(f"Found {num_urdfs} URDFs.")

chance_to_add = 0.5
df = pd.DataFrame(columns=["robot", "original_urdf", "task_number", "randomization_step", "urdf_file", "parent_urdf_file", "inverse_randomization_action"])

for urdf_file in urdf_files:
    task_num = 0
    robot = URDF.load(urdf_file, load_meshes=False)
    
    for i in range(1, 5):
        robot, random_step = randomize_urdf(robot, chance_to_add)
        inverse_randomization_step(random_step)
        save(robot, urdf_file, task_num, i, df, random_step)

    robot_name = os.path.basename(urdf_file[:-5])
    os.makedirs(os.path.dirname(output_path + '/training_data/' + robot_name), exist_ok=True)
    df.to_parquet(output_path + f"/training_data/{robot_name}/{robot_name}.parquet")
