import os
from gen_ea_rl.helpers.helpers import get_all_urdf_files, read_urdf_texts, save_step_to_file, save_to_file
from gen_ea_rl.helpers.robot_model import Robot, URDF
from gen_ea_rl.models.gpt_oss_20b_vllm import GptOss20BVLLM

output_path = "/workspace/data/output"
urdf_folders = [
    "/workspace/data/output/clean_urdfs/",
]

urdf_files = ["/workspace/data/output/clean_urdfs/anymal/anymal.urdf"]

# urdf_files = get_all_urdf_files(urdf_folders)
urdf_texts = read_urdf_texts(urdf_files)
num_urdfs = len(urdf_texts)
print(f"Loaded {num_urdfs} URDF texts.")

model = GptOss20BVLLM()

for i, urdf_text in enumerate(urdf_texts):
    print(f"Processing URDF {i+1}/{num_urdfs}: {urdf_files[i]}")
    robot = model(mode=Robot, prompt=str(urdf_text))
    # robot = model(mode=Robot, prompt="Remove 01_joint. Make sure the URDF is still consistent and all joints are connected to the parent link or removed.\n"+ str(urdf_text))
    robot = model(mode=Robot, prompt="Add a joint to 01_link. Make sure the URDF is still consistent and all joints are connected to the parent link or removed.\n"+ str(urdf_text))
    output_urdf = model(mode=URDF, prompt=str(robot))
    robot_name = os.path.basename(urdf_files[i])[:-5]
    save_to_file(f"{output_path}/{model.model_name}/{robot_name}/{robot_name}.{output_urdf.type.value}", output_urdf.xml_code)
