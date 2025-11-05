import os
from gen_ea_rl.helpers.helpers import read_step_texts, save_to_file
from gen_ea_rl.helpers.robot_model import Robot, URDF
from gen_ea_rl.models.gpt_oss_20b_vllm import GptOss20BVLLM

output_path = "/workspace/data/output"
steps_folders = [
    "/workspace/data/output/openai/gpt-oss-20b/atlas",
]


# steps = get_all_step_files(steps_folders)
step_files = ["/workspace/data/output/openai/gpt-oss-20b/atlas/07_step.txt"]
step_texts = read_step_texts(step_files)
num_texts = len(step_texts)
print(f"Loaded {num_texts} step texts.")

model = GptOss20BVLLM()

for i, text in enumerate(step_texts):
    print(f"Processing URDF {i+1}/{num_texts}: {step_files[i]}")
    robot = model(mode=Robot, prompt=str(text))
    output_urdf = model(mode=URDF, prompt=str(robot))
    folder_name = os.path.split(os.path.split(step_files[i])[-2])[-1] # get the folder name 
    file_name = os.path.basename(step_files[i])[:-4] # remove .txt
    save_to_file(f"{output_path}/{model.model_name}/{folder_name}/{file_name}.{output_urdf.type.value}", output_urdf.xml_code)
