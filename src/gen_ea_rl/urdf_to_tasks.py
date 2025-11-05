import os
from gen_ea_rl.helpers.helpers import get_all_urdf_files, read_urdf_texts, save_to_file
from gen_ea_rl.helpers.robot_model import Tasks
from gen_ea_rl.models.gpt_oss_20b_vllm import GptOss20BVLLM
import pandas as pd

output_path = "/workspace/data/output"
urdf_folders = [
    "/workspace/data/output/clean_urdf_with_metadata/anymal",
]

urdf_files = get_all_urdf_files(urdf_folders)
urdf_texts = read_urdf_texts(urdf_files)
num_urdfs = len(urdf_texts)
print(f"Loaded {num_urdfs} URDF texts.")

model = GptOss20BVLLM()

for i, urdf_text in enumerate(urdf_texts):
    print(f"Processing URDF {i+1}/{num_urdfs}: {urdf_files[i]}")
    tasks = model(mode=Tasks, prompt=str(urdf_text))
    robot_name = os.path.basename(urdf_files[i])[:-5]
    df = pd.read_parquet(f"{output_path}/training_data/{robot_name}/{robot_name}.parquet")
    df['task'] = None
    for i, task in enumerate(tasks.task_examples):
        df.loc[(df['robot'] == robot_name) & (df['task_number'] == i), 'task'] = task
    df.to_parquet(f"{output_path}/training_data/{robot_name}/{robot_name}.parquet")
    save_to_file(f"{output_path}/{model.model_name}/{robot_name}/{robot_name}_tasks.txt", str(tasks))
