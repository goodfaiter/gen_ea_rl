import os
from gen_ea_rl.helpers.helpers import get_all_files_end_with, read_urdf_text, save_to_file
from gen_ea_rl.helpers.robot_model import UrdfAnalysis
from gen_ea_rl.models.gpt_oss_20b_vllm import GptOss20BVLLM
import pandas as pd


output_path = "/workspace/data/output"

folders = [
    "/workspace/data/output/training_data/anymal",
]

training_files = get_all_files_end_with(folders, end_with=".parquet")
num_files = len(training_files)
print(f"Loaded {num_files} parquet files.")

model = GptOss20BVLLM()

for i, file in enumerate(training_files):
    print(f"Processing {i+1}/{num_files}: {training_files[i]}")
    df = pd.read_parquet(file)
    df['analysis_file'] = None
    for j in df.index:
        task_number = df.at[j, 'task_number']
        randomization_step = df.at[j, 'randomization_step']
        robot_name = df.at[j, 'robot']
        child_urdf_file = read_urdf_text(df.at[j, 'urdf_file'])
        parent_urdf_file = read_urdf_text(df.at[j, 'parent_urdf_file'])
        inverse_randomization_action = df.at[j, 'inverse_randomization_action']
        task = df.at[j, 'task']
        prompt_text = f"Task: \"{task}\".\n\nInput URDF:\n{child_urdf_file}\n\nGiven action \"{inverse_randomization_action}\", output URDF:\n{parent_urdf_file}"
        response = model(mode=UrdfAnalysis, prompt=prompt_text)
        file_name = f"{output_path}/{model.model_name}/{robot_name}/{task_number:02d}_{randomization_step:02d}_analysis.txt"
        df.at[j, 'analysis_file'] = file_name
        save_to_file(file_name, response.chain_of_thought)
    df.to_parquet(file)
