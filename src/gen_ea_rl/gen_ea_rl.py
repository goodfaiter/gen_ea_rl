import torch
from helpers.helpers import get_all_urdf_files, read_urdf_texts, save_step_to_file
from models.gpt_oss_20b import GptOss20B

# Typica settings for large model pytorch
torch.cuda.set_per_process_memory_fraction(1.0, 0)
torch.cuda.empty_cache()

urdf_folders = [
    "/workspace/data/urdfs",
]

urdf_files = get_all_urdf_files(urdf_folders)
urdf_texts = read_urdf_texts(urdf_files)
num_urdfs = len(urdf_texts)
print(f"Loaded {num_urdfs} URDF texts.")
# urdf_texts = urdf_texts[:2]  # limit to first 5 for testing
urdf_texts = [urdf_texts[1]]  # limit to first 5 for testing

steps = [
    "define a one simple sentence description such as 'robot that...'",
    "define task the task robot is used for",
    "define environment the robot is used in",
    "define robot type",
    "define general kinematics",
    "define exact kinematics using Denavit–Hartenberg parameters",
    "define the material parameters",
    "define the sensors needed",
    "define the actuators parameters",
]
num_steps = len(steps)

model = GptOss20B()

for i, urdf_text in enumerate(urdf_texts):
    print(f"Processing URDF {i+1}/{num_urdfs}: {urdf_files[i]}")

    for j in range(num_steps):
        print(f"Processing step {num_steps-j}/{num_steps}")

        msg = ""
        for k in range(num_steps - j):
            msg += f"{k+1}. {steps[k]}\n"

        response_text = model(f"Summarize with following steps:\n" + msg + f"the following URDF file :\n\n{urdf_text}")

        save_step_to_file(i, num_steps - j, urdf_files[i], response_text)

        torch.cuda.empty_cache()