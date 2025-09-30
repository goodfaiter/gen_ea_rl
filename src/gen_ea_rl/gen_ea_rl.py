from transformers import pipeline
import torch
import os

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    batch_size=8,
    torch_dtype="auto",
    device_map="auto",
)

urdf_folders = [
    "/workspace/data/urdfs",
]
urdf_files = []
urdf_texts = []

for folder in urdf_folders:
    print(f"Processing folder: {folder}")
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".urdf"):
                urdf_files.append(os.path.join(root, file))

print(f"Found {len(urdf_files)} URDF files.")

for urdf in urdf_files:
    with open(urdf, "r") as f:
        urdf_texts.append(f.read())

print(f"Loaded {len(urdf_texts)} URDF texts.")
urdf_texts = urdf_texts[:5]  # limit to first 5 for testing

steps = [
    "define task the task the is used for", 
    # "define environment the robot is used in", 
    # "define robot type",
    # "define general kinematics",
    # "define exact kinematics using Denavit–Hartenberg parameters", 
    # "define the material parameters",
    # "define the sensors needed", 
    # "define the actuators parameters", 
]
N = len(steps)
for i, urdf_text in enumerate(urdf_texts):
    print(f"Processing URDF {i+1}/{len(urdf_texts)}")
    messages = []
    for j in range(len(steps)):
        msg = ""
        for k in range(N - j):
            msg += f"{k+1}. {steps[k]}\n"
        messages += {"role": "user", "content": f"Summarize with following steps:\n" + msg + f"the following URDF file :\n\n{urdf_text}"}
    
    outputs = pipe(messages, max_new_tokens=64)

    urdf_file_name = os.path.basename(urdf_files[i][:-5])
    os.makedirs(f"/workspace/data/output/{urdf_file_name}", exist_ok=True)
    for j in range(len(steps)):
        with open(f"/workspace/data/output/{urdf_file_name}/step_{N - j}.txt", "w") as f:
            f.write(outputs[j][0]["generated_text"][-1]["content"])

    torch.cuda.empty_cache()