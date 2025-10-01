import os

def get_all_urdf_files(urdf_folders: list[str]) -> list[str]:

    urdf_files = []

    for folder in urdf_folders:
        print(f"Processing folder: {folder}")
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".urdf"):
                    urdf_files.append(os.path.join(root, file))

    print(f"Found {len(urdf_files)} URDF files.")
    return urdf_files

def read_urdf_texts(urdf_files) -> list[str]:
    urdf_texts = []
    for urdf in urdf_files:
        with open(urdf, "r") as f:
            urdf_texts.append(f.read())
    return urdf_texts

def save_step_to_file(folder_num:int, step_num: int, filename: str, content: str):
    filename = os.path.basename(filename[:-5]) # remove .urdf
    folder_name = f"{folder_num:04d}_{filename}"
    os.makedirs(f"/workspace/data/output/{folder_name}", exist_ok=True)
    with open(f"/workspace/data/output/{folder_name}/{(step_num):02d}_step.txt", "w") as f:
        f.write(content)