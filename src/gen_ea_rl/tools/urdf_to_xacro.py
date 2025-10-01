from helpers.helpers import get_all_urdf_files, read_urdf_texts, save_step_to_file

urdf_folders = [
    "/workspace/data/urdfs",
]

urdf_files = get_all_urdf_files(urdf_folders)
urdf_texts = read_urdf_texts(urdf_files)
num_urdfs = len(urdf_texts)
print(f"Loaded {num_urdfs} URDF texts.")