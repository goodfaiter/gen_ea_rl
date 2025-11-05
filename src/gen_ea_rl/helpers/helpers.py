import os
from bs4 import BeautifulSoup


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


def get_all_files_end_with(folders: list[str], end_with: str) -> list[str]:

    ends_with_files = []

    for folder in folders:
        print(f"Processing folder: {folder}")
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(end_with):
                    ends_with_files.append(os.path.join(root, file))

    print(f"Found {len(ends_with_files)} end_with files.")
    return ends_with_files


def get_all_step_files(folders: list[str]) -> list[str]:

    step_files = []

    for folder in folders:
        print(f"Processing folder: {folder}")
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith("_step.txt"):
                    step_files.append(os.path.join(root, file))

    print(f"Found {len(step_files)} step files.")
    return step_files


def read_txt(file) -> str:
    with open(file, "r") as f:
        return f.read()


def read_step_texts(step_files) -> list[str]:
    step_texts = []
    for step in step_files:
        with open(step, "r") as f:
            text = f.read()
            step_texts.append(text)
    return step_texts


def read_urdf_text(urdf_file: str) -> str:
    with open(urdf_file, "r") as f:
        text = f.read()
        urdf_text = BeautifulSoup(text, "xml")
    return urdf_text


def read_urdf_texts(urdf_files) -> list[str]:
    urdf_texts = []
    for urdf in urdf_files:
        urdf_texts.append(read_urdf_text(urdf))
    return urdf_texts


def save_to_file(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)


def save_step_to_file(step_num: int, filename: str, content: str, suffix: str = ".txt"):
    filename = os.path.basename(filename[:-5])  # remove .urdf
    folder_name = f"{filename}"
    os.makedirs(f"/workspace/data/output/{folder_name}", exist_ok=True)
    with open(f"/workspace/data/output/{folder_name}/{(step_num):02d}_step" + suffix, "w") as f:
        f.write(content)


def save_to_urdf(foldername:str, filename: str, content: str):
    filename = os.path.basename(filename[:-5])  # remove .urdf
    folder_name = f"{filename}"
    os.makedirs(f"/workspace/data/output/{foldername}/{folder_name}", exist_ok=True)
    with open(f"/workspace/data/output/{foldername}/{folder_name}/{(filename)}.urdf", "w") as f:
        f.write(content)
