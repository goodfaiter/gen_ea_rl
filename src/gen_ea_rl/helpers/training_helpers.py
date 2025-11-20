from gen_ea_rl.helpers.robot_model import Modification
from gen_ea_rl.helpers.urdf_helpers import urdf_to_text
from textwrap import dedent
from lxml import etree
import pandas as pd
from yourdfpy import URDF


def get_preamble() -> str:
    return dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {Modification.model_json_schema()}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )


def generate_prompt(task: str, urdf_text: str) -> str:
    # return f'{get_preamble()}\n\nGiven task: "{task}" and following URDF, provide a modification.\nURDF:\n{urdf_text}'
    return f'Given task: "{task}" and following URDF, provide a modification.\nURDF:\n{urdf_text}'


def get_training_text_input(df: pd.DataFrame, index: int) -> str:
    task = df.at[index, "task"]
    urdf = URDF.load(df.at[index, "urdf_file"], load_meshes=False)
    urdf_text = urdf_to_text(urdf)  
    # urdf_xml = urdf.write_xml()
    # urdf_text = etree.tostring(urdf_xml, xml_declaration=True, pretty_print=False, encoding="UTF-8").decode("utf-8")
    return generate_prompt(task, urdf_text)


def get_training_text_output(df: pd.DataFrame, index: int) -> str:
    return df.at[index, "inverse_randomization_action"]


def get_training_text(df: pd.DataFrame, index: int) -> tuple[str, str]:
    input_text = get_training_text_input(df, index)
    output_text = get_training_text_output(df, index)
    return input_text, output_text
