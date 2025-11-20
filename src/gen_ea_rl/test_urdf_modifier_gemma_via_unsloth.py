import unsloth

import random
import numpy as np
import torch

SEED = 200964

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pandas as pd
from gen_ea_rl.helpers.training_helpers import get_training_text, generate_prompt
from gen_ea_rl.helpers.urdf_helpers import urdf_to_text
from gen_ea_rl.helpers.randomizer_helpers import modify_urdf
from gen_ea_rl.helpers.helpers import save_urdf_to_file
from gen_ea_rl.helpers.robot_model import Modification
from yourdfpy import URDF
from transformers import TextStreamer, AutoConfig
from unsloth import FastModel

model_name = "/models/gemma-3-1b-it-urdf"
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=32768,
    load_in_4bit=False,
    load_in_16bit=True,
)

df = pd.read_parquet("/workspace/data/output/training_data/anymal/anymal.parquet")
i = 0
task = df.at[i, "task"]
start_urdf_file = df.at[i, "urdf_file"]
robot = URDF.load(start_urdf_file, load_meshes=False)
urdf_text = urdf_to_text(robot)
num_data = len(df)
output_path = "/workspace/data/output"
max_step = df.at[0, "randomization_step"]

temperature = 1.0
top_p = 0.95
top_k = 64
max_new_tokens = 2048


def input_text_to_ids(input_text: str):
    messages = [{"role": "user", "content": [{"type": "text", "text": input_text}]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs


def generate_example(i):
    input_text, output_text = get_training_text(df, i)
    inputs = input_text_to_ids(input_text)
    return inputs, output_text


#### TESTING TEXT GENERATION W/O GRAMMAR CONSTRAINTS ####
# for i in range(num_data):
#     print(f"\n=== Generation {i+1} ===")
#     inputs, output_text = generate_example(i)
#     output = model.generate(
#         **inputs.to("cuda"),
#         max_new_tokens=max_new_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         # streamer=TextStreamer(tokenizer, skip_prompt=True),
#     )
#     print("=== Output ===")
#     generated_text = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
#     print(generated_text)

#     print("\n=== Expected Output ===")
#     print(output_text)

#     print("\n === Match? ===")
#     match = generated_text == output_text.strip()
#     print("Match!" if match else "No match.")


#### TESTING URDF GENERATION W/O GRAMMAR CONSTRAINTS ####
# for i in range(max_step):
#     input_text = generate_prompt(task=task, urdf_text=urdf_text)
#     messages = [{"role": "user", "content": [{"type": "text", "text": input_text}]}]
#     inputs = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True  # Must add for generation
#     )
#     output = model.generate(
#         **inputs.to("cuda"), max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, #streamer=TextStreamer(tokenizer, skip_prompt=True)
#     )
#     # read from </robot>\nmodel\n
#     output = output[:, inputs["input_ids"].shape[1] :]
#     modification = Modification.model_validate_json(tokenizer.decode(output[0], skip_special_tokens=True).strip())
#     modify_urdf(urdf, modification)
#     save_urdf_to_file(file_path=output_path + f"/{model_name}/{max_step - 1 - i:02d}_modified.urdf", urdf=urdf)
#     urdf_text = urdf_to_text(urdf)
#     print(f"Modification step {i}:\n {modification}\n")


#### TESTING URDF GENERATION WITH GRAMMAR CONSTRAINTS ####
import xgrammar as xgr

config = AutoConfig.from_pretrained(model_name)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = grammar_compiler.compile_json_schema(Modification.model_json_schema(), strict_mode=False)
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

for i in range(max_step):
    print(f"\n=== Generation {i+1} ===")
    input_text_original, output_text = get_training_text(df, i)
    input_text = generate_prompt(task=task, urdf_text=urdf_text)
    print("=== Input ===")
    match = input_text_original.strip() == input_text.strip()
    print("Match!" if match else "No match.")
    if not match:
        print("\nGenerated Input:")
        print(input_text.strip())
        print("Original Input:")
        print(input_text_original.strip())
    inputs = input_text_to_ids(input_text)
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    output = model.generate(
        **inputs.to("cuda"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logits_processor=[xgr_logits_processor],
        # streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    output = output[0, inputs["input_ids"].shape[1] :]

    print("=== Output ===")
    generated_text = tokenizer.decode(output, skip_special_tokens=True).strip()
    print(generated_text)

    print("\n=== Expected Output ===")
    print(output_text)

    print("\n === Match? ===")
    match = generated_text == output_text.strip()
    print("Match!" if match else "No match.")

    modification = Modification.model_validate_json(tokenizer.decode(output, skip_special_tokens=True).strip())
    modify_urdf(robot, modification)
    save_urdf_to_file(file_path=output_path + f"/{model_name}/{max_step - 1 - i:02d}_modified.urdf", urdf=robot)
    urdf_text = urdf_to_text(robot)
    # print(f"Modification step {i}:\n {modification}\n")
