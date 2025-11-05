from unsloth import FastLanguageModel

# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
#     "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
#     "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
#     "unsloth/gpt-oss-120b",
# ] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = None, # None for auto detection
    max_seq_length = 131072, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    # target_modules="all-linear",
    # target_parameters=[
    #     "7.mlp.experts.gate_up_proj",
    #     "7.mlp.experts.down_proj",
    #     "15.mlp.experts.gate_up_proj",
    #     "15.mlp.experts.down_proj",
    #     "23.mlp.experts.gate_up_proj",
    #     "23.mlp.experts.down_proj",
    # ],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = []
    for convo in convos:
        texts.append(
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="medium",
                # Keep model identity accroding to https://cookbook.openai.com/articles/openai-harmony
                # model_identity="You are an assistance AI designed to create, modify, analyze and work with Unified Robot Description Format URDF.",
            )
        )
    return {"text": texts}

from datasets import Dataset
from gen_ea_rl.helpers.helpers  import read_urdf_text, read_txt
import pandas as pd


data = []
df = pd.read_parquet("/workspace/data/output/training_data/anymal/anymal.parquet")
num_data = len(df)

for i in range(num_data):
    task = df.at[i, 'task']
    child_urdf = read_urdf_text(df.at[i, 'urdf_file']).decode(indent_level=2)
    parent_urdf = read_urdf_text(df.at[i, 'parent_urdf_file']).decode(indent_level=2)
    analysis = read_txt(df.at[i, 'analysis_file'])
    data.append({
        "messages": [
            {"role": "developer", "content": "Provide full robot .urdf file. Do not abbereviate. Never skip links or joints.", "thinking": ""},
            {"role": "user", "content": f"Given task: \"{task}\" and following URDF, add or remove link/joint to better perform the task.\nURDF:\n{child_urdf}", "thinking": ""},
            {"role": "assistant", "content": parent_urdf, "thinking": analysis},
            # {"role": "assistant", "content": "POTATO", "thinking": "POTATO"},
        ],
    })

dataset = Dataset.from_list(data)

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func, batched = True)

# dataset
# print(dataset[0]['text'][:1000])
# print('done')

from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        max_length = None, # prevent truncation of long URDFs
    ),
)

# print(trainer.train_dataset[0]["input_ids"])
# print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

from unsloth.chat_templates import train_on_responses_only

gpt_oss_kwargs = dict(instruction_part = "<|start|>user<|message|>", response_part="<|start|>assistant<|channel|>analysis<|message|>")
# gpt_oss_kwargs = dict(instruction_part = "<|start|>user<|message|>", response_part="<|start|>assistant<|channel|>final<|message|>")

trainer = train_on_responses_only(
    trainer,
    **gpt_oss_kwargs,
)

# print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]))
# print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

trainer_stats = trainer.train()
# trainer.train(resume_from_checkpoint = True)
# output_path = "/workspace/data/output"
# model.save_pretrained(output_path + "/finetuned_model")

model.save_pretrained_merged("/models/gpt-oss-20b-urdf", tokenizer, save_method = "mxfp4",)
