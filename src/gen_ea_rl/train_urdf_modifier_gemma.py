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

from unsloth import FastModel, get_chat_template
import pandas as pd
from datasets import Dataset
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from gen_ea_rl.helpers.training_helpers import get_training_text
import os

os.environ["WANDB_API_KEY"] = ""


model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",
    max_seq_length=32768,  # Choose any for long context!
    load_in_4bit=False,  # 4 bit quantization to reduce memory
    load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
    load_in_16bit=True,
    full_finetuning=False,  # [NEW!] We have full finetuning now!
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
    r=512,  # Larger = higher accuracy, but might overfit
    lora_alpha=512,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=SEED,
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

data = []
df = pd.read_parquet("/workspace/data/output/training_data/anymal/anymal.parquet")
num_data = len(df)
# num_data = 21

for i in range(num_data):
    print(
        f"Robot: {df.at[i, 'robot']}. Step {df.at[i, 'randomization_step']:02d}. Task {df.at[i, 'task_number']:02d}. Progress: {i}/{len(df.index)} - {int(100*i/len(df.index))}%"
    )
    input_text, output_text = get_training_text(df, i)
    data.append(
        {
            "conversations": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]
        }
    )

dataset = Dataset.from_list(data)
dataset = standardize_data_formats(dataset, tokenizer=tokenizer)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=3,
        max_steps=150,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="wandb",  # Use TrackIO/WandB etc
    ),
)

print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part="<start_of_turn>user\n",
#     response_part="<start_of_turn>model\n",
# )

trainer_stats = trainer.train()

model.save_pretrained_merged("/models/gemma-3-1b-it-urdf", tokenizer)

model.save_pretrained("/models/gemma-3-1b-it-urdf-lora")
tokenizer.save_pretrained("/models/gemma-3-1b-it-urdf-lora")
