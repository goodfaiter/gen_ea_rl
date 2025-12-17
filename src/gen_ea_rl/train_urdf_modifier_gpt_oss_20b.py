from unsloth import FastLanguageModel
import weave
from unsloth.chat_templates import standardize_sharegpt
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from gen_ea_rl.helpers.training_helpers import get_preamble, get_training_text
from gen_ea_rl.helpers.helpers import read_txt
import pandas as pd
import os
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort,
    ChannelConfig,
    RenderConversationConfig,
)

os.environ["WANDB_API_KEY"] = ""


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    dtype=None,  # None for auto detection
    max_seq_length=131072,  # Choose any for long context!
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    load_in_8bit=False,
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r=256,
    lora_alpha=256,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
    # target_modules=[
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    # ],
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = []
    system_message = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date("2025-11-06")
        .with_channel_config(ChannelConfig.require_channels(["analysis", "final"]))
    )
    for convo in convos:
        convo = convo[0]
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system_message),
                Message.from_role_and_content(Role.DEVELOPER, ""),
                # Message.from_role_and_content(Role.DEVELOPER, convo["developer"]),
                Message.from_role_and_content(Role.USER, convo["user"]),
                Message.from_role_and_content(Role.ASSISTANT, convo["analysis"]).with_channel("analysis"),
                Message.from_role_and_content(Role.ASSISTANT, convo["assistant"]).with_channel("final"),
            ]
        )
        tokens = encoding.render_conversation_for_training(convo, config=RenderConversationConfig(auto_drop_analysis=False))
        text = encoding.decode(tokens)
        texts.append(text)
    return {"text": texts}


data = []
df = pd.read_parquet("/workspace/data/output/training_data/anymal/anymal.parquet")
num_data = len(df)

for i in range(num_data):
    input_text, output_text = get_training_text(df, i)
    analysis = read_txt(df.at[i, "analysis_file"])
    data.append(
        {
            "messages": [
                # {"role": "developer", "content": "", "thinking": ""},
                # {"role": "developer", "content": prompt, "thinking": ""},
                # {"role": "user", "content": f"Given task: \"{task}\" and following URDF, provide a URDF with one link/joint modification.\nURDF:\n{child_urdf_text}", "thinking": ""},
                # {"role": "assistant", "content": parent_urdf_text, "thinking": analysis},
                # {"role": "user", "content": f"Given task: \"{task}\" and following URDF, provide a modification.\nURDF:\n{urdf_text}", "thinking": ""},
                # {"role": "assistant", "content": modification, "thinking": analysis},
                # {"role": "assistant", "content": modification, "thinking": ""},
                # {"role": "assistant", "content": "POTATO", "thinking": ""},
                # {"developer": prompt, "user": f"Given task: \"{task}\" and following URDF, provide a modification.\nURDF:\n{urdf_text}", "analysis": analysis, "assistant": modification}
                # {"developer": get_preamble(), "user": inpiut_text, "analysis": "", "assistant": output_text},
                {"developer": get_preamble(), "user": input_text, "analysis": analysis, "assistant": output_text},
            ],
        }
    )

dataset = Dataset.from_list(data)

dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        num_train_epochs=1,  # Set this for 1 full training run.
        max_steps=50,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",  # Use this for WandB etc
        max_length=None,  # prevent truncation of long URDFs
        completion_only_loss=True,
    ),
)

# print(trainer.train_dataset[0]["input_ids"])
# print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

# from unsloth.chat_templates import train_on_responses_only

# gpt_oss_kwargs = dict(instruction_part = "<|start|>user<|message|>", response_part="<|start|>assistant<|channel|>analysis<|message|>")
# gpt_oss_kwargs = dict(instruction_part="<|start|>user<|message|>", response_part="<|start|>assistant<|channel|>final<|message|>")

# trainer = train_on_responses_only(
#     trainer,
#     **gpt_oss_kwargs,
# )

# print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]))
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

trainer_stats = trainer.train()
# trainer.train(resume_from_checkpoint = True)
# output_path = "/workspace/data/output"
# model.save_pretrained(output_path + "/finetuned_model")

# model.save_pretrained_merged("/models/gpt-oss-20b-urdf", tokenizer)
model.save_pretrained_merged("/models/gpt-oss-20b-urdf", tokenizer, save_method="mxfp4")
