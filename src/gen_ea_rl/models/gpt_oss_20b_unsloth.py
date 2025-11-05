from unsloth import FastLanguageModel, GptOssForCausalLM, AutoTokenizer
from transformers import TextStreamer

class GptOss20BUnsloth:
    model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"

    def __init__(self, max_new_tokens, reasoning, identity, instructions):

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            dtype=None,  # None for auto detection
            max_seq_length=10000,  # Choose any for long context!
            load_in_4bit=True,  # 4 bit quantization to reduce memory
            full_finetuning=False,  # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )

        self.max_new_tokens = max_new_tokens
        self.reasoning = reasoning
        self.identity = identity
        self.developer_instructions = instructions

    def __call__(self, *args, **kwds):
        prompt = args[0]

        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            developer_instructions=self.developer_instructions,
            model_identity=self.identity,
            reasoning_effort=self.reasoning,  # **NEW!** Set reasoning effort to low, medium or high
        ).to("cuda")

        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, streamer=TextStreamer(self.tokenizer))
        return self.tokenizer.decode(output[0]).split("<|start|>assistant<|channel|>final<|message|>")[1].split("<|return|>")[0]
