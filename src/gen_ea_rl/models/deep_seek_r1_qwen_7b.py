from vllm import LLM, SamplingParams


class DeepSeekR1Qwen7B:
    # Instructions https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def __init__(self, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        self.model = LLM(self.model_name, gpu_memory_utilization=0.95, max_model_len=32768, enforce_eager=True)
        self.sampling_params = SamplingParams(temperature=0.6, max_tokens=max_new_tokens)

    def __call__(self, *args, **kwds):
        prompt = args[0]
        prompt += "\n\n Be concise. Be Exhaustive. Imagine you have to recreate a URDF from the notes you write down."
        prompt += "<think>\n"  # Instructions from https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
        output = self.model.generate(prompt, self.sampling_params)
        if output[0].outputs[0].finish_reason is not "stop":
            raise Exception('"Did not finish the response."')
        else:
            return output[0].outputs[0].text
