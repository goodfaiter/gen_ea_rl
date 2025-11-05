from transformers import AutoModelForCausalLM, AutoTokenizer
from openai_harmony import HarmonyEncodingName, load_harmony_encoding, Conversation, Message, Role, SystemContent, DeveloperContent
import torch 

class GptOss20B:
    model_name = "unsloth/gpt-oss-20b-GGUF"

    def __init__(self, max_new_tokens, reasoning, identity, instructions):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            # torch_dtype="auto",
            # dtype=torch.bfloat16,
            dtype="auto",
            # attn_implementation="sdpa",
            # attn_implementation="flash_attention_2",
            # attn_implementation="kernels-community/vllm-flash-attn3",
        )
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        self._system_msg = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new()
            # .with_required_channels(["final"])
            .with_reasoning_effort(reasoning)
            .with_model_identity(identity),
        )
        self._developer_msg = Message.from_role_and_content(
            Role.DEVELOPER, DeveloperContent.new().with_instructions(instructions)
        )

    def __call__(self, *args, **kwds):
        prompt = args[0]

        convo = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("Respond clearly")
            ),
            Message.from_role_and_content(Role.USER, prompt)
        ])     
        prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prefill_tensor = torch.tensor([prefill_ids], device=self.model.device)
        stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        outputs = self.model.generate(
            input_ids=prefill_tensor,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=stop_token_ids
        )
        # completion_ids = outputs[0][len(prefill_ids):]
        # entries = self.encoding.parse_messages_from_completion_tokens(prefill_ids, Role.ASSISTANT)
        response = self.tokenizer.decode(outputs[0])
        print(response)
        
        # return entries[-1].content[0].text if entries else ""
        return response
