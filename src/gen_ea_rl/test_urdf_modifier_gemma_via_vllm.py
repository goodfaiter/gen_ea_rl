from gen_ea_rl.helpers.robot_model import Modification
from gen_ea_rl.helpers.helpers import save_urdf_to_file
from gen_ea_rl.helpers.urdf_helpers import urdf_to_text, modify_urdf
from gen_ea_rl.helpers.training_helpers import get_training_text, generate_prompt
from gen_ea_rl.models.vllm import vllm
import pandas as pd
from yourdfpy import URDF

output_path = "/workspace/data/output"
df = pd.read_parquet("/workspace/data/output/training_data/anymal/anymal.parquet")
task = df.at[0, "task"]
max_step = df.at[0, "randomization_step"]
start_urdf_file = df.at[0, "urdf_file"]
urdf = URDF.load(start_urdf_file, load_meshes=False)
urdf_text = urdf_to_text(urdf)

model_name = "/models/gemma-3-1b-it-urdf"
model = vllm(model_name)
model.temperature = 1.0
model.top_p = 0.95
model.max_completion_tokens = 2048
# model.top_logprobs = 20

save_urdf_to_file(file_path=output_path + f"/{model_name}/{max_step:02d}_modified.urdf", urdf=urdf)
for j in range(0, max_step):
    input_text = generate_prompt(task=task, urdf_text=urdf_text)
    modification = model(prompt=input_text, mode=Modification)
    # modification = Modification.model_validate_json(df.at[j, "inverse_randomization_action"])
    modify_urdf(urdf, modification)
    save_urdf_to_file(file_path=output_path + f"/{model_name}/{max_step - 1 - j:02d}_modified.urdf", urdf=urdf)
    urdf_text = urdf_to_text(urdf)
    print(f"Modification step {j}:\n {modification}\n")
