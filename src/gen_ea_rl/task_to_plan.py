from enum import Enum
from typing import Literal, Union
from gen_ea_rl.models.vllm import vllm
from pydantic import BaseModel, Field


class Potato(BaseModel):
    name: Literal["potato"] = "potato"


class Knife(BaseModel):
    name: Literal["knife"] = "knife"


class Peeler(BaseModel):
    name: Literal["peeler"] = "peeler"


class Move(BaseModel):
    name: Literal["move"] = "move"
    object: Union[Potato, Knife, Peeler] = Field(description="Object to move.")


class Hold(BaseModel):
    name: Literal["hold"] = "hold"
    object: Union[Potato, Knife, Peeler] = Field(description="Object to hold.")


class Rotate(BaseModel):
    name: Literal["rotate"] = "rotate"
    object: Union[Potato, Knife, Peeler] = Field(description="Object to rotate.")


class Peel(BaseModel):
    name: Literal["peel"] = "peel"
    object: Union[Potato] = Field(description="Object to peel.")


class Cut(BaseModel):
    name: Literal["cut"] = "cut"
    object: Union[Potato] = Field(description="Object to cut.")    


class Action(BaseModel):
    description: str = Field(description="Description of the action to be performed.")
    action: Union[Move, Hold, Rotate, Peel, Cut] = Field(description="Action to perform.")


class TaskPlan(BaseModel):
    task: str
    plan: list[Action] = Field(description="Step by step plan for the robot to execute the task.")


class HighLevelPlan(BaseModel):
    task: str
    plan: list[str] = Field(description="High level step by step plan for this task.")


model = vllm(model_name="openai/gpt-oss-20b")
model.reasoning_effort = "medium"

task = "Peel a potato"

prompt_text = f"Given a task: '{task}', provide a step-by-step plan to perform it. Be short and clear in your instructions."
response = model(mode=HighLevelPlan, prompt=prompt_text)

print(response.task)
for step in response.plan:
    print(f"- Step: {step}")
    prompt_text = f"Given the step: '{step}', provide concise plan. Be short and clear in your instructions. Your options are : Move, Hold, Rotate, Peel, Cut. Your objects are : Potato, Knife, Peeler."
    actions = model(mode=TaskPlan, prompt=prompt_text)
    for action in actions.plan:
        print(f"   - {action.action}, Description: {action.description}")


# (.venv) root@graniel:/workspace#  cd /workspace ; /usr/bin/env /.venv/bin/python /root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 45623 -- /workspace/src/gen_ea_rl/task_to_plan.py 
# Peel and cut a potato
# - Step: 1. Gather tools: potato, wash bowl, peeler, sharp knife, cutting board, knife safety gloves (optional).
#    - name='move' object=Potato(name='potato'), Description: Move potato to the cutting board
#    - name='hold' object=Potato(name='potato'), Description: Hold the potato (gloves optional)
#    - name='rotate' object=Potato(name='potato'), Description: Rotate the potato for a better angle
#    - name='peel' object=Potato(name='potato'), Description: Peel the potato with the peeler
#    - name='cut' object=Potato(name='potato'), Description: Cut the potato with the knife
# - Step: 2. Rinse the potato under cold running water to remove dirt.
#    - name='move' object=Potato(name='potato'), Description: Move the potato to the sink or basin
#    - name='hold' object=Potato(name='potato'), Description: Hold the potato while it is in the water
#    - name='rotate' object=Potato(name='potato'), Description: Rotate the potato to clean all sides under the running water
# - Step: 3. Pat dry. Use a vegetable peeler to remove the skin in a continuous motion from top to bottom.
#    - name='move' object=Potato(name='potato'), Description: Hold the potato steady
#    - name='move' object=Peeler(name='peeler'), Description: Hold the peeler
#    - name='move' object=Peeler(name='peeler'), Description: Move the peeler from top to bottom in a continuous stroke
#    - name='peel' object=Potato(name='potato'), Description: Remove the skin
# - Step: 4. Place the peeled potato on the cutting board.
#    - name='move' object=Potato(name='potato'), Description: Move Potato to cutting board.
# - Step: 5. Cut the potato into the desired shape (e.g., slice, cube, or wedges).
#    - name='move' object=Potato(name='potato'), Description: Hold Potato
#    - name='move' object=Knife(name='knife'), Description: Hold Knife
#    - name='move' object=Potato(name='potato'), Description: Cut Potato into desired shape
# - Step: 6. Gather or discard any excess skin, and wash the cut pieces if needed before cooking.
#    - name='hold' object=Potato(name='potato'), Description: Hold the potato
#    - name='peel' object=Potato(name='potato'), Description: Peel any remaining excess skin
#    - name='cut' object=Potato(name='potato'), Description: Cut the potato into desired pieces
#    - name='move' object=Potato(name='potato'), Description: Move the cut pieces to a bowl or sink for washing

# (.venv) root@graniel:/workspace#  cd /workspace ; /usr/bin/env /.venv/bin/python /root/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 40079 -- /workspace/src/gen_ea_rl/task_to_plan.py 
# Peel a potato
# - Step: Wash the potato thoroughly under running water to remove dirt.
#    - name='hold' object=Potato(name='potato'), Description: Hold the potato steady
#    - name='move' object=Potato(name='potato'), Description: Move the potato under running water
#    - name='rotate' object=Potato(name='potato'), Description: Rotate the potato to clean all sides
# - Step: Trim off both ends to create a flat surface for stability.
#    - name='hold' object=Potato(name='potato'), Description: Hold the potato steady
#    - name='cut' object=Knife(name='knife'), Description: Use knife to cut off one end
#    - name='rotate' object=Potato(name='potato'), Description: Rotate the potato to the other end
#    - name='cut' object=Knife(name='knife'), Description: Use knife to cut off the remaining end
# - Step: Use a vegetable peeler (or a small knife) to remove the skin in long, consistent strokes from top to bottom.
#    - name='hold' object=Potato(name='potato'), Description: Hold the potato steady
#    - name='peel' object=Peeler(name='peeler'), Description: Use the peeler (or knife) to start at the top and move down in long strokes
#    - name='move' object=Peeler(name='peeler'), Description: Move the peeler continuously downward, maintaining a smooth motion
#    - name='rotate' object=Potato(name='potato'), Description: Rotate the potato to expose the next segment and repeat
# - Step: Flip and repeat on all sides until the peel is completely removed.
#    - name='move' object=Potato(name='potato'), Description: Move Potato to the Peeler
#    - name='hold' object=Potato(name='potato'), Description: Hold the Potato firmly
#    - name='rotate' object=Potato(name='potato'), Description: Rotate the Potato 180° (flip)
#    - name='peel' object=Potato(name='potato'), Description: Peel one side of the potato
#    - name='peel' object=Potato(name='potato'), Description: Repeat rotate + peel on each remaining side until no peel remains
# - Step: Rinse the peeled potato in cold water, then pat dry before using.
#    - name='move' object=Potato(name='potato'), Description: Hold the peeled potato in your hand, keeping it upright above a sink
#    - name='move' object=Potato(name='potato'), Description: Rotate the potato under cold running water until no dirt remains
#    - name='move' object=Potato(name='potato'), Description: Hold the potato and pat it dry with a clean towel

