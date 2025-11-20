from gen_ea_rl.helpers.robot_model import Modification, ModificationType, to_yourdfpy_link, to_yourdfpy_joint
from gen_ea_rl.helpers.urdf_helpers import add_link, remove_link, generate_link
from yourdfpy import URDF
import random


def modify_urdf(urdf: URDF, modification: Modification) -> None:
    """Modify a URDF model."""
    if modification.modification_type == ModificationType.ADD:
        link = to_yourdfpy_link(modification.link)
        joint = to_yourdfpy_joint(modification.joint)
        add_link(urdf, link, joint, modification.child_joint_names)
    elif modification.modification_type == ModificationType.REMOVE:
        remove_link(urdf, modification.link.name)


def randomize_urdf(urdf: URDF, add_chance: float) -> URDF:
    """Randomly modify a URDF model by adding or removing links/joints."""
    operation = random.random()
    if operation < (1.0 - add_chance) and len(urdf.robot.links) > 1:
        # Remove a random link (not the base link)
        link_to_remove = urdf.robot.links[random.randint(1, len(urdf.robot.links) - 1)]
        removed_link, removed_joint, child_joints = remove_link(urdf, link_to_remove.name)
        step = Modification.from_yourdfpy(
            modification_type=ModificationType.REMOVE,
            link=removed_link,
            joint=removed_joint,
            child_joint_names=[child_joint.name for child_joint in child_joints],
        )
    else:
        # Add a new link to a random existing link
        parent_link = urdf.robot.links[random.randint(0, len(urdf.robot.links) - 1)]
        # TODO add child modification
        new_link, new_joint = generate_link(urdf, parent_link.name)
        add_link(urdf, new_link, new_joint)
        step = Modification(modification_type=ModificationType.ADD, link=new_link, joint=new_joint)
    return urdf, step
