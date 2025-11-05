from pydantic import BaseModel, Field, BeforeValidator, AfterValidator, ValidationError
from typing import Union, Optional
from typing_extensions import Annotated
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from instructor import llm_validator
from pydantic.json_schema import SkipJsonSchema
from gen_ea_rl.helpers.helpers import save_to_file
import yourdfpy
from trimesh.transformations import translation_from_matrix, euler_from_matrix
import numpy as np


class ModificationType(Enum):
    ADD = "added"
    REMOVE = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def num_must_be_above_zero(num: float) -> float:
    if num <= 0:
        raise ValueError("This value has to be above zero.")
    return num

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def dof_must_be_non_negative(dof: int) -> int:
    if dof < 0:
        raise ValueError("Degrees of freedom can not be negative.")
    return dof

class Vec(BaseModel):
    def __init__(self, vec: np.ndarray):
        if len(vec) is not 3:
            raise ValidationError("Vec input should be size of 3.")
        super().__init__(x=vec[0], y=vec[1], z=vec[2])
    x: float
    y: float
    z: float

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def vec_nums_must_be_above_zero(vec: Vec) -> Vec:
    if vec.x <= 0 or vec.y <= 0 or vec.z <= 0:
        raise ValueError("The values in this vector must be above zero.")
    return vec

class Inertia(BaseModel):
    def __init__(self, inertial: yourdfpy.Inertial):
        xyz = translation_from_matrix(inertial.origin)
        rpy = euler_from_matrix(inertial.origin)
        super().__init__(mass=inertial.mass, 
                         origin_xyz=Vec(xyz),
                         origin_rpy=Vec(rpy), 
                         i_xx_yy_zz=Vec([inertial.inertia[0, 0], inertial.inertia[1, 1], inertial.inertia[2, 2]]))
    # 4x4 transform to roll pitch yaw:

    # chain_of_thought: str = Field(description="If the inertias is missing. Think of a mass and inertias that would fit here, given the robot use case and link/joint information.")
    # mass: Annotated[float, AfterValidator(num_must_be_above_zero)]
    mass: float
    origin_xyz: Vec
    origin_rpy: Vec
    # i_xx_yy_zz: Annotated[Vec, AfterValidator(vec_nums_must_be_above_zero)] = Field(description="If some values are zero, please provide non-zero value that would make contextual sense. Note inertias are typically small.")
    i_xx_yy_zz: Vec

class Box(BaseModel):
    size: Annotated[Vec, AfterValidator(vec_nums_must_be_above_zero)] = Field(description="If some values are zero, please provide non-zero value given the robot use case and link/joint information.")

class Sphere(BaseModel):
    radius: Annotated[float, AfterValidator(num_must_be_above_zero)] = Field(description="If some values are zero, please provide non-zero value given the robot use case and link/joint information.")

class Cylinder(BaseModel):
    radius: Annotated[float, AfterValidator(num_must_be_above_zero)] = Field(description="If some values are zero, please provide non-zero value given the robot use case and link/joint information.")
    length: Annotated[float, AfterValidator(num_must_be_above_zero)] = Field(description="If some values are zero, please provide non-zero value given the robot use case and link/joint information.")

class Collision(BaseModel):
    def __init__(self, coll: yourdfpy.Collision):
        xyz = translation_from_matrix(coll.origin)
        rpy = euler_from_matrix(coll.origin)
        if coll.geometry.box is not None:
            box = coll.geometry.box
            geom = Box(size=Vec([box.size[0], box.size[1], box.size[2]]))
        elif coll.geometry.cylinder is not None:
            cylinder = coll.geometry.cylinder
            geom = Cylinder(radius=cylinder.radius, length=cylinder.length)
        elif coll.geometry.sphere is not None:
            sphere = coll.geometry.sphere
            geom = Sphere(radius=sphere.radius)
        super().__init__(origin_xyz=Vec(xyz), origin_rpy=Vec(rpy), geometry_type=geom)

    # chain_of_thought: str = Field(description="If the collision is missing. Think of a collision parameters that would fit here, given the robot use case and link/joint information.")
    origin_xyz: Vec
    origin_rpy: Vec
    geomtery_type: Union[Box, Sphere, Cylinder] = Field(description="If the collision is missing or set to zero in one dimension, think of a collision type that would fit here, given the robot use case and link/joint information.")

class Revolute(BaseModel):
    effort: float
    velocity: float
    lower: float
    upper: float
    axis: Vec

class Continuous(BaseModel):
    lower: float
    upper: float
    axis: Vec

class Prismatic(BaseModel):
    effort: float
    velocity: float
    lower: float
    upper: float
    axis: Vec

class Fixed(BaseModel):
    pass

class Joint(BaseModel):
    def __init__(self, joint: yourdfpy.Joint):
        xyz = translation_from_matrix(joint.origin)
        rpy = euler_from_matrix(joint.origin)
        if joint.type == 'revolute':
            joint_type = Revolute(effort=joint.limit.effort, velocity=joint.limit.velocity, lower=joint.limit.lower, upper=joint.limit.upper, axis=Vec(joint.axis))
        elif joint.type == 'continuous':
            joint_type = Continuous(lower=joint.limit.lower, upper=joint.limit.upper, axis=Vec(joint.axis))
        elif joint.type == 'fixed':
            joint_type = Fixed()
        elif joint.type == 'prismatic':
            joint_type = Prismatic(effort=joint.limit.effort, velocity=joint.limit.velocity, lower=joint.limit.lower, upper=joint.limit.upper, axis=Vec(joint.axis))
        super().__init__(name=joint.name, parent_link = joint.parent, child_link=joint.child,
                         origin_xyz=Vec(xyz), origin_rpy=Vec(rpy), joint_type=joint_type)
    name: str
    parent_link: str
    child_link: str
    origin_xyz: Vec
    origin_rpy: Vec
    joint_type: Union[Revolute, Prismatic, Fixed]

class Link(BaseModel):
    def __init__(self, link: yourdfpy.Link):
        super().__init__(name=link.name, inertial=Inertia(link.inertial), collisions=[Collision(coll) for coll in link.collisions])
    name: str
    inertial: Inertia
    collisions: list[Collision]

class Robot(BaseModel):
    name:str
    # base_link: Link
    links: list[Link]
    joints: list[Joint]

class Tasks(BaseModel):
    # chain_of_thought: str = Field(description="Extract the information. Be concise. Be exhaustive as if you had to recreate the URDF from this data. Use contextual information of the whole URDF.")
    task_examples: list[str] = Field(description="Provide a list of examples of specific tasks this robot could do. Omit meta data about the robot.", min_length=20)

    @staticmethod
    def developer() -> str:
        return "You are a robot design creation AI. You extract specific tasks a robot could do given meta data and its URDF."

    def __str__(self):
        string = ""
        for i, task in enumerate(self.task_examples):
            string  += f"{i:02d}. {task}\n"
        return string
    

class RobotDesign(BaseModel):
    chain_of_thought: str = Field(description="Extract the information. Be concise. Be exhaustive as if you had to recreate the URDF from this data. Use contextual information of the whole URDF.")
    approxiamted_materials: str = Field(description="Provide material information given the task_examples.")
    actuator_information: str = Field(description="Provide actuator information given the task_examples.")
    actuated_num_dof: Annotated[int, AfterValidator(dof_must_be_non_negative)] = Field(description="Provide actuated degrees of freedom.")
    total_num_dof: Annotated[int, AfterValidator(dof_must_be_non_negative)] = Field(description="Provide total degrees of freedom.")
    gripper_type: Optional[str] = Field(default=None, description="If the robot has a gripper, provide the gripper type.")
    robot_type: str
    environments_used_in: list[str] = Field(description="Provide examples of environments this robot would be used in.", min_length=20)
    task_examples: list[str] = Field(description="Provide examples of tasks this robot could do.", min_length=20)

    def __str__(self):
        string = ""

        # string += "Chain of thought:\n" + self.chain_of_thought + "\n\n"

        def bullets(pre:str, l:list[str]) -> str:
            temp = pre + "\n"
            for i in range(len(l)):
                temp  += f"    {pre[0]}.{i+1} {l[i]}\n"
            return temp
        
        string+=bullets("1. Task examples:", self.task_examples)
        # string += f"1. {self.task_examples}\n"
        string+=bullets("2. Environment examples:", self.environments_used_in)
        # string += f"2. {self.environments_used_in}\n"
        string += f"3. Robot Type: {self.robot_type}\n"
        string += f"4. Gripper Type: {self.gripper_type}\n"
        string += f"5. Total DOF: {self.total_num_dof}. Total actuated DOF: {self.actuated_num_dof}\n"
        string += f"6. Actuator Information: {self.actuator_information}\n"
        string += f"7. Material Information: {self.approxiamted_materials}"
        return string

class FileType(Enum):
    # XACRO = "xacro"
    URDF = "urdf"

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def urdf_check(urdf: str) -> str:
    save_to_file("/data/output/temp.urdf", urdf)
    try:
        model = yourdfpy.URDF.load("/data/output/temp.urdf")
    except Exception as e:
        raise ValueError(f"Unable to load the URDF/XACRO. Do not abbreviate or skip joints or links.")
    
    if not model.validate():
        import os
        out = os.popen(model.validate()).read()
        raise ValueError(f"Unable to validate the URDF/XACRO. Error: {out}")
            
    return urdf

class URDF(BaseModel):
    developer: SkipJsonSchema[str] = "You are a robot design creation AI. You create Unified Robot Description Format (URDF)."
    # xml_code: Annotated[str, AfterValidator(urdf_check)] = Field(description="Provide full robot xml. Do not abbereviate. Never skip links or joints. You can use XACRO notation.")
    xml_code: Annotated[str, AfterValidator(urdf_check)] = Field(description="Provide full robot .urdf file. Do not abbereviate. Never skip links or joints.")
    type: FileType


class UrdfAnalysis(BaseModel):
    chain_of_thought: str = Field(description="Analysis, chain of thought of the change in URDF given a task.")

    @staticmethod
    def developer() -> str:
        return "Output URDF is the input URDF but with either an addtion of a link, removal of a link or no change to optimize for a given task. Pretend you are thinking out loud why you perform the changes that go from first URDF to second URDF. For example: \"To optimize for this task, this link should be removed/added/modified...\"."


class Modification(BaseModel):
    def __init__(self, random_type: ModificationType, link: yourdfpy.Link, joint: yourdfpy.Joint):
        super().__init__(randomization_type=random_type, link=Link(link), joint=Joint(joint))

    modification_type: ModificationType = None
    link: Link
    joint: Joint

def inverse_randomization_step(step: Modification) -> None:
    if step.modification_type is ModificationType.ADD:
        step.modification_type = ModificationType.REMOVE
    elif step.modification_type is ModificationType.REMOVE:
        step.modification_type = ModificationType.ADD

def to_yourdfpy_link(link: Link) -> yourdfpy.Link:
    """
    Convert a Link Pydantic model back to a yourdfpy.Link object
    """
    # Create the link with basic properties
    yourdfpy_link = yourdfpy.Link(name=link.name)
    
    # Convert inertial properties
    if link.inertial:
        inertial = link.inertial
        # Create origin matrix from xyz and rpy
        origin_matrix = np.eye(4)
        origin_matrix[:3, 3] = [inertial.origin_xyz.x, inertial.origin_xyz.y, inertial.origin_xyz.z]
        
        # Create rotation matrix from rpy
        from trimesh.transformations import euler_matrix
        rotation_matrix = euler_matrix(inertial.origin_rpy.x, inertial.origin_rpy.y, inertial.origin_rpy.z)
        origin_matrix[:3, :3] = rotation_matrix[:3, :3]
        
        # Create inertia matrix
        inertia_matrix = np.eye(3)
        inertia_matrix[0, 0] = inertial.i_xx_yy_zz.x
        inertia_matrix[1, 1] = inertial.i_xx_yy_zz.y
        inertia_matrix[2, 2] = inertial.i_xx_yy_zz.z
        
        yourdfpy_link.inertial = yourdfpy.Inertial(
            mass=inertial.mass,
            origin=origin_matrix,
            inertia=inertia_matrix
        )
    
    # Convert collisions
    yourdfpy_link.collisions = []
    for collision in link.collisions:
        # Create origin matrix for collision
        coll_origin = np.eye(4)
        coll_origin[:3, 3] = [collision.origin_xyz.x, collision.origin_xyz.y, collision.origin_xyz.z]
        
        # Create rotation matrix from rpy
        coll_rotation = euler_matrix(collision.origin_rpy.x, collision.origin_rpy.y, collision.origin_rpy.z)
        coll_origin[:3, :3] = coll_rotation[:3, :3]
        
        # Create geometry based on type
        geometry = yourdfpy.Geometry()
        if isinstance(collision.geometry_type, Box):
            geometry.box = yourdfpy.Box(size=[collision.geometry_type.size.x, 
                                            collision.geometry_type.size.y, 
                                            collision.geometry_type.size.z])
        elif isinstance(collision.geometry_type, Sphere):
            geometry.sphere = yourdfpy.Sphere(radius=collision.geometry_type.radius)
        elif isinstance(collision.geometry_type, Cylinder):
            geometry.cylinder = yourdfpy.Cylinder(radius=collision.geometry_type.radius, 
                                                length=collision.geometry_type.length)
        
        yourdfpy_collision = yourdfpy.Collision(origin=coll_origin, geometry=geometry)
        yourdfpy_link.collisions.append(yourdfpy_collision)
    
    return yourdfpy_link

def to_yourdfpy_joint(joint: Joint) -> yourdfpy.Joint:
    """
    Convert a Joint Pydantic model back to a yourdfpy.Joint object
    """
    # Create origin matrix
    origin_matrix = np.eye(4)
    origin_matrix[:3, 3] = [joint.origin_xyz.x, joint.origin_xyz.y, joint.origin_xyz.z]
    
    # Create rotation matrix from rpy
    from trimesh.transformations import euler_matrix
    rotation_matrix = euler_matrix(joint.origin_rpy.x, joint.origin_rpy.y, joint.origin_rpy.z)
    origin_matrix[:3, :3] = rotation_matrix[:3, :3]
    
    # Determine joint type and create appropriate joint
    if isinstance(joint.joint_type, Fixed):
        joint_type = 'fixed'
        yourdfpy_joint = yourdfpy.Joint(
            name=joint.name,
            type=joint_type,
            parent=joint.parent_link,
            child=joint.child_link,
            origin=origin_matrix
        )
    
    elif isinstance(joint.joint_type, Revolute):
        joint_type = 'revolute'
        limit = yourdfpy.Limit(
            effort=joint.joint_type.effort,
            velocity=joint.joint_type.velocity,
            lower=joint.joint_type.lower,
            upper=joint.joint_type.upper
        )
        yourdfpy_joint = yourdfpy.Joint(
            name=joint.name,
            type=joint_type,
            parent=joint.parent_link,
            child=joint.child_link,
            origin=origin_matrix,
            axis=[joint.joint_type.axis.x, joint.joint_type.axis.y, joint.joint_type.axis.z],
            limit=limit
        )
    
    elif isinstance(joint.joint_type, Prismatic):
        joint_type = 'prismatic'
        limit = yourdfpy.Limit(
            effort=joint.joint_type.effort,
            velocity=joint.joint_type.velocity,
            lower=joint.joint_type.lower,
            upper=joint.joint_type.upper
        )
        yourdfpy_joint = yourdfpy.Joint(
            name=joint.name,
            type=joint_type,
            parent=joint.parent_link,
            child=joint.child_link,
            origin=origin_matrix,
            axis=[joint.joint_type.axis.x, joint.joint_type.axis.y, joint.joint_type.axis.z],
            limit=limit
        )
    
    else:
        raise ValueError(f"Unknown joint type: {type(joint.joint_type)}")
    
    return yourdfpy_joint