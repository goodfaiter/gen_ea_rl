from yourdfpy import URDF, Joint, Link, Limit, Inertial, Cylinder, Box, Sphere, Collision, Geometry
import numpy as np
import random
import math
from gen_ea_rl.helpers.robot_model import Modification, ModificationType, to_yourdfpy_link, to_yourdfpy_joint


def sample_cylinder_surface(l: float, r: float) -> np.ndarray:
    z = np.random.uniform(0, l)
    theta = np.random.uniform(0, 2 * math.pi)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return np.array([x, y, z])


def sample_box_surface(size: np.ndarray) -> np.ndarray:
    face = random.randint(0, 5)
    if face == 0:  # +x face
        x = size[0] / 2
        y = random.uniform(-size[1] / 2, size[1] / 2)
        z = random.uniform(-size[2] / 2, size[2] / 2)
    elif face == 1:  # -x face
        x = -size[0] / 2
        y = random.uniform(-size[1] / 2, size[1] / 2)
        z = random.uniform(-size[2] / 2, size[2] / 2)
    elif face == 2:  # +y face
        x = random.uniform(-size[0] / 2, size[0] / 2)
        y = size[1] / 2
        z = random.uniform(-size[2] / 2, size[2] / 2)
    elif face == 3:  # -y face
        x = random.uniform(-size[0] / 2, size[0] / 2)
        y = -size[1] / 2
        z = random.uniform(-size[2] / 2, size[2] / 2)
    elif face == 4:  # +z face
        x = random.uniform(-size[0] / 2, size[0] / 2)
        y = random.uniform(-size[1] / 2, size[1] / 2)
        z = size[2] / 2
    else:  # -z face
        x = random.uniform(-size[0] / 2, size[0] / 2)
        y = random.uniform(-size[1] / 2, size[1] / 2)
        z = -size[2] / 2
    return np.array([x, y, z])


def sample_sphere_surface(radius: float) -> np.ndarray:
    phi = random.uniform(0, math.pi)
    theta = random.uniform(0, 2 * math.pi)
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    return np.array([x, y, z])


def random_axis() -> np.ndarray:
    """Generate a random axis aligned with one of the primary axes."""
    axis_rand = random.random()
    sign_rand = random.random()
    sign = 1 if sign_rand > 0.5 else -1
    if axis_rand < 0.33:
        return np.array([sign * 1, 0, 0])
    elif axis_rand < 0.66:
        return np.array([0, sign * 1, 0])
    else:
        return np.array([0, 0, sign * 1])


def random_rotation_matrix() -> np.ndarray:
    """Generate a random rotation matrix."""
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, 2 * math.pi)
    z = random.uniform(0, 2 * math.pi)

    R_z = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])

    R_y = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [-math.sin(phi), 0, math.cos(phi)]])

    R_x = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])

    R = np.matmul(R_z, np.matmul(R_y, R_x))
    return R


def approximate_max_size(urdf: URDF) -> float:
    max_size = 0.0
    for joint in urdf.robot.joints:
        max_size = max(max_size, joint.origin[0:3, 3].max())
    for link in urdf.robot.links:
        for collision in link.collisions:
            if collision.geometry.cylinder is not None:
                max_size = max(max_size, collision.geometry.cylinder.length)
                max_size = max(max_size, collision.geometry.cylinder.radius)
            elif collision.geometry.box is not None:
                max_size = max(max_size, collision.geometry.box.size.max())
            elif collision.geometry.sphere is not None:
                max_size = max(max_size, collision.geometry.sphere.radius)
    return max_size


def find_unused_joint_name(urdf: URDF) -> str:
    i = 0
    for joint in urdf.robot.joints:
        if int(joint.name[:2]) == i:
            i += 1
            if i == len(urdf.robot.joints):
                joint_name = f"{i:02d}_joint"
            continue
        else:
            joint_name = f"{i:02d}_joint"
            break
    return joint_name


def find_unused_link_name(urdf: URDF) -> str:
    i = 0
    for link in urdf.robot.links:
        if int(link.name[:2]) == i:
            i += 1
            if i == len(urdf.robot.links):
                link_name = f"{i:02d}_link"
            continue
        else:
            link_name = f"{i:02d}_link"
            break
    return link_name


def generate_link(urdf: URDF, parent_link_name: str) -> tuple[Link, Joint]:
    # Find a joint name that is not used yet
    joint_name = find_unused_joint_name(urdf)
    parent_link = urdf.link_map[parent_link_name]
    child_link_name = find_unused_link_name(urdf)

    origin = np.eye(4)
    surface_point = np.zeros(3)
    if len(parent_link.collisions) > 0:
        parent_collision = parent_link.collisions[random.randint(0, len(parent_link.collisions) - 1)]
        if parent_collision.geometry.cylinder is not None:
            surface_point = sample_cylinder_surface(
                l=parent_collision.geometry.cylinder.length, r=parent_collision.geometry.cylinder.radius
            )
        elif parent_collision.geometry.box is not None:
            box_size = parent_collision.geometry.box.size
            surface_point = sample_box_surface(size=box_size)
        elif parent_collision.geometry.sphere is not None:
            radius = parent_collision.geometry.sphere.radius
            surface_point = sample_sphere_surface(radius=radius)
        origin[0:3, 3] = surface_point
        origin = np.matmul(origin, parent_collision.origin)

    # calculate "convex" orientation
    normal = surface_point / np.linalg.norm(surface_point)
    orthogonal_vector = np.array(normal)
    orthogonal_vector[2] = (-normal[1] - normal[0]) / np.abs(1.0e-3 + normal[2])
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    orthogonal_vector_cross = np.cross(normal, orthogonal_vector)
    orthogonal_vector_cross = orthogonal_vector_cross / np.linalg.norm(orthogonal_vector_cross)
    origin[0:3, 0] = orthogonal_vector
    origin[0:3, 1] = orthogonal_vector_cross
    origin[0:3, 2] = normal

    axis = random_axis()
    joint_random = random.random()
    if joint_random < 0.33:
        joint = Joint(
            name=joint_name,
            type="revolute",
            parent=parent_link_name,
            child=child_link_name,
            axis=axis,
            limit=Limit(lower=-1.57, upper=1.57, effort=80 * random.random(), velocity=8.0 * random.random()),
            origin=origin,
        )
    elif joint_random < 0.66:
        joint = Joint(
            name=joint_name,
            type="prismatic",
            parent=parent_link_name,
            child=child_link_name,
            axis=axis,
            limit=Limit(lower=0.0, upper=0.5 * random.random(), effort=100 * random.random(), velocity=0.5 * random.random()),
            origin=origin,
        )
    else:
        joint = Joint(
            name=joint_name,
            type="fixed",
            parent=parent_link_name,
            child=child_link_name,
            origin=origin,
        )

    max_size = approximate_max_size(urdf)
    min_size = 0.02 * max_size

    origin = np.eye(4)
    collision_random = random.random()
    if collision_random < 0.33:
        geometry = Geometry(
            cylinder=Cylinder(radius=min_size + 0.05 * max_size * random.random(), length=min_size + 0.5 * max_size * random.random())
        )
        origin[0:3, 3] = np.array([0, 0, geometry.cylinder.length / 2])
    elif collision_random < 0.66:
        geometry = Geometry(
            box=Box(
                size=np.array(
                    [
                        min_size + 0.2 * max_size * random.random(),
                        min_size + 0.2 * max_size * random.random(),
                        min_size + 0.2 * max_size * random.random(),
                    ]
                )
            )
        )
        origin[0:3, 3] = np.array([0, 0, geometry.box.size[2] / 2])
    else:
        geometry = Geometry(
            sphere=Sphere(
                radius=min_size + 0.1 * max_size * random.random(),
            )
        )
        origin[0:3, 3] = np.array([0, 0, geometry.sphere.radius])
    collision = Collision(name="", geometry=geometry, origin=origin)

    # TODO mass based on collision size and density, inertia calculation
    inertial = Inertial(mass=1.0 * random.random(), origin=origin, inertia=np.zeros([3, 3]))
    link = Link(name=child_link_name, inertial=inertial, visuals=[], collisions=[collision])
    return link, joint


def add_link(urdf: URDF, link: Link, joint: Joint, child_joint_names: list[str]) -> None:
    """Add a link to a URDF model."""
    urdf.robot.links.append(link)
    urdf.link_map[link.name] = link
    urdf.robot.joints.append(joint)
    urdf.joint_map[joint.name] = joint

    # Add if link is in between
    try:
        for child_joint_name in child_joint_names:
            urdf.joint_map[child_joint_name].parent = link.name
    except Exception as e:
        print(f"{e}")
        pass

def modify_urdf(urdf: URDF, modification: Modification) -> None:
    """Modify a URDF model."""
    if modification.modification_type == ModificationType.ADD:
        link = to_yourdfpy_link(modification.link)
        joint = to_yourdfpy_joint(modification.joint)
        add_link(urdf, link, joint)
    elif modification.modification_type == ModificationType.REMOVE:
        remove_link(urdf, modification.link.name)

def remove_link(urdf: URDF, link_name: str) -> tuple[Link, Joint]:
    """Remove a link from a URDF model and return removed link and joint."""
    if link_name not in urdf.link_map:
        print(f"Link {link_name} not found in URDF.")
        return urdf

    parent_joint = [joint for joint in urdf.joint_map.values() if joint.child == link_name][0]
    parent_link = parent_joint.parent
    child_joints = [joint for joint in urdf.joint_map.values() if joint.parent == link_name]

    # Modify all children joints
    for child_joint in child_joints:
        child_joint.parent = parent_link
        # child_joint.origin = np.matmul(np.transpose(child_joint.origin), parent_joint.origin)
        # TODO do the same for joint axis but first turn into rot mat

    # Remove the link and its associated joint
    removed_joint = None
    removed_link = None
    for joint in urdf.robot.joints:
        if joint.name == parent_joint.name:
            urdf.joint_map.pop(joint.name)
            urdf.robot.joints.remove(joint)
            removed_joint = joint
            break
    for link in urdf.robot.links:
        if link.name == link_name:
            urdf.link_map.pop(link_name)
            urdf.robot.links.remove(link)
            removed_link = link
            break
    return removed_link, removed_joint, child_joints


def remove_joint(urdf: URDF, joint_name: str) -> URDF:
    """Remove a joint from a URDF model."""
    if joint_name not in urdf.joint_map:
        print(f"Joint {joint_name} not found in URDF.")
        return urdf

    joint_to_remove = urdf.joint_map[joint_name]
    parent_link = urdf.link_map[joint_to_remove.parent]
    child_link = urdf.link_map[joint_to_remove.child]

    # Reassign child link's parent to the removed joint's parent link
    for joint in urdf.robot.joints:
        if joint.parent == child_link.name:
            joint.parent = parent_link.name
            joint.origin = np.matmul(joint.origin, joint_to_remove.origin)
            # TODO adjust axis
            break

    for link in urdf.robot.links:
        if link.name == child_link.name:
            for collision in link.collisions:
                collision.origin = np.matmul(collision.origin, joint_to_remove.origin)
                parent_link.collisions.append(collision)
            # TODO inertias origin adjustment according to mass
            link.inertial.origin = np.matmul(link.inertial.origin, joint_to_remove.origin)
            link.inertial.mass += parent_link.inertial.mass
            break

    # Remove the joint
    for joint in urdf.robot.joints:
        if joint.name == joint_name:
            urdf.robot.joints.remove(joint)
            break
    for link in urdf.robot.links:
        if link.name == child_link.name:
            urdf.robot.links.remove(link)
            break
    return urdf


def randomize_urdf(urdf: URDF, add_chance: float) -> URDF:
    """Randomly modify a URDF model by adding or removing links/joints."""
    operation = random.random()
    if operation < (1.0 - add_chance) and len(urdf.robot.links) > 1:
        # Remove a random link (not the base link)
        link_to_remove = urdf.robot.links[random.randint(1, len(urdf.robot.links) - 1)]
        removed_link, removed_joint, child_joints = remove_link(urdf, link_to_remove.name)
        step = Modification.from_yourdfpy(modification_type=ModificationType.REMOVE, link=removed_link, joint=removed_joint, child_joint_names=[child_joint.name for child_joint in child_joints])
    else:
        # Add a new link to a random existing link
        parent_link = urdf.robot.links[random.randint(0, len(urdf.robot.links) - 1)]
        #TODO add child modification
        new_link, new_joint = generate_link(urdf, parent_link.name)
        add_link(urdf, new_link, new_joint)
        step = Modification(modification_type=ModificationType.ADD, link=new_link, joint=new_joint)
    return urdf, step


def reorder_urdf(urdf: URDF) -> URDF:
    """Reorder links and joints in a URDF model to maintain consistent naming."""
    link_to_old_name = dict()
    for i, link in enumerate(urdf.robot.links):
        urdf.link_map.pop(link.name)
        new_name = f"{i:02d}_link"
        link_to_old_name[link.name] = new_name
        link.name = new_name
        urdf.link_map[new_name] = link

    for i, joint in enumerate(urdf.robot.joints):
        urdf.joint_map.pop(joint.name)
        joint.name = f"{i:02d}_joint"
        joint.parent = link_to_old_name[joint.parent]
        joint.child = link_to_old_name[joint.child]
        urdf.joint_map[joint.name] = joint

    return link_to_old_name
