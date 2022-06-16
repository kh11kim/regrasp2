from cgitb import handler
from dataclasses import dataclass, field
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.transform import Rotation, Transform
from regrasp.utils.robot import Robot, Panda
from regrasp.utils.gripper import Gripper
from regrasp.utils.world import BulletWorld
import numpy as np
from regrasp.utils.body import Body
from typing import Dict, Union, Optional, List
from functools import partial
from pybullet_utils.bullet_client import BulletClient
from collections import namedtuple
from copy import copy, deepcopy
from collections import namedtuple


@dataclass
class Config:
    """Configuration of robots
    """
    q: np.ndarray
    T: Optional[Transform] = field(default_factory=lambda: None) #not registered
    index: int = field(default_factory=lambda: -1) #not registered

# Node of Kinematic Graph : Movable, Placeable

class Movable(Body):
    def __init__(
        self,
        physics_client: BulletClient,
        body_uid: int,
    ):
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid
        )
    
    def set_grasp_set(self, grasp_pose_set: List["Grasp"]):
        self.grasp_set = [Grasp(self, i, T) for i, T in enumerate(grasp_pose_set)]
    
    def set_placement_axis_set(self, placement_axis_set: List[np.ndarray]):
        self.placement_axis_set = placement_axis_set

    def get_grasp_set(self) -> List["Grasp"]:
        return self.grasp_set
    
    def sample_grasp(self) -> "Grasp":
        return np.random.choice(self.grasp_set)
    
    def sample_placement_axis(self) -> np.ndarray:
        i = np.random.choice(range(len(self.placement_axis_set)))
        return self.placement_axis_set[i]

    @classmethod
    def from_body(cls, body: Body):
        return cls(body.physics_client, body.uid)

class Placeable(Body):
    def __init__(
        self,
        physics_client: BulletClient,
        body_uid: int,
    ):
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid
        )
    
    def sample_from_plane(self, inner=0.05):
        lower, upper = self.get_AABB()
        xy = np.random.uniform(
            low=lower[:2]+inner, high=upper[:2]-inner
        )
        z = upper[-1]
        return np.array([*xy, z])

    @classmethod
    def from_body(cls, body: Body):
        return cls(body.physics_client, body.uid)

# Edge of Kinematic Graph
class Edge:
    def __init__(self, movable: Movable, tf: Transform):
        self.movable = movable
        self.tf = tf # transform to parent wrt movable

class Grasp(Edge):
    def __init__(
        self,
        movable: Movable,
        index: int,
        tf: Transform
    ):
        super().__init__(movable=movable, tf=tf)
        self.index = index
    
    def get_assigned_pose(self, pre=False):
        #assign to current pose of the movable
        assigned_grasp_pose = self.movable.get_base_pose() * self.tf
        if pre == False:    
            return assigned_grasp_pose
        return assigned_grasp_pose * Transform(translation=[0,0,-0.05])
    
class Placement(Edge):
    def __init__(
        self,
        movable: Movable,
        placeable: Placeable,
        tf: Transform,
    ):
        super().__init__(movable=movable, tf=tf)
        self.placeable = placeable

    def get_assigned_pose(self, grasp: Grasp, pre=False):    
        #assign to current pose of the movable and grasp
        assigned_pose = self.placeable.get_base_pose() * self.tf.inverse() * grasp.tf
        if pre == False:
            return assigned_pose
        return Transform(translation=[0,0,0.05]) * assigned_pose
    



class KinGraph:
    def __init__(
        self, 
        world: BulletWorld, 
        scene_maker:Optional[BulletSceneMaker]=None
    ):
        self.world = world
        self.robots = {}
        self.movable = {}
        self.placeable = {}
        self.parent = {}
        self.child = {}
        self.edge = {}
        self.sm = scene_maker

    def copy(self)->"KinGraph":
        kingraph_new = KinGraph(
            self.world,
            self.sm
        )
        kingraph_new.robots = copy(self.robots)
        kingraph_new.movable = copy(self.movable)
        kingraph_new.placeable = copy(self.placeable)
        kingraph_new.parent = deepcopy(self.parent)
        kingraph_new.child = deepcopy(self.child)
        kingraph_new.edge = copy(self.edge)
        return kingraph_new

    def set_movable(
        self, 
        name: str, 
        movable: Body, 
        parent_name: str,
        edge: Edge = None
    ):
        # Movable should be defined with 
        # Parent(robot/placeable) and Edge(grasp/placement)
        if (parent_name in self.placeable):
            placeable = self.placeable[parent_name]
            parent_pose = placeable.get_base_pose()
            const_tf = movable.get_base_pose().inverse() * parent_pose
            edge = Placement(movable, placeable, const_tf)
        elif parent_name in self.robots:
            pass #directly 
        elif (parent_name in self.movable):
            movable_parent = self.movable[parent_name]
            parent_pose = movable_parent.get_base_pose()
            const_tf = movable.get_base_pose().inverse() * parent_pose
            edge = Placement(movable, movable_parent, const_tf)
            
        self.movable[name] = movable
        self.parent[name] = parent_name
        self.child[name] = []
        self.child[parent_name].append(name)
        
        self.edge[name] = edge

    def assign_const(self):
        def assign_obj(obj_name):
            parent_name = self.parent[obj_name]
            if parent_name in self.robots.keys():
                robot = self.robots[parent_name]
                parent_pose = robot.get_ee_pose()
            elif parent_name in self.placeable.keys():
                placeable = self.placeable[parent_name]
                parent_pose = placeable.get_base_pose()
            else: #movable
                parent_pose = assign_obj(parent_name)
            pose = parent_pose * self.edge[obj_name].tf.inverse()
            self.movable[obj_name].set_base_pose(pose)
            return pose
        
        for obj_name in self.movable:
            assign_obj(obj_name)

    def set_robot(
        self, name, robot: Panda
    ):
        self.robots[name] = robot
        self.child[name] = []

    def set_placeable(
        self, name, obj: Body,
    ):
        self.placeable[name] = obj
        self.child[name] = []

    def mode_switch(
        self, 
        obj_name: str, 
        parent_name: str, 
        edge: str,
    ):
        parent_old_name = self.parent[obj_name]
        self.parent[obj_name] = parent_name
        self.child[parent_old_name] = \
            [child for child in self.child[parent_old_name] if child != obj_name]
        self.child[parent_name].append(obj_name)
        self.edge[obj_name] = edge

    def sample_T_target_from_grasp(
        self,
        movable_name: str,
        pre=True,
    ):
        movable: Movable = self.movable[movable_name]
        grasp = movable.sample_grasp()
        return grasp, grasp.get_assigned_pose(pre=pre)

    def sample_T_target_from_placement(
        self, 
        movable_name: str,
        placeable_name: str,
        pre=True,
    ):
        def get_pose_by_z_axis_and_point(z_axis, xyz):
            if np.allclose(z_axis, np.array([1, 0, 0])) | \
                np.allclose(z_axis, np.array([-1, 0, 0])):
                x_axis = np.array([0, 1, 0])
            else:
                x_axis = np.array([1, 0, 0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis/np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            rot = np.vstack([x_axis, y_axis, z_axis]).T
            rot = Rotation.from_matrix(rot)
            yaw = np.random.uniform(0, np.pi*2)
            yaw_rot = Rotation.from_rotvec(yaw * z_axis)
            pose = Transform(rot.inv()*yaw_rot, xyz)
            return pose

        movable: Movable = self.movable[movable_name]
        placeable: Placeable = self.placeable[placeable_name]
        
        xyz = placeable.sample_from_plane()
        z_axis = movable.sample_placement_axis()
        pose_ = get_pose_by_z_axis_and_point(z_axis, xyz)
        movable.set_base_pose(pose_)

        # solve penetration
        T_curr = movable.get_base_pose()
        _, extent = movable.get_AABB(output_center_extent=True)
        z = xyz[-1] + extent[-1]/2
        pose = Transform(pose_.rotation, np.array([*xyz[:2], z]))
        movable.set_base_pose(T_curr)
        movable.set_base_pose(pose) #debug
        edge_tf = (placeable.get_base_pose().inverse() * pose).inverse()
        placement = Placement(movable, placeable, edge_tf)
        # assign current grasp
        curr_edge = self.edge[movable_name]
        if type(curr_edge) is Placement:
            # only for test
            pass
        elif type(curr_edge) is Grasp:
            #grasp target
            pose = placement.get_assigned_pose(curr_edge, pre=pre)
        #self.sm.view_frame(pose, "pre_pose")
        return placement, pose

    def is_collision(self, config: Config):
        self.robots['panda'].set_arm_angles(config.q)
        self.assign_const()
        #robot
        if self.world.get_contacts(name="panda"):
            return True
        for movable_name in self.movable.keys():
            parent = self.parent[movable_name] # str
            child = self.child[movable_name] # List
            if self.world.get_contacts(
                name=movable_name, 
                exception=[parent, *child]):
                return True
        return False

def get_config_(robot:Panda, movable:Dict[str, Body]):
    return Config(
        q=robot.get_arm_angles(),
        T=robot.get_ee_pose()
    )

def set_config_(config:Config, robot:Panda, movable:Dict[str, Body]):
    robot.set_arm_angles(config.q)
    movable["box1"].set_base_pose(config.box1)
    movable["box2"].set_base_pose(config.box2)
    movable["box3"].set_base_pose(config.box3)

def get_random_joint_(robot:Panda):
    return np.random.uniform(
        low=robot.arm_lower_limit,
        high=robot.arm_upper_limit
    )
    
# def is_collision_(config: Config, kingraph: KinGraph, robot:Panda, world: BulletWorld):
#     robot.set_arm_angles(config.q)
#     #set_config_(config, robot, movable)
#     kingraph.assign_const()
#     for obj_name in ["panda", "box1", "box2", "box3"]:
#         if world.get_contacts(name=obj_name):
#             return True
#     return False

def embed_custom_fn(
    world: BulletWorld, robot:Panda, movable:Dict[str, Body]
):
    # is_collision_fn = partial(is_collision_, robot=robot, world=world)
    # get_config_fn = partial(get_config_, robot=robot, movable=movable)
    # set_config_fn = partial(set_config_, robot=robot, movable=movable)
    get_random_joint_fn = partial(get_random_joint_, robot=robot)
    return get_random_joint_fn
        # is_collision_fn, 
        # get_config_fn,
        # set_config_fn,