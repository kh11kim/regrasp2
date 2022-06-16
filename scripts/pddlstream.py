import matplotlib.pyplot as plt
import numpy as np

import pybullet as p
from regrasp.utils import scene_maker
from regrasp.utils.world import BulletWorld
from regrasp.utils.transform import Rotation, Transform, orn_error
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.gripper import Gripper
from regrasp.utils.body import Body
from regrasp.utils.robot import Robot, Panda
from regrasp.planning.rrt import BiRRT
from tqdm import tqdm

from typing import Callable, Optional, List
from dataclasses import dataclass, field
from functools import partial
from copy import deepcopy
from regrasp.planning.data import Hand, Grasp, Config, DualArm, Mode
from regrasp.planning.tsrrt import ModeNode, handover_tsrrt_, Node
import time
from regrasp.planning.transition_planner import TransitionPlanner

def grasp_is_collision_(
    grasp1: Grasp, 
    grasp2: Grasp, 
    gripper1: Gripper,
    gripper2: Gripper,
    world: BulletWorld,
):
    gripper1.reset(grasp1.T, name="hand1")
    gripper2.reset(grasp2.T, name="hand2")
    #print(f"grasp1 {grasp1.index} grasp2 {grasp2.index}")
    #if (grasp1.index == 90) & (grasp2.index in [8, 192, 287, 295, 353, 365, 423, 425, 433, 435]):
        #print("here")
    world.step(only_collision_detection=True)
    for gripper in [gripper1, gripper2]:
        if world.get_contacts(body=gripper.body):
            return True
    return False

def sample_grasp_l(grasp_set):
    while True:
        yield np.random.choice(grasp_set)
    
    
def sample_grasp_r(
    grasp_l,
    grasp_set, 
    grasp_is_collision: Callable[[Grasp, Grasp], bool]
):
    while True:
        grasp_r = np.random.choice(grasp_set)
        if grasp_is_collision(grasp_l, grasp_r):
            grasp_r = None
        yield grasp_r

def sample_poses():
    while True:
        p = np.random.uniform(low=[0.2, -0.1, 0.2], high=[0.7, 0.1, 1])
        yield p

def sample_IK(grasp, obj_pose, obj:Body, robot:Panda):
    while True:
        obj.set_base_pose(obj_pose)
        T = obj_pose * grasp.T
        q = robot.inverse_kinematics(pose=T)
        yield q


def main():
    is_box_left = True
    world, robots, scene_maker = set_world(gui=True, box_left=is_box_left)
    box_pose_init = Transform(Rotation.identity(), [0.3, 0.3, 0.2+0.05])
    box = world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", box_pose_init, scale=1.1)
    world.wait_for_rest()

    #load grasp set
    grasp_set = np.load("grasp_set.npz", allow_pickle=True)["x"]
    grasp_set = [Grasp(T, i) for i, T in enumerate(grasp_set)]
    mode_set = []
    for hand in [Hand.OBJ_IN_LEFT, Hand.OBJ_IN_RIGHT]:
        for grasp in grasp_set:
            mode_set.append(Mode(hand, grasp))
    
    #check world
    check_world = BulletWorld(gui=False)
    #sm = BulletSceneMaker(check_world)
    check_hand1 = Gripper(check_world)
    check_hand2 = Gripper(check_world)
    check_box = check_world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", scale=1.1)
    grasp_is_collision = partial(
        grasp_is_collision_,
        gripper1=check_hand1,
        gripper2=check_hand2,
        world=check_world,
    )

    
    for i in range(100):
        obj_pose = sample_poses()
        grasp_l = sample_grasp_l(obj_pose)
        grasp_r = sample_grasp_r(grasp_l, grasp_set, grasp_is_collision)
        q_l = sample_IK(grasp_l, obj_pose, box, robots.left)
        q_r = sample_IK(grasp_r, obj_pose, box, robots.right)
        pose = next(obj_pose)
        g_l = next(grasp_l)
        g_r = next(grasp_r)
        
        
        
        scene_maker.view_frame(obj_pose)
    input()


def set_world(gui, box_left=True):
    world = BulletWorld(gui=gui)
    distance_between_robot = 0.6
    table_height = 0.2

    panda1 = world.load_robot(
        name="panda1",
        pose = Transform(Rotation.identity(), [0,distance_between_robot/2,0])
    )
    panda2 = world.load_robot(
        name="panda2",
        pose = Transform(Rotation.identity(), [0,-distance_between_robot/2,0])
    )
    scene_maker = BulletSceneMaker(world)
    scene_maker.create_plane(z_offset=-0.4)
    scene_maker.create_table(2, 2, 0.4)
    scene_maker.create_table(0.5, 0.5, 0.2, x_offset=0.4, y_offset=distance_between_robot/2, z_offset=table_height)
    scene_maker.create_table(0.5, 0.5, 0.2, x_offset=0.4, y_offset=-distance_between_robot/2, z_offset=table_height)
    
    world.set_gravity([0,0,-9.81])
    robots = DualArm(
        left=panda1,
        right=panda2,
    )
    world.wait_for_rest()

    return world, robots, scene_maker

if __name__ == "__main__":
    main()