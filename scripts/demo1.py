import matplotlib.pyplot as plt
import numpy as np

import pybullet as p
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


def is_collision_(
    config: Config, 
    world: BulletWorld, 
    robots: DualArm, 
    obj: Body
) -> bool:
    robots.left.set_arm_angles(config.q_l)
    robots.right.set_arm_angles(config.q_r)
    if config.mode.hand == Hand.OBJ_IN_LEFT:
        T_ee = robots.left.forward_kinematics(config.q_l)
    else:
        T_ee = robots.right.forward_kinematics(config.q_r)
    T_obj = T_ee * config.mode.grasp.T.inverse()
    obj.set_base_pose(T_obj)

    world.step(only_collision_detection=True)
    for body in [robots.left, robots.right, obj]:
        if world.get_contacts(body=body):
            return True
    return False

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

def main():
    is_box_left = True

    #real world
    world, robots, scene_maker = set_world(gui=True, box_left=is_box_left)
    table_height = 0.2
    box_pose_init = Transform(Rotation.identity(), [0.3, 0.3, table_height+0.05])
    box = world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", box_pose_init, scale=1.1)
    world.wait_for_rest()
    #load grasp set
    grasp_set = np.load("grasp_set.npz", allow_pickle=True)["x"]
    grasp_set = [Grasp(T, i) for i, T in enumerate(grasp_set)]
    mode_set = []
    for hand in [Hand.OBJ_IN_LEFT, Hand.OBJ_IN_RIGHT]:
        for grasp in grasp_set:
            mode_set.append(Mode(hand, grasp))
    
    #test
    is_collision = partial(
        is_collision_,
        world=world,
        robots=robots,
        obj=box,
    )
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
    
    # prepare planner
    tp = TransitionPlanner(
        robots,
        scene_maker,
        is_collision,
        grasp_is_collision,
        p_global_explore=0.8,
        p_constraint_explore=0.1
    )

    # start configuration
    mode_start = Mode(
        hand=Hand.OBJ_IN_LEFT if is_box_left else Hand.OBJ_IN_RIGHT,
        grasp=grasp_set[90],
    )
    mode_goal = Mode(
        hand=Hand.OBJ_IN_LEFT,
        grasp=grasp_set[435] #feasible 435
    )
    robot_start = robots.left if mode_start.hand == Hand.OBJ_IN_LEFT \
                              else robots.right
    joints = robot_start.inverse_kinematics(pose=box.get_base_pose()*mode_start.grasp.T)
    robot_start.set_arm_angles(joints)
    config_start = Config(
        mode=mode_start,
        q_l=robots.left.get_arm_angles(),
        q_r=robots.right.get_arm_angles(),
        T_l=robots.left.get_ee_pose(),
        T_r=robots.right.get_ee_pose(),
    )

    #grasp_numbers = {i:0 for i in range(600)}
    new_config = config_start
    for i in range(100):
        config_trans, mode_trans = tp.plan(new_config, [mode_goal], mode_set)
        if mode_trans == mode_goal:
            break
        else:
            new_config = deepcopy(config_trans)
            new_config.mode = mode_trans
        #grasp_numbers[grasp_number] += 1
        #grasp_numbers.append(grasp_number)
    
    # for grasp_number in grasp_numbers:
    #     if grasp_numbers[grasp_number] != 0:
    #         print(f"grasp {grasp_number} : {grasp_numbers[grasp_number]}")

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

# def check_grasp_feasibility_(
#     grasp: Grasp, 
#     mode_set: List[Mode],
#     gripper1: Gripper,
#     gripper2: Gripper,
#     world: BulletWorld,
# ) -> bool:
    

#     feasible_mode_set = []
#     for mode_candidate in mode_set:
#         if not gripper_is_collision(grasp, mode_candidate.grasp, gripper1, gripper2, world):
#             feasible_mode_set.append(mode_candidate)
#     return feasible_mode_set
if __name__ == "__main__":
    main()