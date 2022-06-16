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

def main():
    set_world(gui=True)
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