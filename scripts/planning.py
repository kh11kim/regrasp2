from matplotlib.pyplot import get
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
from regrasp.planning.data import Mode, Config, Grasp, DualArm
from regrasp.planning.tsrrt import ModeNode, handover_tsrrt_, Node
import time

# @dataclass
# class HandoverNode:
#     grasp_l: Grasp
#     grasp_r: Grasp
#     path_to_parent: List[Node]
#     index: int = field(default_factory=lambda: -1) #not registered

class ModeTree:
    def __init__(self, root: ModeNode):
        self.V = []
        self.parent = {}
        self.root = root
        if root.mode == Mode.OBJ_IN_LEFT:
            self.l = {root.grasp.index:root.grasp}
            self.r = {}
        else:
            self.r = {root.grasp.index:root.grasp}
            self.l = {}
    
    def add_node(self, node: ModeNode):
        node.index = len(self.V)
        self.V.append(node)
        if node.mode == Mode.OBJ_IN_LEFT:
            self.l[node.grasp.index] = node.grasp
            self.r[node.grasp_handover.index] = node.grasp_handover
        else:
            self.r[node.grasp.index] = node.grasp
            self.l[node.grasp_handover.index] = node.grasp_handover
    
    def get_goal_grasp_set(self, curr_mode: Mode):
        if curr_mode == Mode.OBJ_IN_LEFT:
            return list(self.r.values())
        else:
            return list(self.l.values())
    
    def next(self):
        if len(self.V) == 0:
            return self.root
        last_node = self.V[-1]
        #next mode
        next_mode = Mode.OBJ_IN_LEFT if last_node.mode == Mode.OBJ_IN_RIGHT else Mode.OBJ_IN_RIGHT
        return ModeNode(
            mode=next_mode,
            grasp=last_node.grasp_handover,
            config_start=last_node.config_handover,
        )
    
    def get_mode_node_by_grasp(self, mode: Mode, grasp: Grasp):
        for V in self.V:
            if (mode == V.mode) & (V.grasp_handover.index == grasp.index):
                return V
        raise Exception("?")
        

def dual_arm_is_collision_(
    q_l: np.ndarray,
    q_r: np.ndarray,
    T_obj: Optional[Transform] = None,
    grasp_obj: Optional[Grasp] = None,
    mode: Optional[Mode] = None,
    world: Optional[BulletWorld] = None, 
    robots: Optional[DualArm] = None,
    obj: Optional[Body] = None, 
):
    # set configurations
    robots.left.set_arm_angles(q_l)
    robots.right.set_arm_angles(q_r)
    # if T_obj is None, get T_obj from grasp, mode
    if T_obj is None:
        if mode == Mode.OBJ_IN_LEFT:
            T_ee = robots.left.forward_kinematics(q_l)
        else:
            T_ee = robots.right.forward_kinematics(q_r)
        T_obj = T_ee * grasp_obj.T.inverse()
    obj.set_base_pose(T_obj)
    
    # check
    world.step(only_collision_detection=True)
    for body in [robots.left, robots.right, obj]:
        if world.get_contacts(body=body):    
            return True
    return False

def grasp_checker_is_collision_(
    grasp1: Grasp,
    grasp2: Grasp,
    hands: List[Gripper],
    world: BulletWorld,
):
    assert len(hands) == 2 #dual arm
    # set configurations
    hands[0].reset(grasp1.T, name="hand1")
    hands[1].reset(grasp2.T, name="hand2")
    # check
    world.step(only_collision_detection=True)
    for hand in hands:
        if world.get_contacts(body=hand.body):
            return True
    return False

def get_feasible_grasp_set_(
    grasp: Grasp,
    grasp_set: List[Grasp],
    is_collision: Callable[[Grasp, Grasp], bool]
):
    feasible_grasp_set = []
    for grasp_candidate in grasp_set:
        if not is_collision(grasp, grasp_candidate):
            feasible_grasp_set.append(grasp_candidate)
    return feasible_grasp_set

def main():
    # grasp_set = get_grasp_set(gui=True)
    # np.savez("grasp_set.npz", x=grasp_set)
    start_mode = Mode.OBJ_IN_LEFT
    box_left = start_mode == Mode.OBJ_IN_LEFT

    # Set World
    world, objects, scene_maker = set_world(gui=False, box_left=box_left)
    robots = DualArm(objects["panda1"], objects["panda2"])
    box = objects["box"]
    hand = Gripper(world)
    dual_arm_is_collision = partial(
        dual_arm_is_collision_,
        world=world,
        robots=robots,
        obj=box
    )
    
    # object-level collision checker
    check_world = BulletWorld(gui=True)
    sm = BulletSceneMaker(check_world)
    check_hand1 = Gripper(check_world)
    check_hand2 = Gripper(check_world)
    check_box = check_world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", scale=1.1)
    grasp_set = np.load("grasp_set.npz", allow_pickle=True)["x"]
    grasp_set = [Grasp(T, i) for i, T in enumerate(grasp_set)]
    get_feasible_grasp_set = partial(
        get_feasible_grasp_set_,
        grasp_set=grasp_set,
        is_collision=partial(
            grasp_checker_is_collision_,
            hands=[check_hand1, check_hand2],
            world=check_world
        )
    )
    
    # set rrt
    handover_tsrrt = partial(
        handover_tsrrt_,
        obj=box,
        robots=robots,
        sm=scene_maker,
        get_feasible_grasp_set=get_feasible_grasp_set,
        is_collision_=dual_arm_is_collision
    )
    

    # make start/goal config
    grasp =  grasp_set[90] #get_random_grasp(box, grasp_set, hand1) get_random_grasp(box, grasp_set, hand) #
    grasp_target = grasp_set[120]
    goal_mode = Mode.OBJ_IN_LEFT

    robot_start = robots.left if start_mode == Mode.OBJ_IN_LEFT \
                              else robots.right
    joints = robot_start.inverse_kinematics(pose=box.get_base_pose()*grasp.T)
    robot_start.set_arm_angles(joints)
    
    mode_node_start = ModeNode(
        mode=start_mode,
        grasp=grasp,
        config_start=Config(
            q_l=robots.left.get_arm_angles(),
            q_r=robots.right.get_arm_angles(),
            T_obj=box.get_base_pose()
        )
    )
    mode_node_goal = ModeNode(
        mode=goal_mode,
        grasp=grasp_target,
        config_start=Config(
            q_l=robots.left.arm_central,
            q_r=robots.left.arm_central,
            T_obj=robots.left.forward_kinematics(robots.left.arm_central)*grasp_target.T.inverse()
        )
    )

    mode_tree_fwd = ModeTree(root=mode_node_start)
    mode_tree_bwd = ModeTree(root=mode_node_goal)
    

    # Plan
    is_fwd = True
    for _ in range(100):
        # if mode_node.mode == Mode.OBJ_IN_LEFT:
        #     goal_grasp_set = [grasp_set[120]]
        # else:
        #     goal_grasp_set = []
        if is_fwd:
            tree1, tree2 = mode_tree_fwd, mode_tree_bwd
            print("fwd")
        else:
            tree1, tree2 = mode_tree_bwd, mode_tree_fwd
            print("bwd")

        next_mode_node = tree1.next()
        goal_grasp_set = tree2.get_goal_grasp_set(next_mode_node.mode)
        goal_grasp_set_index = [grasp.index for grasp in goal_grasp_set]
        print(f"goal_grasp_set:{goal_grasp_set_index}")
        mode_node = handover_tsrrt(next_mode_node, goal_grasp_set)
        tree1.add_node(mode_node)
        if mode_node.grasp_handover.index in goal_grasp_set_index:
            if mode_node.mode == Mode.OBJ_IN_LEFT:
                mode_next = Mode.OBJ_IN_RIGHT
            else:
                mode_next = Mode.OBJ_IN_LEFT
            mode_node_ = tree2.get_mode_node_by_grasp(mode_node.mode, mode_node.grasp_handover)
            is_collision = partial(
                dual_arm_is_collision,
                grasp_obj=mode_node.grasp_handover,
                mode=mode_next
            )
            get_random_config = lambda : np.random.uniform(
                low=[*robots.left.arm_lower_limit]*2, high=[*robots.left.arm_upper_limit]*2,
            )
            start = np.hstack([mode_node.config_handover.q_l, mode_node.config_handover.q_r])
            goal = np.hstack([mode_node_.config_start.q_l, mode_node.config_start.q_r])
            rrt = BiRRT(
                start=start,
                goal=goal,
                get_random_config=get_random_config,
                is_collision=is_collision
            )
            path = rrt.plan()
            if path is not None:

                print("goal!")
                break
        else:
            if mode_node.mode == Mode.OBJ_IN_LEFT:
                log = f"handover: left-to-right"
            else:
                log = f"handover: right-to-left"
            print(log)
            print(f"handover_grasp_index: {mode_node.grasp_handover.index}")
            is_fwd = not is_fwd

            
            
    
    for mode in mode_tree.V:
        for node in mode.trajectory:
            robots.left.set_arm_angles(node.q_l)
            robots.right.set_arm_angles(node.q_r)
            box.set_base_pose(node.T_obj)
            time.sleep(0.1)


def get_random_grasp(obj, grasp_set, hand):
    for i in range(100):
        i = np.random.choice(range(len(grasp_set)))
        rand_grasp = grasp_set[i]
        hand_tcp_pose = obj.get_base_pose() * rand_grasp
        hand.reset(hand_tcp_pose)
        if not hand.detect_contact():
            hand.remove()
            break
    return rand_grasp


def get_grasp_set(gui=False):
    """Make grasp set
    
    Returns:
        list[Transform]: grasp_set list

    example:
        grasp_set = get_grasp_set(hand, box, sm)
        np.savez("grasp_set.npz", x=grasp_set)
    """
    x_max = 0.01
    y_max = 0.025
    z_max = 0.01
    xx = np.linspace(-x_max, x_max, 3)
    yy = np.linspace(-y_max, y_max, 5)
    zz = np.linspace(-z_max, z_max, 3)
    tt = np.linspace(-np.pi, np.pi, 10)
    world = BulletWorld(gui=gui)
    sm = BulletSceneMaker(world)
    hand = Gripper(world)
    box = world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", scale=1.1)
    #sm.view_frame("obj", box.get_base_pose(), length=0.04)

    total_number = (len(xx)*len(yy)*len(tt)) *4
    pbar = tqdm(total=total_number)
    
    #yz
    grasp_pose = []
    for y in yy:
        for z in zz:
            for theta in tt:
                for i in [0,1]:
                    rot = Rotation.from_euler("zyx", [np.pi/2, 0, 0])
                    rev = Rotation.from_rotvec([0,0,np.pi])
                    rot_iter = Rotation.from_rotvec([theta,0,0])
                    if i == 1:
                        rot = rot * rev
                    rot = rot_iter * rot
                    point = np.array([0, y, z])                
                    tcp = Transform(rot, point)
                    hand.reset(tcp)
                    if not hand.detect_contact():
                        grasp_pose.append(tcp)                    
                    hand.remove()
                    pbar.update()
    #xy
    for x in xx:
        for y in yy:
            for theta in tt:
                for i in [0,1]:
                    if i == 1:
                        rot = Rotation.from_euler("zyx", [0,theta,-np.pi/2])
                    else:
                        rot = Rotation.from_euler("zyx", [np.pi,theta,-np.pi/2])
                    point = np.array([x, y, 0])                
                    tcp = Transform(rot, point)
                    hand.reset(tcp)
                    #sm.view_frame("tcp", tcp, length=0.04)
                    if not hand.detect_contact():
                        pose = hand.get_tcp_pose()
                        # sm.view_frame("grasp", pose, length=0.04)
                        # sm.view_frame("tcp", pose, length=0.04)
                        grasp_pose.append(tcp)                    
                    hand.remove()
                    pbar.update()
    pbar.close()
    return grasp_pose

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
    if box_left == True:
        box_pose_init = Transform(Rotation.identity(), [0.3, 0.3, table_height+0.05])
    else:
        box_pose_init = Transform(Rotation.identity(), [0.3, -0.3, table_height+0.05])
    box = world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", box_pose_init, scale=1.1)
    
    world.set_gravity([0,0,-9.81])
    objects = dict(
        panda1=panda1,
        panda2=panda2,
        box=box
    )
    world.wait_for_rest()

    return world, objects, scene_maker

if __name__ == "__main__":
    main()