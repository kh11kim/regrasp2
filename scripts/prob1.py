from re import L
from urllib import robotparser
import numpy as np

from regrasp.utils.world import BulletWorld
from regrasp.utils.transform import Rotation, Transform
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.robot import Robot, Panda
from regrasp.utils.gripper import Gripper
from regrasp.utils.body import Body
from typing import Dict, Union
from dataclasses import dataclass, field
from regrasp.planning.planner import *
from functools import partial
from collections import namedtuple
Mode = namedtuple("Mode", ["tree", "kingraph", "plan"])

def get_graspset_of_box(gui):
    """Usage:
    grasp_set = get_graspset_of_box(gui=False)
    np.savez("data/urdfs/block/grasp_set.npz", grasp_set=grasp_set)
    """
    fake_world = BulletWorld(gui=gui)
    scene_maker = BulletSceneMaker(fake_world)
    box_edge = 0.03
    box_size = np.array([box_edge, box_edge, box_edge])
    box1 = fake_world.load_urdf("box1", "data/urdfs/block/urdf/block_r.urdf")
    hand = Gripper(fake_world)

    sample_per_plane = 3
    yaw_per_point = 10
    yaws = np.linspace(0, 2*np.pi, yaw_per_point, endpoint=False)

    grasp_set = []
    # xz
    sample_points_plane = (np.random.rand(sample_per_plane, 2) - 0.5) * 2 * 0.015
    for pt in sample_points_plane:
        for yaw in yaws:
            for is_rev in [0, 1]:
                tcp = Transform(Rotation.from_euler("zyx",[0,0,0]), [pt[0], 0, pt[1]])
                rot_rev = Transform(Rotation.from_euler("xyz", [0,0,np.pi*is_rev]))
                rot_yaw = Transform(Rotation.from_euler("zxy", [0,0,yaw]))
                hand.reset(tcp*rot_yaw*rot_rev)
                if not hand.detect_contact():
                    obj_to_grasp = tcp*rot_yaw*rot_rev
                    grasp_set.append(obj_to_grasp)
                hand.remove()
    # yz
    sample_points_plane = (np.random.rand(sample_per_plane, 2) - 0.5) * 2 * 0.015
    for pt in sample_points_plane:
        for yaw in yaws:
            for is_rev in [0, 1]:
                tcp = Transform(Rotation.from_euler("zyx",[np.pi/2,0,np.pi/2]), [0, pt[0], pt[1]])
                rot_rev = Transform(Rotation.from_euler("xyz", [0,0,np.pi*is_rev]))
                rot_yaw = Transform(Rotation.from_euler("zxy", [0,0,yaw]))
                hand.reset(tcp*rot_yaw*rot_rev)
                if not hand.detect_contact():
                    obj_to_grasp = tcp*rot_yaw*rot_rev
                    grasp_set.append(obj_to_grasp)
                hand.remove()
    # xy
    sample_points_plane = (np.random.rand(sample_per_plane, 2) - 0.5) * 2 * 0.015
    for pt in sample_points_plane:
        for yaw in yaws:
            for is_rev in [0, 1]:
                tcp = Transform(Rotation.from_euler("zyx",[0,0,np.pi/2]), [pt[0], pt[1], 0])
                rot_rev = Transform(Rotation.from_euler("xyz", [0,0,np.pi*is_rev]))
                rot_yaw = Transform(Rotation.from_euler("zxy", [0,0,yaw]))
                hand.reset(tcp*rot_yaw*rot_rev)
                if not hand.detect_contact():
                    obj_to_grasp = tcp*rot_yaw*rot_rev
                    grasp_set.append(obj_to_grasp)
                hand.remove()
    return grasp_set

def set_world(gui: bool):
    world = BulletWorld(gui=gui)
    distance_between_table = 0.4
    table_height = 0.2

    robot = world.load_robot(
        name="panda",
        pose = Transform(Rotation.identity(), [0,0,0])
    )
    hand = Gripper(world)
    scene_maker = BulletSceneMaker(world)
    scene_maker.create_plane(z_offset=-0.4)
    scene_maker.create_table("ground", 2, 2, 0.4)
    scene_maker.create_table("table1", 0.3, 0.3, 0.2, x_offset=0.4, y_offset=distance_between_table/2, z_offset=table_height)
    scene_maker.create_table("table2", 0.3, 0.3, 0.2, x_offset=0.4, y_offset=-distance_between_table/2, z_offset=table_height)
    placeable = {}
    for obj in ["table1", "table2"]:
        placeable[obj] = Placeable.from_body(world.bodies[obj])

    box_edge = 0.03
    box_size = np.array([box_edge, box_edge, box_edge])
    box1_pos = np.array([0.4, -0.2, 0.2+box_edge/2])
    box2_pos = np.array([0.4, -0.2, 0.2+box_edge/2+box_edge])
    box3_pos = np.array([0.4, -0.2, 0.2+box_edge/2+box_edge*2])
    box1 = world.load_urdf(
        "box1", "data/urdfs/block/urdf/block_r.urdf", Transform(translation=box1_pos))
    box2 = world.load_urdf(
        "box2", "data/urdfs/block/urdf/block_g.urdf", Transform(translation=box2_pos))
    box3 = world.load_urdf(
        "box3", "data/urdfs/block/urdf/block_b.urdf", Transform(translation=box3_pos))
    box_grasp_set = np.load("data/urdfs/block/grasp_set.npz", allow_pickle=True)["grasp_set"]
    box_placement_axis_set = [
        np.array([1,0,0]), np.array([-1,0,0]),
        np.array([0,1,0]), np.array([0,-1,0]),
        np.array([0,0,1]), np.array([0,0,-1]),
    ]
    movable: Dict[Grasp] = {}
    for obj in ["box1", "box2", "box3"]:
        movable[obj] = Movable.from_body(world.bodies[obj])
        movable[obj].set_grasp_set(box_grasp_set)
        movable[obj].set_placement_axis_set(box_placement_axis_set)
    
    world.set_gravity([0,0,-9.81])
    world.wait_for_rest()
    
    kingraph = KinGraph(world, scene_maker)
    kingraph.set_robot("panda", robot)
    #kingraph.set_hand(hand)
    kingraph.set_placeable("table1", placeable["table1"])
    kingraph.set_placeable("table2", placeable["table2"])
    kingraph.set_movable("box1", movable["box1"], "table2", "placement")
    kingraph.set_movable("box2", movable["box2"], "box1", "placement")
    kingraph.set_movable("box3", movable["box3"], "box2", "placement")
    #TEST: 
    # grasp = Grasp(box3, 85, box_grasp_set[85])
    # kingraph.set_movable("box3", movable["box3"], "robot", "grasp", const=grasp)
    return world, robot, kingraph

def main():
    #np.random.seed(4)
    world, robot, kingraph = set_world(gui=False)
    get_random_joint_fn = embed_custom_fn(world, robot, kingraph.movable)
    sm = BulletSceneMaker(world)
    hand = Gripper(world)
    mp = MotionPlanner(robot, world, hand) #is_collision_fn, 
    
    kingraph.assign_const()
    # algorithm
    # forward path
    box3: Movable = kingraph.movable["box3"]
    #grasp_affordance_set = get_grasp_affordance(box3, hand, world)
    grasp_affordance_set = box3.get_grasp_set()
    
    plan_skeleton = [
        (0, "move_free", "box3", None),
        (1, "move_grasp", "box3", "table1"),
        (2, "move_free", "box2", None),
        (3, "move_grasp", "box2", "table1"),
        (4, "move_free", "box1", None),
        (5, "move_grasp", "box1", "table1"),
    ]
    
    config_init = Config(
        q=robot.get_arm_angles(),
        T=robot.get_ee_pose()
    )
    tree_init = Tree(config_init)
    plan = plan_skeleton[0]
    modes = [Mode(tree_init, kingraph, plan)]

    while True:
        i = np.random.choice(range(len(modes)))
        mode = modes[i]

        plan_no = mode.plan[0]
        action_name = mode.plan[1]
        movable_name = mode.plan[2]
        placeable_name = mode.plan[3]
        
        edge = mp.plan_mode_switch(
            tree=mode.tree,
            kingraph=mode.kingraph,
            action_name=action_name,
            movable_name=movable_name,
            placeable_name=placeable_name,
            max_iter=10,
        )
        if edge is not None:
            kingraph_new = mode.kingraph.copy()
            parent_name = "panda" if action_name == "move_free" else placeable_name
            kingraph_new.mode_switch(movable_name, parent_name, edge)
            kingraph_new.assign_const()
            config_new = Config(
                q=robot.get_arm_angles(),
                T=robot.get_ee_pose()
            )
            tree_new = Tree(config_new)
            if plan_no + 1 >= len(plan_skeleton):
                print("MMMP end")
                break
            else:
                plan_new = plan_skeleton[plan_no+1]
                mode_new = Mode(tree_new, kingraph_new, plan_new)
                modes.append(mode_new)
    

    input()

if __name__ == "__main__":
    main()


# Legacy code

# def get_grasp_affordance(obj: Movable, hand: Gripper, world: BulletWorld):
#     """ get affordance without collision
#     """
#     grasp_set = obj.get_grasp_set()
#     pregrasp_pose = Transform(translation=[0,0,-0.05])

#     grasp_affordance_set = []
#     for grasp in grasp_set:
#         assigned_grasp = grasp.get_assigned_pose()
#         hand.reset(assigned_grasp)
#         if not world.get_contacts(body=hand.body, thres=0):
#             hand.reset(assigned_grasp * pregrasp_pose)
#             if not world.get_contacts(body=hand.body, thres=0):
#                 grasp_affordance_set.append(grasp)
#     hand.remove()
#     return grasp_affordance_set



# def mode_switch(grasp: Grasp, config: Config, robot: Panda, mp: MotionPlanner):
#     def step(config_old: Config, T_target: Transform, step_size=0.05):
#         config_new = copy.deepcopy(config_old)
#         config_new.q = mp.steer(
#             robot.get_arm_angles(), 
#             curr_pose=config_old.T, 
#             target_pose=T_target, 
#             q_delta_max=step_size
#         )
#         config_new.T = robot.forward_kinematics(config_new.q)
#         robot.set_arm_angles(config_new.q)
#         return config_new
    
#     EPS = 0.01
#     T_grasp = grasp.get_assigned_pose()
#     config_old = copy.deepcopy(config)
#     traj = []
#     while distance_ts(config_old.T, T_grasp) > EPS:
#         config_new = step(config_old, T_grasp)
#         traj.append(config_new)
#         config_old = config_new
#     traj.append(step(config_old, T_grasp, step_size=0.1))
#     return traj