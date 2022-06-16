import numpy as np
from regrasp.utils.world import BulletWorld
from regrasp.utils.transform import Rotation, Transform, orn_error
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.robot import Robot, Panda
from regrasp.utils.body import Body
from typing import List, Callable
from regrasp.planning.data import Grasp, Config1
from regrasp.utils.object import Attachment
from functools import partial
from regrasp.utils.gripper import Gripper
from regrasp.planning.rrt import BiRRT

def set_world(gui):
    world = BulletWorld(gui=gui)
    table_height = 0.2

    panda = world.load_robot(
        name="panda",
        pose = Transform(Rotation.identity(), [0,0,0])
    )
    scene_maker = BulletSceneMaker(world)
    scene_maker.create_plane(z_offset=-0.4)
    scene_maker.create_table(2, 2, 0.4)
    scene_maker.create_table(0.5, 0.5, 0.2, x_offset=0.4, y_offset=0, z_offset=table_height)
    
    world.set_gravity([0,0,-9.81])

    return world, panda, scene_maker

def set_check_world(gui=False):
    world = BulletWorld(gui=gui)
    hand = Gripper(world)
    scene_maker = BulletSceneMaker(world)
    scene_maker.create_plane()
    box = world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", Transform(), scale=1.1)
    return world, box, hand


def get_stable_placements():
    rot_list = [
        Rotation.identity(),
        Rotation.from_euler("xyz", [0,np.pi/2,0]),
        Rotation.from_euler("xyz", [0,np.pi,0]),
        Rotation.from_euler("xyz", [0,-np.pi/2,0]),
        Rotation.from_euler("xyz", [np.pi/2,0,0]),
        Rotation.from_euler("xyz", [-np.pi/2,0,0]),
    ]
    stable_placement_list = []
    for rot in rot_list:
        stable_placement_list.append(Transform(rot, [0,0,0]))
    return stable_placement_list

def sample_placement(
    top_body: Body, 
    bottom_body: Body, 
    placement_set: List[Transform],
    percent=0.9,
    max_attempt = 1000,
    epsilon: float = 1e-3
):
    for _ in range(max_attempt):
        yaw = np.random.uniform(0, np.pi*2)
        placement_random = np.random.choice(placement_set)
        obj_pose = Transform(Rotation.from_euler("xyz", [0,0,yaw])) * placement_random
        top_body.set_base_pose(obj_pose)
        bottom_aabb = bottom_body.get_AABB()
        center, extent = top_body.get_AABB(output_center_extent=True)
        
        xy_random = np.random.uniform(
            low=bottom_aabb[0]*percent, 
            high=bottom_aabb[1]*percent)[:2]
        z = bottom_aabb[1][-1] + extent[-1]/2 + epsilon
        orn = obj_pose.rotation
        obj_pose = Transform(orn, [*xy_random, z])
        top_body.set_base_pose(obj_pose)
        return obj_pose
    return None

def sample_grasp(
    placement: Transform,
    grasp_set: List[Grasp],
    obj: Body,
    hand: Gripper,
    check_world: BulletWorld,
    max_attempt = 100,
):
    orn = placement.rotation
    obj.set_base_pose(Transform(orn, [0,0,0]))
    plane = check_world.bodies['plane']
    _, upper = plane.get_AABB()
    _, extent = obj.get_AABB(output_center_extent=True)
    z = upper[-1] + extent[-1]/2
    obj_pose = Transform(orn, [0,0,z])
    obj.set_base_pose(obj_pose)
    
    for _ in range(max_attempt):
        grasp = np.random.choice(grasp_set) 
        hand.reset(obj_pose * grasp.T)
        if not check_world.get_contacts(body=hand.body):
            return grasp
    return None

def get_IK(
    robot: Panda, 
    obj_pose: Transform, 
    grasp: Grasp,
    is_collision: Callable[[Config1], bool],
    num_attempts: int = 10,
):
    def joint_interpolation(q_start, q_end, T_obj, is_collision):
        path = np.linspace(q_start, q_end, 10)
        for q in path:
            robot.set_arm_angles(q)
            if is_collision(q, T_obj):
                return None
        return path

    for _ in range(num_attempts):
        random_joint = np.random.uniform(low=robot.arm_lower_limit, high=robot.arm_upper_limit)
        robot.set_arm_angles(random_joint) #seed
        grasp_pose = obj_pose * grasp.T
        pre_pose = grasp_pose * Transform(Rotation.identity(), [0,0,-0.05])
        q_pre = robot.inverse_kinematics(pose=pre_pose)
        if (q_pre is None):
            continue
        if is_collision(q_pre, obj_pose):
            continue        
        q_grasp = robot.inverse_kinematics(pose=grasp_pose)
        if (q_grasp is None):
            continue
        if is_collision(q_grasp, obj_pose):
            continue
        path = joint_interpolation(q_pre, q_grasp, obj_pose, is_collision)
        if path is None:
            continue
        return (Config1(q=q_pre, grasp=grasp), path)

def is_collision_(q:np.ndarray, T_obj: Transform, world: BulletWorld, robot: Panda, object: Body):
    movable = [robot, object]
    robot.set_arm_angles(q)
    object.set_base_pose(T_obj)
    
    for o in movable:
        if world.get_contacts(body=o):
            return True
    return False

def get_free_motion(
    config1: Config1, 
    config2: Config1, 
    robot: Panda, 
    object: Body,
    is_collision: Callable[[Config1],bool]
):
    robot.set_arm_angles(config1.q)
    if config1.grasp is not None:
        T_obj = robot.get_ee_pose() * config1.grasp.T.inverse()
    else:
        T_obj = config1.T_obj
    is_col = partial(
        is_collision,
        T_obj=T_obj
    )
    object.set_base_pose(T_obj)
    get_random_config_fn = lambda : np.random.uniform(robot.arm_lower_limit, robot.arm_upper_limit)
    planner = BiRRT(
        start=config1.q,
        goal=config2.q,
        get_random_config=get_random_config_fn,
        is_collision=is_col,
    )
    path = planner.plan()
    return path

def get_holding_motion(
    config1: Config1, 
    config2: Config1, 
    robot: Panda, 
    object: Body,
    is_collision: Callable[[Config1],bool]
):
    robot.set_arm_angles(config1.q)
    if config1.grasp is not None:
        T_obj = robot.get_ee_pose() * config1.grasp.T.inverse()
    else:
        T_obj = config1.T_obj
    is_col = partial(
        is_collision,
        T_obj=T_obj
    )
    object.set_base_pose(T_obj)
    get_random_config_fn = lambda : np.random.uniform(robot.arm_lower_limit, robot.arm_upper_limit)
    planner = BiRRT(
        start=config1.q,
        goal=config2.q,
        get_random_config=get_random_config_fn,
        is_collision=is_col,
    )
    path = planner.plan()
    return path


def main():
    main_gui = True
    world, panda, scene_maker = set_world(gui=main_gui)
    box_pose_init = Transform(Rotation.identity(), [0.5, 0, 0.2+0.05+0.1])
    box = world.load_urdf("box", "data/urdfs/blocks/cuboid.urdf", box_pose_init, scale=1.1)
    world.wait_for_rest()
    #check world
    check_world, check_box, check_hand = set_check_world(gui=not main_gui)

    
    #scene_maker.view_frame(box.get_base_pose(), "frame_box")
    for T in get_stable_placements():
        box.set_base_pose(box_pose_init * T)
    
    table = world.bodies['table']
    table_aabb = table.get_AABB()

    #preloading placement, grasp set
    placement_set = get_stable_placements()
    grasp_set = np.load("grasp_set.npz", allow_pickle=True)["x"]
    grasp_set = [Grasp(T, i) for i, T in enumerate(grasp_set)]

    
    
    is_collision = partial(
        is_collision_, 
        world=world, robot=panda, object=box
    )
    
    results = []
    for i in range(10):
        pose = sample_placement(box, table, placement_set)
        grasp = sample_grasp(pose, grasp_set, check_box, check_hand, check_world)
        box.set_base_pose(pose)
        result = get_IK(panda, pose, grasp, is_collision)
        if result is not None:
            results.append(result)
    
    path = get_free_motion(results[0][0], results[1][0], panda, box, is_collision)
    
        
    
    #box.get_
    #box.set_base_pose()
    input()

if __name__ == "__main__":
    main()