import numpy as np
from dataclasses import dataclass, field
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.transform import Transform, Rotation, orn_error, slerp
from regrasp.planning.data import Mode, Config, Grasp, DualArm
from functools import partial
from typing import Callable, List, Optional
from regrasp.utils.body import Body
from regrasp.utils.robot import Panda


@dataclass
class ModeNode:
    """Node for Global planner
    """
    mode: Mode
    grasp: Grasp
    config_start: Optional[Config] = field(default_factory=lambda: None)
    grasp_handover: Optional[Grasp] = field(default_factory=lambda: None)
    config_handover: Optional[Config] = field(default_factory=lambda: None)
    trajectory: List = field(default_factory=lambda: []) #not registered
    index: int = field(default_factory=lambda: -1) #not registered

@dataclass
class Node:
    """Node for TS-RRT
    """
    q_l: np.ndarray
    q_r: np.ndarray
    T_l: Transform
    T_r: Transform
    T_obj: Transform
    mode: Mode
    grasp_obj: Grasp
    index: int = field(default_factory=lambda: -1) #not registered

def distance_ts(
    T1: Transform, 
    T2: Transform, 
    rot_weight=0.5
) -> float:
    """Distance between two transformations in task-space

    Args:
        T1 (Transform): transform class
        T2 (Transform): transform class
        rot_weight (float, optional): weight to rotation distance. Defaults to 0.5.

    Returns:
        float: distance
    """

    linear = np.linalg.norm(T1.translation - T2.translation)
    qtn1, qtn2 = T1.rotation.as_quat(), T2.rotation.as_quat()
    if qtn1 @ qtn2 < 0:
        qtn2 = -qtn2
    angular = np.arccos(np.clip(qtn1 @ qtn2, -1, 1))
    #print(f"distance - linear: {linear} angular: {angular}")
    return linear + rot_weight * angular

def distance_cs(
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    return np.linalg.norm(q1 - q2)

class Tree:
    """Tree structure for TS-RRT
    """
    def __init__(
        self, 
        root: Node, 
        dist_ts: Callable[[Transform, Transform],float] = distance_ts,
        dist_cs: Callable[[np.ndarray, np.ndarray],float] = distance_cs
    ):
        root.index = 0
        self.V = [root] #vertice 
        #edge information
        self.parent = {} 
        #self.child = {}
        self.dist_ts = dist_ts
        self.dist_cs = dist_cs
    
    def add_node(self, node: Node, parent: Node):
        assert parent.index is not None
        node.index = len(self.V)
        self.V.append(node)
        self.parent[node.index] = parent.index
        # if parent.index in self.child:
        #     self.child[parent.index].append(node.index)
        # else:
        #     self.child[parent.index] = [node.index]
    
    def nearest_constraint(self, T_l_r: Transform):
        distances = []
        for node in self.V:
            d = self.dist_ts(node.T_l * T_l_r, node.T_r)
            distances.append(d)
        return self.V[np.argmin(distances)]

    def nearest_cs(self, q_l, q_r):
        distances = []
        for node in self.V:
            q_rand = np.hstack([q_l, q_r])
            q = np.hstack([node.q_l, node.q_r])
            d = self.dist_cs(q, q_rand)
            distances.append(d)
        return self.V[np.argmin(distances)]
    
    def backtrack(self, node):
        path = [node]
        parent_idx = self.parent[node.index]
        while True:
            if parent_idx == 0:
                break
            path.append(self.V[parent_idx])
            parent_idx = self.parent[parent_idx]
        return path[::-1]

def handover_tsrrt_(
    mode_node: ModeNode,
    goal_grasp_set: List[Grasp],
    obj: Body,
    robots: DualArm,
    sm: BulletSceneMaker,
    get_feasible_grasp_set: Callable[[Grasp], List[Grasp]],
    is_collision_: Callable
):  
    # Input
    q_l, q_r = mode_node.config_start.q_l, mode_node.config_start.q_r
    T_l = robots.left.forward_kinematics(q_l)
    T_r = robots.right.forward_kinematics(q_r)
    T_obj = mode_node.config_start.T_obj
    grasp_obj = mode_node.grasp
    mode = mode_node.mode

    # RRT init
    node_init = Node(q_l, q_r, T_l, T_r, T_obj, mode, grasp_obj)
    tree = Tree(node_init, distance_ts, distance_cs)

    is_collision = partial(
        is_collision_,
        grasp_obj=grasp_obj,
        mode=mode_node.mode
    )
    extend_to_goal = partial(
        extend_to_goal_,
        mode=mode,
        tree=tree,
        grasp_obj=grasp_obj,
        robots=robots,
        sm=sm,
        is_collision=is_collision
    )
    extend_randomly = partial(
        extend_randomly_,
        mode=mode,
        grasp_obj=grasp_obj,
        tree=tree,
        robots=robots,
        is_collision=is_collision,
    )

    # make feasible grasp set
    feasible_grasp_set = get_feasible_grasp_set(mode_node.grasp)
    feasible_grasp_index = [grasp.index for grasp in feasible_grasp_set]
    goal_grasp_set = [grasp for grasp in goal_grasp_set if grasp.index in feasible_grasp_index]
    goal_grasp_index = [grasp.index for grasp in goal_grasp_set]

    # Plan
    result = None
    for _ in range(1000):
        p_explore, p_goal = np.random.random(2)
        if p_explore < 0.5:
            extend_randomly()
        else:
            if (p_goal < 0.9) & (len(goal_grasp_set) != 0):
                rand_grasp = np.random.choice(goal_grasp_set)
            else:
                rand_grasp = np.random.choice(feasible_grasp_set)
            result = extend_to_goal(rand_grasp=rand_grasp)

        if result is not None:
            (handover_node, handover_grasp) = result
            mode_node.grasp_handover = handover_grasp
            mode_node.config_handover = Config(
                q_l=handover_node.q_l,
                q_r=handover_node.q_r,
                T_obj=handover_node.T_obj
            )
            mode_node.trajectory = tree.backtrack(handover_node)
            return mode_node

def extend_to_goal_(
    rand_grasp: Grasp,
    mode: Mode,
    tree: Tree,
    grasp_obj: Grasp,
    robots: DualArm,
    sm: BulletSceneMaker,
    is_collision: Callable[[np.ndarray, np.ndarray], bool],
):
    if Mode.OBJ_IN_LEFT:
        grasp_l = get_pre_grasp(grasp_obj)
        grasp_r = get_pre_grasp(rand_grasp)
    else:
        grasp_l = get_pre_grasp(rand_grasp)
        grasp_r = get_pre_grasp(grasp_obj)
    
    #get tree node
    T_l_r = grasp_l.T.inverse() * grasp_r.T
    node_tree = tree.nearest_constraint(T_l_r)

    node_old = node_tree
    nodes = []
    for _ in range(3):
        q_l, q_r = node_old.q_l, node_old.q_r
        T_l, T_r = node_old.T_l, node_old.T_r
        T_obj = node_old.T_obj

        # make target Transformation
        obj_l = node_old.T_l * grasp_l.T.inverse()
        obj_r = node_old.T_r * grasp_r.T.inverse()
        pos_mid = obj_l.translation + (obj_r.translation - obj_l.translation)/2
        orn_mid = slerp(obj_l.rotation.as_quat(), obj_r.rotation.as_quat(), 0.5)
        obj_mid = Transform(Rotation.from_quat(orn_mid), pos_mid)
    
        T_target_l = obj_mid * grasp_l.T
        T_target_r = obj_mid * grasp_r.T
        sm.view_frame(T_target_l, "left_Target")
        sm.view_frame(T_target_r, "right_Target")
        
        q_l_new = steer(q_l, T_l, T_target_l, robots.left.get_jacobian)
        # if collision, go back
        is_left_advanced = is_right_advanced = False
        if not is_collision(q_l=q_l_new, q_r=q_r):
            q_l = q_l_new
            is_left_advanced = True

        q_r_new = steer(q_r, T_r, T_target_r, robots.right.get_jacobian)
        if not is_collision(q_l=q_l, q_r=q_r_new):
            q_r = q_r_new
            is_right_advanced = True
        
        if (is_left_advanced == False) & (is_right_advanced == False):
            return None

        T_l = robots.left.forward_kinematics(q_l)
        T_r = robots.right.forward_kinematics(q_r)
        T_ee = T_l if mode == Mode.OBJ_IN_LEFT else T_r
        T_obj = T_ee * grasp_obj.T.inverse()
        
        node_new = Node(
            q_l=q_l,
            q_r=q_r,
            T_l=T_l,
            T_r=T_r,
            T_obj=T_obj,
            mode=mode,
            grasp_obj=grasp_obj
        )
        nodes.append(node_new)
        
        if distance_ts(T_l*T_l_r, T_r) < 0.1:
            #goal
            node_prev = node_tree
            for node in nodes:
                tree.add_node(node, node_prev)
                node_prev = node
            return (node_new, rand_grasp) # handover grasp
        else:
            node_old = node_new
    node_prev = node_tree
    for node in nodes:
        tree.add_node(node, node_prev)
        node_prev = node
    return None

def extend_randomly_(
    mode: Mode,
    grasp_obj: Grasp,
    tree: Tree,
    robots: DualArm, 
    is_collision: Callable[[np.ndarray, np.ndarray], bool],
):
    def get_random_joint():
        return np.random.uniform(
            low=robots.left.arm_lower_limit,
            high=robots.left.arm_upper_limit
        )
    def steer_cs(q, q_target):
        mag = np.linalg.norm(q_target - q)
        if mag < 0.2:
            q_delta = q_target - q
        else:
            q_delta = (q_target - q) / mag * 0.2
        return q_delta + q
    
    
    q_target_l = get_random_joint()
    q_target_r = get_random_joint()
    node_tree = tree.nearest_cs(q_target_l, q_target_r)
    q_l, q_r = node_tree.q_l, node_tree.q_r

    is_left_advanced = is_right_advanced = False
    # left
    q_l_new = steer_cs(node_tree.q_l, q_target_l)
    if not is_collision(q_l_new, node_tree.q_r):
        q_l = q_l_new
        is_left_advanced = True
    q_r_new = steer_cs(node_tree.q_r, q_target_r)
    if not is_collision(q_l, q_r_new):
        q_r = q_r_new
        is_right_advanced = True
    
    if (is_left_advanced == False) & (is_right_advanced == False):
        return None
    else:
        T_l = robots.left.forward_kinematics(q_l)
        T_r = robots.right.forward_kinematics(q_r)
        T_ee = T_l if mode == Mode.OBJ_IN_LEFT else T_r
        T_obj = T_ee * grasp_obj.T.inverse()
        node_new = Node(
            q_l=q_l, 
            q_r=q_r, 
            T_l=T_l, 
            T_r=T_r, 
            T_obj=T_obj,
            mode=mode,
            grasp_obj=grasp_obj
        )
        tree.add_node(node_new, node_tree)
    

def steer(
    q: np.ndarray,
    curr_pose: Transform,
    target_pose: Transform, 
    get_jacobian: Callable[[np.ndarray],np.ndarray]
):
    damping = 0.1
    pos_err = target_pose.translation - curr_pose.translation
    orn_err = orn_error(target_pose.rotation.as_quat(), curr_pose.rotation.as_quat())
    err = np.hstack([pos_err, orn_err*1.5])
    jac = get_jacobian(q)
    lmbda = np.eye(6) * damping ** 2
    jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + lmbda)
    q_delta = limit_step_size(jac_pinv @ err)
    return q_delta + q

def limit_step_size(q_delta: np.ndarray, max_delta_mag=0.2):
    mag = np.linalg.norm(q_delta)
    if mag > max_delta_mag:
        q_delta = q_delta/mag*max_delta_mag
    return q_delta

def get_pre_grasp(grasp: Grasp):
    T = grasp.T * Transform(Rotation.identity(), [0,0,-0.05])
    return Grasp(T, grasp.index)