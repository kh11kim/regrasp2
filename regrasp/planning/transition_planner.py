import numpy as np
from dataclasses import dataclass, field
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.transform import Transform, Rotation, orn_error, slerp
from regrasp.planning.data import Mode, Config, Grasp, DualArm, Hand
from functools import partial
from typing import Callable, List, Optional, Tuple
from regrasp.utils.body import Body
from regrasp.utils.robot import Panda
from copy import deepcopy
from time import time

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
        root: Config, 
        dist_ts: Callable[[Transform, Transform],float] = distance_ts,
        dist_cs: Callable[[np.ndarray, np.ndarray],float] = distance_cs
    ):
        root.index = 0
        self.V = [root] #vertice 
        #edge information
        self.parent = {} 
        self.dist_ts = dist_ts
        self.dist_cs = dist_cs
    
    def add_node(self, node: Config, parent: Config):
        assert parent.index is not None
        node.index = len(self.V)
        self.V.append(node)
        self.parent[node.index] = parent.index
    
    def nearest_constraint(self, T_l_r: Transform) -> Config:
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

class TransitionPlanner:
    def __init__(
        self,
        robots: DualArm, #fk, jacobian
        sm: BulletSceneMaker,
        is_collision: Callable[[Config],bool], #collision checking,
        is_grasp_collision: Callable[[Grasp, Grasp], bool],#feasible mode checking
        p_global_explore: float = 0.5,
        p_constraint_explore: float = 0.1,
    ):
        self.robots = robots
        self.sm = sm
        self.is_collision = is_collision
        self.is_grasp_collision = is_grasp_collision
        self.p_global_explore = p_global_explore
        self.p_constraint_explore = p_constraint_explore
        
    def plan(
        self,
        config_start: Config,
        goal_mode_set: List[Mode],
        mode_set: List[Mode]
    ):
        print(f"start with hand: {config_start.mode.hand}, grasp: {config_start.mode.grasp.index}")
        tree = Tree(config_start)
        mode_curr = config_start.mode
        tic = time()
        feasible_mode_set = self.check_next_mode_feasibility(mode_curr, mode_set)
        elapsed_next_mode = time()
        goal_mode_set = [mode for mode in feasible_mode_set if mode in goal_mode_set]
        config_trans = None
        for _ in range(1000):
            p1, p2 = np.random.random(2)
            if p1 < self.p_global_explore:
                self.extend_randomly(tree)
            else:
                if (p2 < self.p_constraint_explore) | (len(goal_mode_set)==0):
                    mode_target = np.random.choice(feasible_mode_set)
                else:
                    mode_target = np.random.choice(goal_mode_set)
                config_trans = self.extend_to_constraint(tree, mode_curr, mode_target)
            if config_trans is not None:
                elapsed = time() - tic
                print(f"transition to hand: {mode_target.hand}, grasp: {mode_target.grasp.index}")
                print(f"elapsed : {elapsed} elapsed_next_mode_check:{elapsed_next_mode - tic}")
                break
        return config_trans, mode_target
        

    def check_next_mode_feasibility(
        self, 
        mode_curr: Mode, 
        mode_set: List[Mode]
    ):
        next_mode_set = [m for m in mode_set if m.hand != mode_curr.hand]
        feasible_mode_set = []
        for mode_candidate in next_mode_set:
            if not self.is_grasp_collision(mode_curr.grasp, mode_candidate.grasp):
                feasible_mode_set.append(mode_candidate)
        indexes = [m.grasp.index for m in feasible_mode_set]
        #print(f"mode:{indexes}")
        return feasible_mode_set

    def extend_randomly(
        self,
        tree: Tree
    ):
        q_target_l = self.get_random_joint()
        q_target_r = self.get_random_joint()
        node_tree = tree.nearest_cs(q_target_l, q_target_r)
        q_l, q_r = node_tree.q_l, node_tree.q_r

        is_left_advanced = is_right_advanced = True
        node_new = deepcopy(node_tree)
        node_new.q_l = self.steer_cs(q_target_l, node_tree.q_l)
        if self.is_collision(node_new):
            node_new.q_l = q_l
            is_left_advanced = False
        node_new.q_r = self.steer_cs(q_target_r, node_tree.q_r)
        if self.is_collision(node_new):
            node_new.q_r = q_r
            is_right_advanced = False
        
        if (is_left_advanced == False) & (is_right_advanced == False):
            return None
        
        node_new.T_l = self.robots.left.forward_kinematics(node_new.q_l)
        node_new.T_r = self.robots.right.forward_kinematics(node_new.q_r)
        tree.add_node(node_new, node_tree)

    def extend_to_constraint(
        self,
        tree: Tree,
        mode_curr: Mode,
        mode_target: Mode
    ) -> Optional[Config]:
        
        if mode_curr.hand == Hand.OBJ_IN_RIGHT:
            grasp_r, grasp_l = mode_curr.grasp, mode_target.grasp
            constraint_l = self.get_pre_grasp(grasp_l)
            constraint_r = grasp_r
        else:
            grasp_l, grasp_r = mode_curr.grasp, mode_target.grasp
            constraint_l = grasp_l
            constraint_r = self.get_pre_grasp(grasp_r)
        
        Tconstraint_l_r = constraint_l.T.inverse() * constraint_r.T
        node_tree = tree.nearest_constraint(Tconstraint_l_r)

        node_old = node_tree
        nodes = []
        is_goal = False
        for _ in range(4):
            Tobj_l = node_old.T_l * grasp_l.T.inverse()
            Tobj_r = node_old.T_r * grasp_r.T.inverse()
            pos_mid = Tobj_l.translation + (Tobj_r.translation - Tobj_l.translation)/2
            orn_mid = slerp(Tobj_l.rotation.as_quat(), Tobj_r.rotation.as_quat(), 0.5)
            obj_mid = Transform(Rotation.from_quat(orn_mid), pos_mid)

            T_target_l = obj_mid * constraint_l.T
            T_target_r = obj_mid * constraint_r.T
            
            node_new = self.move_arm(node_old, T_target_l, T_target_r)
            
            if node_new is None:
                return None

            node_new.T_l = self.robots.left.forward_kinematics(node_new.q_l)
            node_new.T_r = self.robots.right.forward_kinematics(node_new.q_r)
            nodes.append(node_new)

            
            #debug
            T_pre_to_grasp = Transform(Rotation.identity(), [0,0,0.05])
            if mode_curr.hand == Hand.OBJ_IN_RIGHT:
                grasp_new_l = node_new.T_l
                grasp_new_r = node_new.T_r
            else:
                grasp_new_l = node_new.T_l
                grasp_new_r = node_new.T_r           
            self.sm.view_frame(grasp_new_l, "left_Target")
            self.sm.view_frame(grasp_new_r, "right_Target")
            #debug2
            if mode_curr.hand == Hand.OBJ_IN_RIGHT:
                grasp_target_view = node_new.T_r * Tconstraint_l_r.inverse()
            else:
                grasp_target_view = node_new.T_l * Tconstraint_l_r
            self.sm.view_frame(grasp_target_view, "grasp_target")

            if distance_ts(node_new.T_l*Tconstraint_l_r, node_new.T_r) < 0.01:
                #final checking
                for _ in range(3):
                    node_final = self.move_arm(node_new, obj_mid * grasp_l.T, obj_mid * grasp_r.T) #T_target_l, T_target_r)
                    nodes.append(node_final)
                if node_final is not None:
                    #for _ in range(3):
                        #node_final = self.move_arm(node_new, obj_mid * grasp_l.T, obj_mid * grasp_r.T) #T_target_l, T_target_r)
                        #nodes.append(node_final)
                    is_goal = True
                    break
            node_old = node_new

        node_prev = node_tree
        for node in nodes:
            tree.add_node(node, node_prev)
            node_prev = node
        return node_new if is_goal else None
        
    def move_arm(
        self,
        node_old:Config,
        T_target_l: Transform,
        T_target_r: Transform
    ) -> Config:
        is_left_advanced = is_right_advanced = True
        node_new = deepcopy(node_old)

        node_new.q_l = self.steer_ts(
            node_old.q_l, T_target_l, node_old.T_l,
            self.robots.left
        )
        if self.is_collision(node_new):
            node_new.q_l = node_old.q_l
            is_left_advanced = False

        node_new.q_r = self.steer_ts(
            node_old.q_r, T_target_r, node_old.T_r, 
            self.robots.right
        )
        if self.is_collision(node_new):
            node_new.q_r = node_old.q_r
            is_right_advanced = False
        
        if (is_left_advanced == False) & (is_right_advanced == False):
            return None
        return node_new

    def steer_ts(
        self, 
        q: np.ndarray,
        T_target: Transform,  
        T: Transform, 
        robot: Panda
    ):
        damping = 0.1
        pos_err = T_target.translation - T.translation
        orn_err = orn_error(T_target.rotation.as_quat(), T.rotation.as_quat())
        err = np.hstack([pos_err, orn_err*1.5])
        jac = robot.get_jacobian(q)
        lmbda = np.eye(6) * damping ** 2
        jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + lmbda)
        q_delta = self.limit_step_size(jac_pinv @ err)
        return q_delta + q

    def limit_step_size(self, q_delta: np.ndarray, max_delta_mag=0.2):
        mag = np.linalg.norm(q_delta)
        if mag > max_delta_mag:
            q_delta = q_delta/mag*max_delta_mag
        return q_delta

    def get_pre_grasp(self, grasp: Grasp, z_dist=-0.05) -> Grasp:
        T = grasp.T * Transform(Rotation.identity(), [0,0,z_dist])
        return Grasp(T, grasp.index)

    def get_random_joint(self):
        return np.random.uniform(
            low=self.robots.left.arm_lower_limit,
            high=self.robots.left.arm_upper_limit
        )
    
    def steer_cs(self, q_target: np.ndarray, q: np.ndarray):
        mag = np.linalg.norm(q_target - q)
        if mag < 0.2:
            q_delta = q_target - q
        else:
            q_delta = (q_target - q) / mag * 0.2
        return q_delta + q
    
    def steer_jacobian(self):
        pass