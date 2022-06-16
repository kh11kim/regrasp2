from regrasp.planning.config import *
from typing import List, Callable
from dataclasses import dataclass, field
from regrasp.utils.transform import Transform, Rotation, orn_error, slerp
import copy
from regrasp.utils.gripper import Gripper

def distance_ts(
    T1: Transform, 
    T2: Transform, 
    rot_weight=0.5
) -> float:
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
        root: Config
    ):
        root.index = 0
        self.V = [root] #vertice 
        self.parent = {} 
    
    def add_node(self, node: Config, parent: Config):
        assert parent.index is not None
        node.index = len(self.V)
        self.V.append(node)
        self.parent[node.index] = parent.index
    
    def nearest_tspace(self, T: Transform)->Config:
        distances = []
        for node in self.V:
            d = distance_ts(node.T, T)
            distances.append(d)
        return self.V[np.argmin(distances)]

    def nearest_cspace(self, q)->Config:
        distances = []
        for node in self.V:
            d = distance_cs(node.q, q)
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



class MotionPlanner:
    def __init__(
        self, 
        robot:Panda,
        world: BulletWorld,
        hand: Gripper,
        q_delta_max: float = 0.2,
        DLS_damping: float = 0.1,
        eps_ts: float = 0.05
    ):
        self.robot = robot
        self.q_delta_max = q_delta_max
        self.DLS_damping = DLS_damping
        self.eps_ts = eps_ts
        self.hand = hand
        self.world = world

    def plan_mode_switch(
        self,
        tree: Tree,
        kingraph: KinGraph,
        action_name: str,
        movable_name: str,
        placeable_name: Optional[str]=None,
        max_iter=1000,
        p_exploration=0.5
    ):
        """ randomized J+RRT
        """
        for _ in range(max_iter):
            p = np.random.random()
            if p < p_exploration:
                self.extend_randomly(tree, kingraph)
            else:
                if action_name == "move_free":
                    edge, T_target_pre = kingraph.sample_T_target_from_grasp(movable_name)
                    T_target = T_target_pre * Transform(translation=[0,0,0.05])
                elif action_name == "move_grasp":
                    edge, T_target_pre = kingraph.sample_T_target_from_placement(
                        movable_name, placeable_name)
                    T_target = Transform(translation=[0,0,-0.05]) * T_target_pre
                
                node_final = self.extend_to_pose(T_target_pre, tree, kingraph)
                if node_final:
                    if self.check_mode_switch(node_final, T_target, kingraph):
                        print("success")
                        return edge
        return None
    
    def check_mode_switch(
        self, 
        node_final: Config, 
        T_target: Transform,
        kingraph: KinGraph
    ):
        EPS = 0.01
        result = False
        with self.robot.no_set_joint():
            config = copy.deepcopy(node_final)
            for _ in range(5):
                config.q = self.steer(
                    q=config.q,
                    curr_pose=config.T,
                    target_pose=T_target,
                    q_delta_max=0.05
                )
                self.robot.set_arm_angles(config.q)
                config.T = self.robot.get_ee_pose()
                kingraph.assign_const()
                if not kingraph.is_collision(config):
                    #print(distance_ts(config.T, T_target))
                    if distance_ts(config.T, T_target) < EPS:
                        result = True
                        break
                else:
                    break
                
        kingraph.assign_const()
        return result
        


    def limit_step_size(self, q_delta: np.ndarray, q_delta_max: Optional[np.ndarray]=None):
        if q_delta_max is None:
            q_delta_max = self.q_delta_max
        mag = np.linalg.norm(q_delta, np.inf)
        if mag > q_delta_max:
            q_delta = q_delta / mag * q_delta_max
        return q_delta

    def steer(
        self,
        q: np.ndarray,
        curr_pose: Transform,
        target_pose: Transform,
        q_delta_max: Optional[float] = None
    ) -> np.ndarray:
        if q_delta_max is None:
            q_delta_max = self.q_delta_max
        pos_err = target_pose.translation - curr_pose.translation
        orn_err = orn_error(target_pose.rotation.as_quat(), curr_pose.rotation.as_quat())
        err = np.hstack([pos_err, orn_err*2])
        jac = self.robot.get_jacobian(q)
        lmbda = np.eye(6) * self.DLS_damping ** 2
        jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + lmbda)
        q_delta = self.limit_step_size(jac_pinv @ err, q_delta_max)
        return q_delta + q

    def extend_to_pose(self, T_target: Transform, tree: Tree, kingraph: KinGraph):
        def add_nodes(nodes:Config, parent:Config, tree:Tree):
            for node in nodes:
                tree.add_node(node, parent)
                parent = node

        EXTEND_IN_A_ROW = 3
        node_tree = tree.nearest_tspace(T_target)

        node_old = node_tree
        nodes = []
        for _ in range(EXTEND_IN_A_ROW):
            node_new = copy.deepcopy(node_old)
            node_new.q = self.steer(node_old.q, node_old.T, T_target)

            if not kingraph.is_collision(node_new):
                node_new.T = self.robot.forward_kinematics(node_new.q)
                nodes.append(node_new)

                if distance_ts(T_target, node_new.T) < self.eps_ts:
                    # goal(reached)
                    node_final = copy.deepcopy(node_new)
                    node_final.q = self.steer(node_new.q, node_new.T, T_target)
                    nodes.append(node_final)
                    add_nodes(nodes, node_tree, tree)
                    return node_final
                
                node_old = node_new
            else:
                # if collision, reject all extension
                return None

        # advanced
        add_nodes(nodes, node_tree, tree)
        return None


    def extend_randomly(self, tree: Tree, kingraph: KinGraph):
        def get_random_joint():
            return np.random.uniform(
                low=self.robot.arm_lower_limit,
                high=self.robot.arm_upper_limit
            )
        
        q_target = get_random_joint()
        node_tree = tree.nearest_cspace(q_target)
        q_delta = self.limit_step_size(q_target - node_tree.q)
        node_new = copy.deepcopy(node_tree)
        node_new.q = node_tree.q + q_delta
        if not kingraph.is_collision(node_new):
            node_new.T = self.robot.forward_kinematics(node_new.q)
            tree.add_node(node_new, node_tree)
        