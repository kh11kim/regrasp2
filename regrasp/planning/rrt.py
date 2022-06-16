import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class Node:
    x: np.ndarray
    parent = None

class Tree:
    def __init__(self, root: Node):
        self.root = root
        self.data = [root]
    
    def add_node(self, node: None, parent: Node):
        node.parent = parent
        self.data.append(node)
    
    def nearest(self, node):
        distances = []
        for node_tree in self.data:
            d = np.linalg.norm(node_tree.x - node.x)
            distances.append(d)
        idx = np.argmin(distances)
        return self.data[idx]
    
    def backtrack(self, node):
        path = [node.x]
        parent = node.parent
        while True:
            if parent is None:
                break
            path.append(parent.x)
            parent = parent.parent
        return path[::-1]

class BiRRT:
    def __init__(
        self, 
        start: np.ndarray,
        goal: np.ndarray,
        get_random_config: Callable[[],np.ndarray],
        is_collision: Callable[[np.ndarray], bool],
        eps: float = 0.2,
        p_goal: float = 0.2,
        max_iter: int = 10000
    ):
        self.start = Node(start)
        self.goal = Node(goal)
        self.eps = eps
        self.p_goal = p_goal
        self.max_iter = max_iter

        self.tree_start = Tree(self.start)
        self.tree_goal = Tree(self.goal)
        self.is_collision = is_collision
        self.is_goal = lambda node: self.distance(node, self.goal) < self.eps
        self.get_random_config = get_random_config
        self.get_random_node = lambda : Node(self.get_random_config())

    def connect(self, tree, node):
        result = "advanced"
        while result == "advanced":
            result = self.extend(tree, node)
        return result

    def distance(self, node1:Node, node2:Node):
        return np.linalg.norm(node1.x - node2.x)

    def extend(self, tree, node_rand):
        node_near = tree.nearest(node_rand)
        node_new = self.control(node_near, node_rand)
        if node_new is not None:
            if not self.distance(node_rand, node_new) < self.eps: #self.is_goal(node_new):
                # to prevent node is in both tree, add_node() is after IF statement
                tree.add_node(node_new, node_near)
                self._node_new = node_new
            else:
                self.last_node = node_new
                return "reached"
            return "advanced"
        return "trapped"
    
    def control(self, node_near:Node, node_rand:Node):
        mag = self.distance(node_near, node_rand)
        if mag <= self.eps:
            node_new = node_rand
        else:
            x_new = node_near.x + (node_rand.x - node_near.x) /mag * self.eps
            node_new = Node(x_new)

        if not self.is_collision(node_new.x):
            return node_new
        else:
            return None

    def plan(self):
        tree_a = self.tree_start
        tree_b = self.tree_goal
        for i in range(self.max_iter):
            # if np.random.uniform(0, 1) < self.p_goal:
            #     node_rand = self.goal
            # else:
            node_rand = self.get_random_node()
            
            if not self.extend(tree_a, node_rand) == "trapped":
                if self.connect(tree_b, self._node_new) == "reached":
                    return self.get_path()
            (tree_a, tree_b) = (tree_b, tree_a)
        return []
    
    def get_path(self):
        node_tree_start = self.tree_start.nearest(self.last_node)
        node_tree_goal = self.tree_goal.nearest(self.last_node)
        path_from_start = self.tree_start.backtrack(node_tree_start)
        path_from_goal = self.tree_goal.backtrack(node_tree_goal)
        return [*path_from_start, *path_from_goal[::-1]]
