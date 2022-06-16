from regrasp.utils.robot import Robot, Panda
from regrasp.utils.body import Body
from regrasp.planning.data import Grasp

class Attachment:
    """This class defines the kinematic relation of a manipulator and an object
    """
    def __init__(
        self, 
        parent: Robot,
        child: Body,
        grasp: Grasp
    ):
        self.parent = parent
        self.child = child
        self.grasp = grasp
    
    def assign(self):
        ee_pose = self.parent.get_ee_pose()
        child_pose = ee_pose * self.grasp.T.inverse()
        self.child.set_base_pose(child_pose)
