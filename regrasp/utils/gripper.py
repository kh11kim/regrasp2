from regrasp.utils.world import BulletWorld
from regrasp.utils.body import Body
from regrasp.utils.transform import Rotation, Transform

class Gripper:
    def __init__(self, world: BulletWorld):
        self.world = world
        self.body = None
        self.urdf_path = "data/urdfs/panda/hand.urdf"
        self.max_opening_width = 0.08
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.103])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp: Transform, name=None):
        if name is None:
            self.name = "hand"
        else:
            self.name = name
        T_world_body = T_world_tcp * self.T_tcp_body
        if not self.name in self.world.bodies:
            self.body = self.world.load_urdf(self.name, self.urdf_path, T_world_body)
        else:
            self.body.set_base_pose(T_world_body)
        self.grip()
    
    def get_tcp_pose(self):
        T_world_body = self.body.get_base_pose()
        return T_world_body * self.T_body_tcp

    def remove(self):
        self.world.remove_body(name=self.name)

    def detect_contact(self):
        self.world.step(only_collision_detection=True)
        if self.world.get_contacts(body=self.body):
            return True
        return False
    
    def grip(self, width=None):
        assert self.body is not None
        if width is None:
            width = self.max_opening_width
        self.body.set_joint_angle(0, width / 2)
        self.body.set_joint_angle(1, width / 2)

    def close(self):
        assert self.body is not None
        self.body.set_joint_angle(0, 0)
        self.body.set_joint_angle(1, 0)

if __name__ == "__main__":
    world = BulletWorld(gui=True)
    hand = Gripper(world)
    T_tcp = Transform(Rotation.identity(), [0,0,0.5])
    hand.reset(T_tcp)
    hand.grip()
    hand.grip(0)
    