import numpy as np
from pybullet_utils.bullet_client import BulletClient
from regrasp.utils.transform import Rotation, Transform

JOINT_ATRTIBUTE_NAMES = \
    ["joint_index","joint_name","joint_type",
    "q_index", "u_index", "flags", 
    "joint_damping", "joint_friction","joint_lower_limit",
    "joint_upper_limit","joint_max_force","joint_max_velocity",
    "link_name","joint_axis","parent_frame_pos","parent_frame_orn","parent_index"]

class Body:
    def __init__(self, physics_client: BulletClient, body_uid: int):
        self.physics_client = physics_client
        self.uid = body_uid
        self.info = {}
        self.n_joints = self.physics_client.getNumJoints(self.uid)
        for i in range(self.n_joints):
            joint_info = self.physics_client.getJointInfo(self.uid, i)
            self.info[i] = {name:value for name, value in zip(JOINT_ATRTIBUTE_NAMES, joint_info)}

    @classmethod
    def from_urdf(
        cls, 
        physics_client: BulletClient, 
        urdf_path: str, 
        pose: Transform = Transform.identity(), 
        use_fixed_base: bool = False,
        scale: float = 1.0
    ):
        body_uid = physics_client.loadURDF(
            fileName=str(urdf_path),
            basePosition=pose.translation,
            baseOrientation=pose.rotation.as_quat(),
            useFixedBase=use_fixed_base,
            globalScaling=scale,
            flags=physics_client.URDF_USE_SELF_COLLISION
        )
        return cls(physics_client, body_uid)
    
    def get_base_pose(self):
        pos, orn = self.physics_client.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(orn), np.asarray(pos))

    def set_base_pose(self, pose: Transform):
        self.physics_client.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )
    
    def get_base_velocity(self):
        linear, angular = self.physics_client.getBaseVelocity(self.uid)
        return linear, angular
    
    def get_link_pose(self, link: int):
        assert self.n_joints > 0, "This body has no link index: use get_base_pose instead"
        pos, orn = self.physics_client.getLinkState(self.uid, link)[:2]
        return Transform.from_dict({
            "rotation":orn, "translation":pos
        })
    
    def get_AABB(self, output_center_extent=False):
        """base"""
        lower, upper = self.physics_client.getAABB(self.uid, linkIndex=-1)
        lower, upper = np.array(lower), np.array(upper)
        if output_center_extent:
            center = np.mean(lower+upper)
            extent = upper - lower
            return center, extent
        return lower, upper

    def get_link_velocity(self, link: int):
        pass #TODO

    def get_joint_angle(self, joint: int):
        assert self.n_joints > 0, "This body has no joint index"
        return self.physics_client.getJointState(self.uid, joint)[0]
    
    def get_joint_angles(self):
        assert self.n_joints > 0, "This body has no joint index"
        joint_angles = []
        for i in range(self.n_joints):
            joint_angles.append(self.get_joint_angle(i))
        return np.asarray(joint_angles)
    
    def get_joint_velocity(self, joint: int):
        assert self.n_joints > 0, "This body has no joint index"
        return np.asarray(self.physics_client.getJointState(self.uid, joint)[1])
    
    def set_joint_angle(self, joint: int, angle: float):
        assert self.n_joints > 0, "This body has no joint index"
        self.physics_client.resetJointState(self.uid, jointIndex=joint, targetValue=angle)
    
    def set_joint_angles(self, angles: np.ndarray):
        assert self.n_joints > 0, "This body has no joint index"
        assert len(angles) == self.n_joints
        for i, angle in zip(range(self.n_joints), angles):
            self.set_joint_angle(joint=i, angle=angle)  
    
    def set_stable_z(self):
        _, extent = self.get_AABB(output_center_extent=True)
        pose = self.get_base_pose()
        pose.translation[-1] = extent[-1]/2
        self.set_base_pose(pose)