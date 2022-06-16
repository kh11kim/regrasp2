from pybullet_utils.bullet_client import BulletClient
#from regrasp.utils.world import BulletWorld
from regrasp.utils.body import Body
from regrasp.utils.transform import Rotation, Transform
import numpy as np
from typing import Optional, Union
from contextlib import contextmanager

class Robot(Body):
    def __init__(
        self, 
        physics_client: BulletClient, 
        body_uid: int,
        ee_idx: Optional[int],
    ):
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid
        )
        self.ee_idx = ee_idx if ee_idx is not None else (self.n_joints - 1)
        self.set_joint_limits()
        self.set_joint_angles(self.joint_central)
    
    def set_joint_limits(self):
        ll_list = []
        ul_list = []
        mid_list = []
        for joint in self.info:
            ll = self.info[joint]["joint_lower_limit"]
            ul = self.info[joint]["joint_upper_limit"]
            mid = (ll + ul) / 2
            ll_list.append(ll)
            ul_list.append(ul)
            mid_list.append(mid)
        self.joint_lower_limit = np.asarray(ll_list)
        self.joint_upper_limit = np.asarray(ul_list)
        self.joint_central = np.asarray(mid_list)

    def get_ee_pose(self):
        return self.get_link_pose(self.ee_idx)

    @contextmanager
    def no_set_joint(self, no_viz=False):
        if no_viz:
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_RENDERING, 
                0
            )
        joints_temp = self.get_joint_angles()
        yield
        self.set_joint_angles(joints_temp)
        if no_viz:
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_RENDERING, 
                1
            )
    
    @classmethod
    def make(
        cls,
        physics_client: BulletClient,
        pose: Transform = Transform.identity(),
        use_fixed_base: bool = True
    ):
        urdf_path = "data/urdfs/panda/franka_panda.urdf"
        body_uid = physics_client.loadURDF(
            urdf_path,
            pose.translation,
            pose.rotation.as_quat(),
            useFixedBase=use_fixed_base
        )
        return cls(physics_client, body_uid)

class Panda(Robot):
    def __init__(self, physics_client: BulletClient, body_uid: int):
        self.arm_idxs = range(7)
        self.finger_idxs = [8, 9]
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            ee_idx = 10
        )
        self.max_opening_width = 0.08
        self.arm_lower_limit = self.joint_lower_limit[self.arm_idxs]
        self.arm_upper_limit = self.joint_upper_limit[self.arm_idxs]
        self.arm_central = (self.arm_lower_limit + self.arm_upper_limit)/2
        self.open()

    def inverse_kinematics(
        self, 
        pos: np.ndarray = None, 
        pose: Transform = None,
        tol: float = 1e-4,
        max_iter: int = 10
    ):
        assert (pos is None) ^ (pose is None)
        orn = None
        success = False
        if pose is not None:
            pos, orn = pose.translation, pose.rotation.as_quat()
        with self.no_set_joint():
            for i in range(max_iter):
                joint_angles = self.physics_client.calculateInverseKinematics(
                    bodyIndex=self.uid,
                    endEffectorLinkIndex=self.ee_idx,
                    targetPosition=pos,
                    targetOrientation=orn
                )
                self.set_arm_angles(joint_angles)
                pose_curr = self.get_ee_pose()
                if np.linalg.norm(pose_curr.translation - pos) < tol:
                    success = True
                    break
        if success:
            return np.array(joint_angles)[self.arm_idxs]
        return None

    def forward_kinematics(
        self,
        angles: np.ndarray
    ):
        with self.no_set_joint():
            self.set_arm_angles(angles)
            pose = self.get_ee_pose()
        return pose
    
    def get_jacobian(
        self, 
        joint_angles: Optional[np.ndarray] = None,
        local_position: Union[list, np.ndarray] = [0,0,0]
    ):
        if joint_angles is not None:
            assert len(self.arm_idxs) == len(joint_angles)
            joint_angles = np.asarray([*joint_angles, 0, 0])
        if joint_angles is None:
            joint_angles = self.get_joint_angles()
        jac_trans, jac_rot = self.physics_client.calculateJacobian(
            bodyUniqueId=self.uid,
            linkIndex=self.ee_idx,
            localPosition=local_position,
            objPositions=joint_angles.tolist(),
            objVelocities=np.zeros_like(joint_angles).tolist(),
            objAccelerations=np.zeros_like(joint_angles).tolist()
        )
        return np.vstack([jac_trans, jac_rot])[:,:-2]

    def get_arm_angles(self):
        return self.get_joint_angles()[self.arm_idxs]

    def set_arm_angles(self, angles: np.ndarray):
        for i, angle in zip(self.arm_idxs, angles):
            self.set_joint_angle(joint=i, angle=angle) 
    
    def open(self, ctrl=False):
        if ctrl == False:
            for i in self.finger_idxs:
                self.set_joint_angle(joint=i, angle=self.max_opening_width/2)
    
    def close(self, ctrl=False):
        if ctrl == False:
            for i in self.finger_idxs:
                self.set_joint_angle(joint=i, angle=0)

    # def reset(self, T_world_tcp: Transform, name=None):
    #     if name is None:
    #         name = "hand"
    #     T_world_body = T_world_tcp * self.T_tcp_body
    #     self.body = self.world.load_urdf(name, self.urdf_path, T_world_body)

    # def detect_contact(self):
    #     if self.world.get_contacts(self.body):
    #         return True
    #     return False
    
    # def grip(self, width=None):
    #     assert self.body is not None
    #     if width is None:
    #         width = self.max_opening_width
    #     self.body.set_joint_angle(0, width / 2)
    #     self.body.set_joint_angle(1, width / 2)

    # def close(self):
    #     assert self.body is not None
    #     self.body.set_joint_angle(0, 0)
    #     self.body.set_joint_angle(1, 0)