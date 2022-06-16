import numpy as np
import open3d as o3d
import pybullet as p
import collections
from regrasp.utils.world import BulletWorld
from regrasp.utils.transform import Rotation, Transform
from regrasp.utils.scene_maker import BulletSceneMaker
from regrasp.utils.camera import Camera, CameraIntrinsic
from regrasp.perception import Perception, camera_on_sphere
from regrasp.vgn.detection import VGN
from regrasp.utils.gripper import Gripper

State = collections.namedtuple("State", ["tsdf", "pc"])

def set_world(gui):
    world = BulletWorld(gui=gui)
    distance_between_robot = 0.6
    table_height = 0.2

    panda1 = world.load_urdf(
        name="panda1", 
        urdf_path="data/urdfs/panda/franka_panda.urdf",
        pose=Transform(Rotation.identity(), [0,distance_between_robot/2,0])
    )
    panda2 = world.load_urdf(
        name="panda2", 
        urdf_path="data/urdfs/panda/franka_panda.urdf",
        pose=Transform(Rotation.identity(), [0,-distance_between_robot/2,0])
    )
    scene_maker = BulletSceneMaker(world)
    scene_maker.create_plane(z_offset=-0.4)
    scene_maker.create_table(2, 2, 0.4)
    scene_maker.create_table(0.5, 0.5, 0.2, x_offset=0.4, y_offset=distance_between_robot/2, z_offset=table_height)
    scene_maker.create_table(0.5, 0.5, 0.2, x_offset=0.4, y_offset=-distance_between_robot/2, z_offset=table_height)
    box_pose_init = Transform(Rotation.identity(), [0.3, -0.3, table_height+0.05])
    box = world.load_urdf("box", "data/urdfs/blocks/cube.urdf", box_pose_init, scale=1.1)


    
    world.set_gravity([0,0,-9.81])
    objects = dict(
        panda1=panda1,
        panda2=panda2,
        box=box
    )
    return world, objects

def watch_workspace(world: BulletWorld):
    world.physics_client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=0.0,
        cameraPitch=-45,
        cameraTargetPosition=[0.15, 0.50, -0.3],
    )

world, objects = set_world(gui=False)

sim = BulletWorld(gui=True)
watch_workspace(sim)
sm = BulletSceneMaker(sim)
size = 0.3
mid = [size/2, size/2, size/2]
box_pose_init = Transform(Rotation.identity(), mid)
box = sim.load_urdf("box", "data/urdfs/blocks/cube.urdf", box_pose_init, scale=1.67)
sim.draw_workspace(size, mid)
cam = sim.add_camera(
    intrinsic=CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0),
    near=0.1,
    far=2.0
)
hand = Gripper(sim)
perception = Perception(cam)
tsdf, pc = perception.get_tsdf(size, mid, resolution=40)

#o3d.visualization.draw_geometries([pc])
state = State(tsdf, pc)
grasp_planner = VGN("data/models/vgn_conv_2.pth")
grasps, scores, quality = grasp_planner(state)

for i, grasp in enumerate(grasps):
    sm.view_point(f"{i}", grasp.pose.translation, size=0.005)
    
# xx, yy, zz = quality.shape
# voxel_size = tsdf.voxel_size
# for x in range(xx):
#     for y in range(yy):
#         for z in range(zz):
#             if quality[x, y, z] > 0.9:
#                 name = str(f"{x}{y}{z}")
#                 r = quality[x, y, z]/1*0.01
#                 pos = np.asarray([x, y, z])*voxel_size
#                 sm.view_point(name, pos, size=r)

#qual
#size
#for quality
#pose = grasp.pose
# pose.translation *= voxel_size
# width = grasp.width * voxel_size
# points = np.asarray(tsdf.get_cloud().points)
# for i, point in enumerate(points):
#     sm.view_point(f"{i}", point, size=0.005)

print(len(grasps))
for i, grasp in enumerate(grasps):
    sm.view_point("1", grasp.pose.translation, size=0.005)
    print(f"score:{scores[i]}")    
    T_tcp = grasp.pose
    hand_name = "hand"
    hand.reset(T_tcp, hand_name)
    hand.grip()
    sim.remove_body("hand")
input()

# for i, extrinsic in enumerate(extrinsics):
#     sm.view_frame(f"{i}", extrinsic.inverse())
#o3d.visualization.draw_geometries([pc])
# theta = np.pi/6
# phi = np.pi/2
# radius = 0.3
# eye = np.r_[
#     radius * np.sin(theta) * np.cos(phi),
#     radius * np.sin(theta) * np.sin(phi),
#     radius * np.cos(theta),
# ]
# viewmatrix = p.computeViewMatrix(eye, [0,0,0], [0,0,1])

# vm = np.array(viewmatrix).reshape(4,4, order="F")
# vm2 = Transform.look_at(eye, [0,0,0], [0,0,1])

# cam.physics_client.getCameraImage(
#     width=cam.intrinsic.width,
#     height=cam.intrinsic.height,
#     viewMatrix=viewmatrix,
#     projectionMatrix=cam.proj_matrix.flatten(order="F"),
#     renderer=p.ER_TINY_RENDERER
# )

# print(vm)
# print(vm2.as_matrix())
# sm.view_position("1", eye)
# target = np.array([0.0, 0.0, 0.0])
# up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
#T = Transform.look_at(eye, [0,0,0], [0,0,1])
#viewmatrix = p.computeViewMatrix(eye, [0,0,0], [0,0,1])
#T = Transform.from_matrix(np.asarray(viewmatrix).reshape(4,4,order="F"))
#sm.view_frame("2", T.inverse())
#sm.view_frame("2", T.inverse())
#return Transform.look_at(eye, target, up) * origin.inverse()
input()