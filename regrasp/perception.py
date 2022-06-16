import numpy as np
import open3d as o3d
from typing import Union
from regrasp.utils import camera
from regrasp.utils.camera import Camera
from regrasp.utils.transform import Rotation, Transform
from regrasp.utils.scene_maker import BulletSceneMaker
import matplotlib.pyplot as plt

class Perception:
    def __init__(self, camera: Camera):
        self.camera = camera
    
    def get_tsdf(self, size: float, origin: Union[np.ndarray, list], resolution: int = 40):
        #r = 1.5* size
        tsdf = TSDFVolume(size, 40)
        high_res_tsdf = TSDFVolume(size, 120)
        origin = Transform(Rotation.identity(), origin)
        n = 8
        #width, height = self.camera.intrinsic.width, self.camera.intrinsic.height
        extrinsics = np.empty((n, 7), dtype=np.float32)
        #depth_imgs = np.empty((n, height, width), np.float32)
        
        phi_list = np.asarray([0, np.pi/2, np.pi, np.pi*3/2, 0, np.pi/2, np.pi, np.pi*3/2]) + np.pi/6
        theta_list = [np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi*2/3, np.pi*2/3, np.pi*2/3, np.pi*2/3]
        #extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi, theta in zip(phi_list, theta_list)]
        extrinsics = []
        for i in range(n):
            r = np.random.uniform(1.6, 2.4) * size
            extrinsic = camera_on_sphere(origin, r, theta_list[i], phi_list[i])
            depth_img = self.camera.render(extrinsic)[1]
            #plt.imshow(depth_img)
            # extrinsics[i] = extrinsic.to_list()
            # depth_imgs[i] = depth_img
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            #extrinsics.append(extrinsic)
        return tsdf, high_res_tsdf.get_cloud()


def camera_on_sphere(origin: Transform, radius: float, theta: float, phi: float):
    eye = np.r_[
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()

class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(cloud.points)
        distances = np.asarray(cloud.colors)[:, [0]]
        grid = np.zeros((1, 40, 40, 40), dtype=np.float32)
        for idx, point in enumerate(points):
            i, j, k = np.floor(point / self.voxel_size).astype(int)
            grid[0, i, j, k] = distances[idx]
        return grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf