from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import open3d as o3d
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics
from elias.config import Config
from elias.util import load_json
from tqdm.contrib.concurrent import thread_map

from nersemble_benchmark.constants import ASSETS, BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL
from nersemble_benchmark.util.video import VideoFrameLoader


@dataclass
class CameraParams(Config):
    world_2_cam: Dict[str, Pose]
    intrinsics: Dict[str, Intrinsics]


class BaseDataManager:
    def __init__(self, benchmark_folder: str, benchmark_type: str, participant_id: int):
        self._location = f"{benchmark_folder}/{benchmark_type}"
        self._benchmark_type = benchmark_type
        self._participant_id = participant_id

    # ----------------------------------------------------------
    # Assets
    # ----------------------------------------------------------

    def load_camera_calibration(self) -> CameraParams:
        camera_params = load_json(self.get_camera_calibration_path())
        world_2_cam = camera_params['world_2_cam']
        world_2_cam = {serial: Pose(pose, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
                       for serial, pose in world_2_cam.items()}
        intrinsics = {serial: Intrinsics(intr) for serial, intr in camera_params['intrinsics'].items()}
        camera_params = CameraParams(world_2_cam, intrinsics)
        return camera_params

    def list_timesteps(self, sequence_name: str) -> List[int]:
        return list(range(self.get_n_timesteps(sequence_name)))

    def list_serials(self, sequence_name: str) -> List[str]:
        images_folder = Path(self.get_images_path(sequence_name, "serial")).parent
        serials = [file.stem.split('_')[1] for file in images_folder.iterdir()]
        return serials

    def get_n_timesteps(self, sequence_name: str) -> int:
        serial = self.list_serials(sequence_name)[0]
        video_capture = VideoFrameLoader(self.get_images_path(sequence_name, serial))
        n_frames = video_capture.get_n_frames()
        return n_frames

    def load_alpha_map(self, sequence_name: str, serial: str, timestep: int) -> np.ndarray:
        video_capture = VideoFrameLoader(self.get_alpha_maps_path(sequence_name, serial))
        image = video_capture.load_frame(timestep)[..., [0]]
        return image

    def load_image(self, sequence_name: str, serial: str, timestep: int, apply_alpha_map: bool = False, as_uint8: bool = False) -> np.ndarray:
        video_path = self.get_images_path(sequence_name, serial)
        assert Path(video_path).exists(), f"Could not find video {video_path}"
        video_capture = VideoFrameLoader(video_path)
        image = video_capture.load_frame(timestep)

        if apply_alpha_map:
            alpha_map = self.load_alpha_map(sequence_name, serial, timestep)
            image = image / 255.
            alpha_map = alpha_map / 255.
            image = alpha_map * image + (1 - alpha_map)
            if as_uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
        elif not as_uint8:
            image = image / 255.

        return image

    def load_all_images(self, sequence_name: str, serial: str, apply_alpha_map: bool = False, as_uint8: bool = False) -> List[np.ndarray]:
        video_path = self.get_images_path(sequence_name, serial)
        assert Path(video_path).exists(), f"Could not find video {video_path}"
        video_capture = VideoFrameLoader(video_path)

        images = video_capture.load_all_frames()

        if apply_alpha_map:
            alpha_map_path = self.get_alpha_maps_path(sequence_name, serial)
            alpha_video_capture = VideoFrameLoader(alpha_map_path)
            alpha_maps = alpha_video_capture.load_all_frames()

            def process_image(image_and_alpha_map):
                image, alpha_map = image_and_alpha_map

                a = (np.multiply(alpha_map.astype(np.float32), 1.0 / 255))
                image = cv2.convertScaleAbs(image * a + (255 - alpha_map))

                # image = image / 255.
                # alpha_map = alpha_map / 255.
                # image = alpha_map * image + (1 - alpha_map)
                # if as_uint8:
                #     image = np.clip(image * 255, 0, 255).astype(np.uint8)

                return image

            processed_images = thread_map(process_image, zip(images, alpha_maps))
            images = processed_images
            # for image, alpha_map in zip(images, alpha_maps):
            #     image = image / 255.
            #     alpha_map = alpha_map / 255.
            #     image = alpha_map * image + (1 - alpha_map)
            #     if as_uint8:
            #         image = np.clip(image * 255, 0, 255).astype(np.uint8)
            #
            #     processed_images.append(image)

        elif not as_uint8:
            images = [image / 255. for image in images]

        return images

    def has_sequence(self, sequence_name: str) -> bool:
        video_path = self.get_images_path(sequence_name, BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL)
        return Path(video_path).exists()

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_camera_calibration_path(self) -> str:
        relative_path = ASSETS[self._benchmark_type]['per_person']['calibration'].format(p_id=self._participant_id)
        return f"{self._location}/{relative_path}"

    def get_images_path(self, sequence_name: str, serial: str) -> str:
        relative_path = ASSETS[self._benchmark_type]['per_cam']['images'].format(p_id=self._participant_id, seq_name=sequence_name, serial=serial)
        return f"{self._location}/{relative_path}"

    def get_alpha_maps_path(self, sequence_name: str, serial: str) -> str:
        relative_path = ASSETS[self._benchmark_type]['per_cam']['alpha_maps'].format(p_id=self._participant_id, seq_name=sequence_name, serial=serial)
        return f"{self._location}/{relative_path}"


class NVSDataManager(BaseDataManager):
    def __init__(self, benchmark_folder: str, participant_id: int):
        super().__init__(benchmark_folder, "nvs", participant_id)

    # ----------------------------------------------------------
    # Assets
    # ----------------------------------------------------------

    def load_pointcloud(self, sequence_name: str, timestep: int):
        pcd = o3d.io.read_point_cloud(self.get_pointcloud_path(sequence_name, timestep))
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        return points, colors, normals

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_pointcloud_path(self, sequence_name: str, timestep: int) -> str:
        relative_path = ASSETS[self._benchmark_type]['per_timestep']['pointclouds'].format(p_id=self._participant_id, seq_name=sequence_name, timestep=timestep)
        return f"{self._location}/{relative_path}"


@dataclass
class FlameTracking:
    # @formatter:off
    shape: np.ndarray               # (1, 300)
    expression: np.ndarray          # (T, 100)
    rotation: np.ndarray            # (T, 3)
    rotation_matrices: np.ndarray   # (T, 3, 3)
    translation: np.ndarray         # (T, 3)
    jaw: np.ndarray                 # (T, 3)
    frames: np.ndarray              # (T,)
    scale: np.ndarray               # (1, 1)
    neck: np.ndarray                # (T, 3)
    eyes: np.ndarray                # (T, 6)
    # @formatter:on


class MonoFlameAvatarDataManager(BaseDataManager):
    def __init__(self, benchmark_folder: str, participant_id: int):
        super().__init__(benchmark_folder, "mono_flame_avatar", participant_id)

    def load_flame_tracking(self, sequence_name: str) -> FlameTracking:
        flame_tracking = np.load(self.get_flame_tracking_path(sequence_name))
        flame_tracking = FlameTracking(**flame_tracking)
        return flame_tracking

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_flame_tracking_path(self, sequence_name: str) -> str:
        relative_path = ASSETS[self._benchmark_type]['per_sequence']['flame2023_tracking'].format(p_id=self._participant_id, seq_name=sequence_name)
        return f"{self._location}/{relative_path}"
