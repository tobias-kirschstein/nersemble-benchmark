from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics
from elias.config import Config
from elias.util import load_json

from nersemble_benchmark.constants import ASSETS, BENCHMARK_NVS_TRAIN_SERIALS
from nersemble_benchmark.util.video import VideoFrameLoader


@dataclass
class CameraParams(Config):
    world_2_cam: Dict[str, Pose]
    intrinsics: Intrinsics


class NVSDataManager:
    def __init__(self, benchmark_folder: str, participant_id: int):
        self._location = f"{benchmark_folder}/nvs"
        self._participant_id = participant_id

    # ----------------------------------------------------------
    # Assets
    # ----------------------------------------------------------

    def load_camera_calibration(self) -> CameraParams:
        camera_params = load_json(self.get_camera_calibration_path())
        world_2_cam = camera_params['world_2_cam']
        world_2_cam = {serial: Pose(pose, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
                       for serial, pose in world_2_cam.items()}
        intrinsics = Intrinsics(camera_params['intrinsics'])
        camera_params = CameraParams(world_2_cam, intrinsics)
        return camera_params

    def list_timesteps(self, sequence_name: str) -> List[int]:
        return list(range(self.get_n_timesteps(sequence_name)))

    def get_n_timesteps(self, sequence_name: str) -> int:
        video_capture = VideoFrameLoader(self.get_images_path(sequence_name, BENCHMARK_NVS_TRAIN_SERIALS[0]))
        n_frames = video_capture.get_n_frames()
        return n_frames

    def load_alpha_map(self, sequence_name: str, serial: str, timestep: int) -> np.ndarray:
        video_capture = VideoFrameLoader(self.get_alpha_maps_path(sequence_name, serial))
        image = video_capture.load_frame(timestep)[..., [0]]
        return image

    def load_image(self, sequence_name: str, serial: str, timestep: int, apply_alpha_map: bool = False, as_uint8: bool = False) -> np.ndarray:
        video_capture = VideoFrameLoader(self.get_images_path(sequence_name, serial))
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

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_camera_calibration_path(self) -> str:
        relative_path = ASSETS['nvs']['global']['calibration'].format(p_id=self._participant_id)
        return f"{self._location}/{relative_path}"

    def get_images_path(self, sequence_name: str, serial: str) -> str:
        relative_path = ASSETS['nvs']['per_cam']['images'].format(p_id=self._participant_id, seq_name=sequence_name, serial=serial)
        return f"{self._location}/{relative_path}"

    def get_alpha_maps_path(self, sequence_name: str, serial: str) -> str:
        relative_path = ASSETS['nvs']['per_cam']['alpha_maps'].format(p_id=self._participant_id, seq_name=sequence_name, serial=serial)
        return f"{self._location}/{relative_path}"
