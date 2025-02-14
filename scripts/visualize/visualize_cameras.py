from pathlib import Path

import tyro
from dreifus.pyvista import add_camera_frustum, add_coordinate_axes

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_TRAIN_SERIALS, BENCHMARK_NVS_HOLD_OUT_SERIALS
from nersemble_benchmark.data.benchmark_data import NVSDataManager
import pyvista as pv


def main(benchmark_folder: str, participant_id: int, /, timestep: int = 0):
    data_manager = NVSDataManager(benchmark_folder, participant_id)
    sequence_name = [seq_name for p_id, seq_name in BENCHMARK_NVS_IDS_AND_SEQUENCES if p_id == participant_id][0]
    images = {serial: data_manager.load_image(sequence_name, serial, timestep, apply_alpha_map=True) for serial in BENCHMARK_NVS_TRAIN_SERIALS}
    camera_params = data_manager.load_camera_calibration()

    has_pointcloud = Path(data_manager.get_pointcloud_path(sequence_name, timestep)).exists()

    # Visualize train cameras with corresponding image
    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)
    for serial in BENCHMARK_NVS_TRAIN_SERIALS:
        image = images[serial]
        world_2_cam_pose = camera_params.world_2_cam[serial]
        intr = camera_params.intrinsics[serial]

        add_camera_frustum(p, world_2_cam_pose, intr, image=image)

    # Visualize hold-out serials
    for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS:
        world_2_cam_pose = camera_params.world_2_cam[serial]
        intr = camera_params.intrinsics[serial]
        add_camera_frustum(p, world_2_cam_pose, intr, color='red')

    if has_pointcloud:
        points, colors, normals = data_manager.load_pointcloud(sequence_name, timestep)
        p.add_points(points, scalars=colors, rgb=True)

    p.show()


if __name__ == '__main__':
    tyro.cli(main)
