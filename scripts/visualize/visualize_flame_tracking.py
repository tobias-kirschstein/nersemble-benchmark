import numpy as np
import tyro
from dreifus.pyvista import add_camera_frustum, add_coordinate_axes
from dreifus.render import project, draw_onto_image
import pyvista as pv

from nersemble_benchmark.constants import BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL, BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS
from nersemble_benchmark.data.benchmark_data import MonoFlameAvatarDataManager
from nersemble_benchmark.models.flame import FlameProvider


def main(benchmark_folder: str,
         /,
         participant_id: int = 461,
         sequence_name: str = "EXP-1-head",
         timestep: int = 90
         ):
    data_manager = MonoFlameAvatarDataManager(benchmark_folder, participant_id)
    camera_calibration = data_manager.load_camera_calibration()

    # Load FLAME tracking and create FLAME mesh
    flame_tracking = data_manager.load_flame_tracking(sequence_name)
    flame_provider = FlameProvider(flame_tracking)
    mesh = flame_provider.get_mesh(timestep)

    # Create visualizer with FLAME mesh
    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)
    p.add_mesh(mesh)

    serials = [BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL] + BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS
    for serial in serials:
        pose = camera_calibration.world_2_cam[serial]
        intrinsics = camera_calibration.intrinsics[serial]
        if data_manager.has_video(sequence_name, serial):
            # If it is a train sequence, we can also show the corresponding image
            image = data_manager.load_image(sequence_name, serial, timestep, as_uint8=True)
        else:
            # Otherwise, just show a black image (unknown)
            image = np.zeros((512, 512, 3))

        # Project the FLAME vertices onto the camera to ensure that the positioning of the FLAME mesh is correct
        projected_vertices = project(mesh.vertices, pose, intrinsics)
        draw_onto_image(image, projected_vertices, (0, 255, 0))

        # Visualize camera with projected FLAME vertices + potential GT image
        add_camera_frustum(p, pose, intrinsics, image=image, color='red' if serial in BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS else 'lightgray', label=serial)

    p.show()


if __name__ == '__main__':
    tyro.cli(main)
