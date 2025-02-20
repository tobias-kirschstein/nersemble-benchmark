from threading import Thread
from time import sleep
from unittest import TestCase

import numpy as np
from dreifus.pyvista import add_camera_frustum, add_coordinate_axes
from dreifus.render import project, draw_onto_image

from nersemble_benchmark.constants import BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL, BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS
from nersemble_benchmark.data.benchmark_data import MonoFlameAvatarDataManager
from nersemble_benchmark.models.flame import FlameProvider


class MonoAvatarTest(TestCase):

    def test_data_manager(self):
        participant_id = 461
        sequence_name = "FREE"
        timestep = 196
        serial = BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL
        # data_manager = MonoFlameAvatarDataManager("D:/Projects/3D_Face_Scanning_Rig/data/benchmark_data", participant_id)
        data_manager = MonoFlameAvatarDataManager("D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out", participant_id)
        flame_tracking = data_manager.load_flame_tracking(sequence_name)

        flame_provider = FlameProvider(flame_tracking)
        mesh = flame_provider.get_mesh(timestep)

        camera_calibration = data_manager.load_camera_calibration()
        pose = camera_calibration.world_2_cam[serial]
        intrinsics = camera_calibration.intrinsics[serial]
        if data_manager.has_sequence(sequence_name):
            image = data_manager.load_image(sequence_name, serial, timestep)
        else:
            image = np.zeros((512, 512, 3))
        projected_vertices = project(mesh.vertices, pose, intrinsics)
        draw_onto_image(image, projected_vertices, (0, 1, 0))

        poses_hold_out = [camera_calibration.world_2_cam[s] for s in BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS]
        intrinsics_hold_out = [camera_calibration.intrinsics[s] for s in BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS]

        import pyvista as pv
        mesh_container = pv.wrap(mesh)

        p = pv.Plotter()
        p.add_mesh(mesh_container)
        add_camera_frustum(p, pose, intrinsics, image=image)

        for serial_hold_out, pose_hold_out, intr_hold_out in zip(BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS, poses_hold_out, intrinsics_hold_out):
            image_hold_out = np.zeros_like(image)
            projected_hold_out_vertices = project(mesh.vertices, pose_hold_out, intr_hold_out)
            draw_onto_image(image_hold_out, projected_hold_out_vertices, (0, 1, 0))

            add_camera_frustum(p, pose_hold_out, intr_hold_out, image=image_hold_out, color='red', label=serial_hold_out)

        add_coordinate_axes(p, scale=0.1)
        p.show()