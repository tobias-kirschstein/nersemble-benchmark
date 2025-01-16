from unittest import TestCase

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_TRAIN_SERIALS
from nersemble_benchmark.data.benchmark_data import NVSDataManager

TEST_BENCHMARK_FOLDER = "D:/Projects/3D_Face_Scanning_Rig/data/benchmark_data"

class TestNvsDataManager(TestCase):

    def test_nvs_data_manager(self):
        test_participant_id, test_sequence_name = BENCHMARK_NVS_IDS_AND_SEQUENCES[0]
        test_serial = BENCHMARK_NVS_TRAIN_SERIALS[0]
        data_manager = NVSDataManager(TEST_BENCHMARK_FOLDER, test_participant_id)
        timesteps = data_manager.list_timesteps(test_sequence_name)
        self.assertEquals(len(timesteps), 1118)

        camera_calibration = data_manager.load_camera_calibration()
        self.assertEquals(len(camera_calibration.world_2_cam), 16)

        image = data_manager.load_image(test_sequence_name, test_serial, timesteps[-1])
        self.assertEquals(image.shape, (1604, 1100, 3))

        alpha_map = data_manager.load_alpha_map(test_sequence_name, test_serial, timesteps[-1])
        self.assertEquals(alpha_map.shape, (1604, 1100, 1))