from unittest import TestCase

from elias.util import save_img
from elias.util.io import resize_img

from nersemble_benchmark.env import REPO_ROOT
from nersemble_benchmark.data.benchmark_data import NVSDataManager
from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_TRAIN_SERIALS

TEST_BENCHMARK_FOLDER = "D:/Projects/3D_Face_Scanning_Rig/data/benchmark_data"

class ReadmeTest(TestCase):

    def test_load_image(self):
        benchmark_folder = TEST_BENCHMARK_FOLDER
        participant_id, sequence_name = BENCHMARK_NVS_IDS_AND_SEQUENCES[0]  # <- Use first benchmark subject
        serial = BENCHMARK_NVS_TRAIN_SERIALS[0]  # <- Use first train camera

        data_manager = NVSDataManager(benchmark_folder, participant_id)
        image = data_manager.load_image(sequence_name, serial, 0, apply_alpha_map=True, as_uint8=True)
        image = resize_img(image, 0.5)
        save_img(image, f"{REPO_ROOT}/static/images/example_image.jpg", quality=95)

    def test_load_alpha_map(self):
        benchmark_folder = TEST_BENCHMARK_FOLDER
        participant_id, sequence_name = BENCHMARK_NVS_IDS_AND_SEQUENCES[0]  # <- Use first benchmark subject
        serial = BENCHMARK_NVS_TRAIN_SERIALS[0]  # <- Use first train camera

        data_manager = NVSDataManager(benchmark_folder, participant_id)
        alpha_map = data_manager.load_alpha_map(sequence_name, serial, 0)
        alpha_map = resize_img(alpha_map[..., 0], 0.5)
        save_img(alpha_map, f"{REPO_ROOT}/static/images/example_alpha_map.jpg", quality=95)
