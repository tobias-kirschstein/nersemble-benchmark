from unittest import TestCase

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_HOLD_OUT_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_IDS, \
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST, BENCHMARK_MONO_FLAME_AVATAR_SERIALS
from nersemble_benchmark.data.benchmark_data import NVSDataManager, MonoFlameAvatarDataManager
from nersemble_benchmark.data.submission_data import NVSSubmissionDataWriter, MonoFlameAvatarSubmissionDataWriter


class SubmissionDataTest(TestCase):

    def test_nvs_submission_data(self):
        zip_path = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_submissions/submission_nvs.zip"
        benchmark_folder = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out"

        submission_data_manager = NVSSubmissionDataWriter(zip_path)

        for participant, sequence_name in BENCHMARK_NVS_IDS_AND_SEQUENCES:
            data_manager = NVSDataManager(benchmark_folder, participant)
            timesteps = data_manager.list_timesteps(sequence_name)
            for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS:
                # def load_img(t):
                #     return data_manager.load_image(sequence_name, serial, t, apply_alpha_map=True, as_uint8=True)
                #
                # images = thread_map(load_img, timesteps)
                images = data_manager.load_all_images(sequence_name, serial, apply_alpha_map=True, as_uint8=True)

                submission_data_manager.add_video(participant, sequence_name, serial, images)

            break

    def test_mono_flame_avatar_submission_data(self):
        zip_path = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_submissions/submission_mono_flame_avatar.zip"
        benchmark_folder = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out"

        submission_data_manager = MonoFlameAvatarSubmissionDataWriter(zip_path)

        for participant in BENCHMARK_MONO_FLAME_AVATAR_IDS:
            data_manager = MonoFlameAvatarDataManager(benchmark_folder, participant)
            for sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST:
                for serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS:
                    images = data_manager.load_all_images(sequence_name, serial, apply_alpha_map=True, as_uint8=True)
                    submission_data_manager.add_video(participant, sequence_name, serial, images)

            break
