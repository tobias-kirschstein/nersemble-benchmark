from unittest import TestCase

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_HOLD_OUT_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_IDS, \
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST, BENCHMARK_MONO_FLAME_AVATAR_SERIALS
from nersemble_benchmark.data.benchmark_data import NVSDataManager, MonoFlameAvatarDataManager
from nersemble_benchmark.data.submission_data import NVSSubmissionDataWriter, MonoFlameAvatarSubmissionDataWriter, NVSSubmissionDataReader, \
    MonoFlameAvatarSubmissionDataReader


class SubmissionDataTest(TestCase):

    def test_nvs_submission_data(self):
        zip_path = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_submissions/submission_nvs.zip"
        benchmark_folder = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out"

        with NVSSubmissionDataWriter(zip_path) as submission_data_manager:
            for participant, sequence_name in BENCHMARK_NVS_IDS_AND_SEQUENCES:
                data_manager = NVSDataManager(benchmark_folder, participant)
                for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS:
                    images = data_manager.load_all_images(sequence_name, serial, as_uint8=True)

                    submission_data_manager.add_video(participant, sequence_name, serial, images)

                break

    def test_validate_nvs_submission_data(self):
        zip_path = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_submissions/nersemble_deformable3dgs_incomplete.zip"
        benchmark_folder = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out"

        data_reader = NVSSubmissionDataReader(zip_path)
        data_reader.validate_submission()


    def test_mono_flame_avatar_submission_data(self):
        zip_path = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_submissions/submission_mono_flame_avatar.zip"
        benchmark_folder = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out"

        with MonoFlameAvatarSubmissionDataWriter(zip_path) as submission_data_manager:
            for participant in BENCHMARK_MONO_FLAME_AVATAR_IDS:
                data_manager = MonoFlameAvatarDataManager(benchmark_folder, participant)
                for sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST:
                    for serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS:
                        images = data_manager.load_all_images(sequence_name, serial, as_uint8=True)
                        submission_data_manager.add_video(participant, sequence_name, serial, images)

                break

    def test_validate_mono_avatar_submission_data(self):
        zip_path = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_submissions/submission_mono_flame_avatar.zip"
        benchmark_folder = "D:/Projects/3D_Face_Scanning_Rig/analyses/benchmark_v1_hold_out"

        data_reader = MonoFlameAvatarSubmissionDataReader(zip_path)
        result = data_reader.validate_submission()
        print('hi')
