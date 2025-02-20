import re
import zipfile
from abc import abstractmethod
from collections import defaultdict
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import List, Dict, Optional

import mediapy
import numpy as np
from elias.util import ensure_directory_exists_for_file

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_HOLD_OUT_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_IDS, \
    BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL, \
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST


class SubmissionDataWriter:
    def __init__(self, zip_path: str, fps: float, width: int, height: int):
        self._zip_path = zip_path
        ensure_directory_exists_for_file(self._zip_path)
        self._zipf = zipfile.ZipFile(self._zip_path, 'w')
        self._fps = fps
        self._width = width
        self._height = height

    def add_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        assert frames[0].shape[2] == 3, "All frames should have 3 channels"
        assert frames[0].dtype == np.uint8, "Frames should be given as np.uint8 dtype with color values in range 0-255"
        assert frames[0].shape[0] == self._height, "All frames should have height 1604px"
        assert frames[0].shape[1] == self._width, "All frames should have width 1100px"
        self._validate_video(participant_id, sequence_name, serial, frames)

        arcname = f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4"

        with TemporaryDirectory() as temp_dir:
            temp_video_path = f"{temp_dir}/video.mp4"
            mediapy.write_video(temp_video_path, frames, crf=14, fps=self._fps)
            self._zipf.write(temp_video_path, arcname)

    @abstractmethod
    def _validate_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        pass


class NVSSubmissionDataWriter(SubmissionDataWriter):

    def __init__(self, zip_path: str):
        super().__init__(zip_path, 73, 1100, 1604)

    def _validate_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        nvs_ids_and_sequences = dict(BENCHMARK_NVS_IDS_AND_SEQUENCES)
        assert participant_id in nvs_ids_and_sequences, f"Invalid participant_id {participant_id}, should be one of {list(nvs_ids_and_sequences.keys())}"
        assert sequence_name == nvs_ids_and_sequences[participant_id], f"Invalid sequence name {sequence_name} expected {nvs_ids_and_sequences[participant_id]}"
        assert serial in BENCHMARK_NVS_HOLD_OUT_SERIALS, f"Invalid serial. Only the hold-out serials {BENCHMARK_NVS_HOLD_OUT_SERIALS} should be submitted"

    # def add_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
    #     nvs_ids_and_sequences = dict(BENCHMARK_NVS_IDS_AND_SEQUENCES)
    #     assert participant_id in nvs_ids_and_sequences, f"Invalid participant_id {participant_id}"
    #     assert sequence_name == nvs_ids_and_sequences[participant_id], f"Invalid sequence name {sequence_name} expected {nvs_ids_and_sequences[participant_id]}"
    #     assert frames[0].shape[0] == 1604, "All frames should have height 1604px"
    #     assert frames[0].shape[1] == 1100, "All frames should have width 1100px"
    #     assert frames[0].shape[2] == 3, "All frames should have 3 channels"
    #     assert frames[0].dtype == np.uint8, "Frames should be given as np.uint8 dtype with color values in range 0-255"
    #
    #     arcname = f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4"
    #
    #     with TemporaryDirectory() as temp_dir:
    #         temp_video_path = f"{temp_dir}/video.mp4"
    #         mediapy.write_video(temp_video_path, frames, crf=14, fps=73)
    #         self._zipf.write(temp_video_path, arcname)


class MonoFlameAvatarSubmissionDataWriter(SubmissionDataWriter):

    def __init__(self, zip_path: str):
        super().__init__(zip_path, 24.3, 512, 512)

    def _validate_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        valid_participant_ids = BENCHMARK_MONO_FLAME_AVATAR_IDS
        assert participant_id in valid_participant_ids, f"Invalid participant_id {participant_id}"
        assert sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST, \
            f"Invalid sequence name {sequence_name}. Only the hold-out sequences {BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST} should be submitted"
        assert serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS, (f"Invalid serial {serial}. Only the train serial {BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL} "
                                                         f"and the hold-out serials {BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS} should be submitted")


class SubmissionDataReader:
    def __init__(self, zip_path: str):
        self._zip_path = zip_path
        self._zipf = zipfile.ZipFile(self._zip_path, 'r')

    def get_file_overview(self) -> Dict[int, Dict[str, List[str]]]:
        file_overview = defaultdict(lambda: defaultdict(list))  # participant_id => sequence_name => [serial]
        pattern = re.compile("(\d+)/([\w\d\-_+]+)/cam_(\w+)\.mp4")
        for filename in self._zipf.namelist():
            matches = pattern.match(filename)
            if matches:
                participant_id = int(matches[1])
                sequence_name = matches[2]
                serial = matches[3]
                file_overview[participant_id][sequence_name].append(serial)

        return file_overview

    def load_video(self, participant_id: int, sequence_name: str, serial: str) -> List[np.ndarray]:
        import imageio.v3 as iio
        with self._zipf.open(f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4") as f:
            data = BytesIO(f.read())
            frames = iio.imread(data.getvalue(), plugin='pyav')
        return frames


class NVSSubmissionDataReader(SubmissionDataReader):

    def is_complete(self, participant_id: int, sequence_name: str) -> bool:
        file_overview = self.get_file_overview()
        if participant_id not in file_overview:
            return False
        if sequence_name not in file_overview[participant_id]:
            return False
        complete = all([serial in file_overview[participant_id][sequence_name] for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS])
        return complete


class MonoFlameAvatarSubmissionDataReader(SubmissionDataReader):

    def is_complete(self, participant_id: Optional[int] = None) -> bool:
        if participant_id is None:
            participant_ids = BENCHMARK_MONO_FLAME_AVATAR_IDS
        else:
            participant_ids = [participant_id]

        file_overview = self.get_file_overview()

        for participant_id in participant_ids:
            if not participant_id in file_overview:
                return False

            for sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST:
                if not sequence_name in file_overview[participant_id]:
                    return False

                for serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS:
                    if not serial in file_overview[participant_id][sequence_name]:
                        return False

        return True
