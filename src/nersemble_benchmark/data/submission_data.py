import re
import zipfile
from collections import defaultdict
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import List, Dict

import mediapy
import numpy as np
from elias.util import ensure_directory_exists_for_file

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_HOLD_OUT_SERIALS


class NVSSubmissionDataWriter:

    def __init__(self, zip_path: str):
        self._zip_path = zip_path
        self._zipf = zipfile.ZipFile(self._zip_path, 'w')

    def add_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        nvs_ids_and_sequences = dict(BENCHMARK_NVS_IDS_AND_SEQUENCES)
        assert participant_id in nvs_ids_and_sequences, f"Invalid participant_id {participant_id}"
        assert sequence_name == nvs_ids_and_sequences[participant_id], f"Invalid sequence name {sequence_name} expected {nvs_ids_and_sequences[participant_id]}"
        assert frames[0].shape[0] == 1604, "All frames should have height 1604px"
        assert frames[0].shape[1] == 1100, "All frames should have width 1100px"
        assert frames[0].shape[2] == 3, "All frames should have 3 channels"
        assert frames[0].dtype == np.uint8, "Frames should be given as np.uint8 dtype with color values in range 0-255"

        ensure_directory_exists_for_file(self._zip_path)

        with TemporaryDirectory() as temp_dir:
            temp_video_path = f"{temp_dir}/video.mp4"
            mediapy.write_video(temp_video_path, frames, crf=14, fps=73)
            self._zipf.write(temp_video_path, f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4")


class NVSSubmissionDataReader:

    def __init__(self, zip_path: str):
        self._zip_path = zip_path
        self._zipf = zipfile.ZipFile(self._zip_path, 'r')

    def get_file_overview(self) -> Dict[int, Dict[str, List[str]]]:
        file_overview = defaultdict(lambda: defaultdict(list))  # participant_id => sequence_name => [serial]
        pattern = re.compile("(\d+)/(\w+)/cam_(\w+)\.mp4")
        for filename in self._zipf.namelist():
            matches = pattern.match(filename)
            if matches:
                participant_id = int(matches[1])
                sequence_name = matches[2]
                serial = matches[3]
                file_overview[participant_id][sequence_name].append(serial)

        return file_overview

    def is_complete(self, participant_id: int, sequence_name: str) -> bool:
        file_overview = self.get_file_overview()
        if participant_id not in file_overview:
            return False
        if sequence_name not in file_overview[participant_id]:
            return False
        complete = all([serial in file_overview[participant_id][sequence_name] for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS])
        return complete

    def load_video(self, participant_id: int, sequence_name: str, serial: str) -> List[np.ndarray]:
        import imageio.v3 as iio
        with self._zipf.open(f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4") as f:
            data = BytesIO(f.read())
            frames = iio.imread(data.getvalue(), plugin='pyav')
        return frames
