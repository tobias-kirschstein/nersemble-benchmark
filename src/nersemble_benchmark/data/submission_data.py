import re
import zipfile
from abc import abstractmethod
from collections import defaultdict
from io import BytesIO
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Optional, Tuple

import imageio
import mediapy
import numpy as np
import trimesh
from elias.util import ensure_directory_exists_for_file
import imageio.v3 as iio
from elias.util.io import resize_img
from trimesh.exchange.ply import load_ply

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_HOLD_OUT_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_IDS, \
    BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_SERIALS, BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL, \
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST, BENCHMARK_SVFR_IMAGE_KEYS
from nersemble_benchmark.data.benchmark_data import NVSDataManager
from nersemble_benchmark.util.metadata import NVSMetadata, MonoFLAMEAvatarMetadata
from nersemble_benchmark.util.video import VideoFrameLoader


# ==========================================================
# Base Data Reader/Writer
# ==========================================================

class SubmissionDataWriter:
    def __init__(self, zip_path: str):
        self._zip_path = zip_path
        ensure_directory_exists_for_file(self._zip_path)
        self._zipf = zipfile.ZipFile(self._zip_path, 'w')

    def __enter__(self):
        self._zipf.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._zipf.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self._zipf.close()


class VideoSubmissionDataWriter(SubmissionDataWriter):

    def __init__(self, zip_path: str, fps: float, width: int, height: int):
        super().__init__(zip_path)
        self._fps = fps
        self._width = width
        self._height = height

    def add_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        assert frames[0].shape[2] == 3, "All frames should have 3 channels"
        assert frames[0].dtype == np.uint8, "Frames should be given as np.uint8 dtype with color values in range 0-255"
        assert frames[0].shape[0] == self._height, f"All frames should have height {self._height}px"
        assert frames[0].shape[1] == self._width, f"All frames should have width {self._width}px"
        self._validate_video(participant_id, sequence_name, serial, frames)

        arcname = f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4"

        with TemporaryDirectory() as temp_dir:
            temp_video_path = f"{temp_dir}/video.mp4"
            mediapy.write_video(temp_video_path, frames, crf=14, fps=self._fps)
            self._zipf.write(temp_video_path, arcname)

    @abstractmethod
    def _validate_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        pass


class SubmissionDataReader:
    def __init__(self, zip_path: str):
        self._zip_path = zip_path
        self._zipf = zipfile.ZipFile(self._zip_path, 'r')

    @abstractmethod
    def validate_submission(self) -> Dict[str, List]:
        pass


class VideoSubmissionDataReader(SubmissionDataReader):
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

    def load_video(self,
                   participant_id: int,
                   sequence_name: str,
                   serial: str,
                   every_nth_frame: Optional[int] = None,
                   timestep: Optional[int] = None,
                   scale: Optional[float] = None) -> List[np.ndarray]:
        import imageio.v3 as iio
        video_path = self.get_video_path(participant_id, sequence_name, serial)
        with self._zipf.open(video_path) as f:
            data = BytesIO(f.read())
            if every_nth_frame is not None:
                image_props = iio.improps(f)
                frames = iio.imiter(data.getvalue(), plugin='pyav')
                frames = list(islice(frames, 0, image_props.n_images, every_nth_frame))
            else:
                frames = iio.imread(data.getvalue(), plugin='pyav', index=timestep)
                if timestep is not None:
                    frames = [frames]

        if scale is not None:
            frames = [resize_img(frame, scale) for frame in frames]

        return frames

    def get_video_path(self, participant_id: int, sequence_name: str, serial: str) -> str:
        return f"{participant_id:03d}/{sequence_name}/cam_{serial}.mp4"

    @abstractmethod
    def list_expected_files(self) -> List[str]:
        pass

    @abstractmethod
    def list_expected_video_lengths(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_expected_resolution(self) -> Tuple[int, int]:
        pass

    def validate_submission(self) -> Dict[str, List]:
        expected_files = self.list_expected_files()
        expected_lengths = self.list_expected_video_lengths()
        expected_width, expected_height = self.get_expected_resolution()
        actual_files = [file.filename for file in self._zipf.filelist]
        missing_files = []
        wrong_frame_counts = []
        wrong_resolutions = []
        for expected_file in expected_files:
            if expected_file not in actual_files:
                missing_files.append(expected_file)
            else:
                with self._zipf.open(expected_file) as f:
                    image_props = iio.improps(f)
                    n_frames_actual = image_props.n_images
                    actual_height = image_props.shape[1]
                    actual_width = image_props.shape[2]
                    n_frames_expected = expected_lengths[expected_file]
                    if n_frames_actual != n_frames_expected:
                        wrong_frame_counts.append((expected_file, n_frames_actual, n_frames_expected))

                    if actual_height != expected_height or actual_width != expected_width:
                        wrong_resolutions.append((expected_file, (actual_width, actual_height), (expected_width, expected_height)))

        submission_issues = dict()
        if missing_files:
            submission_issues['missing_files'] = missing_files

        if wrong_frame_counts:
            submission_issues['wrong_frame_counts'] = wrong_frame_counts

        if wrong_resolutions:
            submission_issues['wrong_resolutions'] = wrong_resolutions

        return submission_issues


# ==========================================================
# Dynamic Novel View Synthesis Task
# ==========================================================


class NVSSubmissionDataWriter(VideoSubmissionDataWriter):

    def __init__(self, zip_path: str):
        super().__init__(zip_path, 73, 1100, 1604)

    def _validate_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        nvs_ids_and_sequences = dict(BENCHMARK_NVS_IDS_AND_SEQUENCES)
        assert participant_id in nvs_ids_and_sequences, f"Invalid participant_id {participant_id}, should be one of {list(nvs_ids_and_sequences.keys())}"
        assert sequence_name == nvs_ids_and_sequences[participant_id], f"Invalid sequence name {sequence_name} expected {nvs_ids_and_sequences[participant_id]}"
        assert serial in BENCHMARK_NVS_HOLD_OUT_SERIALS, f"Invalid serial. Only the hold-out serials {BENCHMARK_NVS_HOLD_OUT_SERIALS} should be submitted"


class NVSSubmissionDataReader(VideoSubmissionDataReader):

    def is_complete(self, participant_id: int, sequence_name: str) -> bool:
        file_overview = self.get_file_overview()
        if participant_id not in file_overview:
            return False
        if sequence_name not in file_overview[participant_id]:
            return False
        complete = all([serial in file_overview[participant_id][sequence_name] for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS])
        return complete

    def list_expected_files(self) -> List[str]:
        expected_files = []
        for participant_id, sequence_name in BENCHMARK_NVS_IDS_AND_SEQUENCES:
            for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS:
                expected_files.append(self.get_video_path(participant_id, sequence_name, serial))

        return expected_files

    def list_expected_video_lengths(self) -> Dict[str, int]:
        expected_video_lengths = dict()
        nvs_metadata = NVSMetadata.load()
        for participant_id, sequence_name in BENCHMARK_NVS_IDS_AND_SEQUENCES:
            expected_length = len(nvs_metadata.sequences[participant_id].timesteps)
            for serial in BENCHMARK_NVS_HOLD_OUT_SERIALS:
                video_path = self.get_video_path(participant_id, sequence_name, serial)
                expected_video_lengths[video_path] = expected_length

        return expected_video_lengths

    def get_expected_resolution(self) -> Tuple[int, int]:
        return 1100, 1604


# ==========================================================
# Monocular 3D FLAME Avatar Reconstruction Task
# ==========================================================


class MonoFlameAvatarSubmissionDataWriter(VideoSubmissionDataWriter):

    def __init__(self, zip_path: str):
        super().__init__(zip_path, 24.3, 512, 512)

    def _validate_video(self, participant_id: int, sequence_name: str, serial: str, frames: List[np.ndarray]):
        valid_participant_ids = BENCHMARK_MONO_FLAME_AVATAR_IDS
        assert participant_id in valid_participant_ids, f"Invalid participant_id {participant_id}"
        assert sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST, \
            f"Invalid sequence name {sequence_name}. Only the hold-out sequences {BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST} should be submitted"
        assert serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS, (f"Invalid serial {serial}. Only the train serial {BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL} "
                                                               f"and the hold-out serials {BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS} should be submitted")


class MonoFlameAvatarSubmissionDataReader(VideoSubmissionDataReader):

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

    def list_expected_files(self) -> List[str]:
        expected_files = []
        for participant_id in BENCHMARK_MONO_FLAME_AVATAR_IDS:
            for sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST:
                for serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS:
                    expected_files.append(self.get_video_path(participant_id, sequence_name, serial))

        return expected_files

    def list_expected_video_lengths(self) -> Dict[str, int]:
        expected_video_lengths = dict()
        mono_avatar_metadata = MonoFLAMEAvatarMetadata.load()
        for participant_id in BENCHMARK_MONO_FLAME_AVATAR_IDS:
            for sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST:
                for serial in BENCHMARK_MONO_FLAME_AVATAR_SERIALS:
                    expected_length = mono_avatar_metadata.participants_metadata[participant_id].sequences_metadata[sequence_name].n_frames
                    video_path = self.get_video_path(participant_id, sequence_name, serial)
                    expected_video_lengths[video_path] = expected_length

        return expected_video_lengths

    def get_expected_resolution(self) -> Tuple[int, int]:
        return 512, 512


# ==========================================================
# Single-view 3D Face Reconstruction Task
# ==========================================================


class SVFRSubmissionDataWriter(SubmissionDataWriter):
    def _add_mesh(self,
                  participant_id: int,
                  sequence_name: str,
                  timestep: int,
                  serial: str,
                  mesh: trimesh.Trimesh,
                  svfr_task: str,
                  now_landmarks: Optional[np.ndarray] = None):
        assert len(mesh.vertices) > 0, f"Mesh for person {participant_id}, {sequence_name}_{timestep}_{serial} has no vertices"
        assert len(mesh.faces) > 0, f"Mesh for person {participant_id}, {sequence_name}_{timestep}_{serial} has no faces"
        assert not np.isnan(mesh.vertices).any(), f"Mesh for person {participant_id}, {sequence_name}_{timestep}_{serial} contains NaN coordinates"
        assert not np.isnan(mesh.faces).any()
        assert len(
            mesh.vertices) == 5023 or now_landmarks is not None, "If mesh has a different topology than FLAME, 7 landmarks following the NoW convention have to be provided for alignment to GT mesh. See: https://github.com/soubhiksanyal/now_evaluation/blob/main/landmarks_7_annotated.png"
        assert now_landmarks is None or now_landmarks.shape == (7, 3), f"NoW landmarks expected shape: 7x3. Got: {now_landmarks.shape}"

        mesh_path = f"{participant_id:03d}/{sequence_name}_{timestep:03d}_{serial}/mesh_{svfr_task}.ply"
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

        with self._zipf.open(mesh_path, 'w') as f:
            mesh.export(f, 'ply')

        if now_landmarks is not None:
            landmarks_path = f"{participant_id:03d}/{sequence_name}_{timestep:03d}_{serial}/landmarks_{svfr_task}.npy"
            with self._zipf.open(landmarks_path, 'w') as f:
                np.save(f, now_landmarks)

    def add_posed_mesh(self,
                       participant_id: int,
                       sequence_name: str,
                       timestep: int,
                       serial: str,
                       mesh: trimesh.Trimesh,
                       now_landmarks: Optional[np.ndarray] = None):
        self._add_mesh(participant_id, sequence_name, timestep, serial, mesh, 'posed', now_landmarks=now_landmarks)

    def add_neutral_mesh(self,
                         participant_id: int,
                         sequence_name: str,
                         timestep: int,
                         serial: str,
                         mesh: trimesh.Trimesh,
                         now_landmarks: Optional[np.ndarray] = None):
        self._add_mesh(participant_id, sequence_name, timestep, serial, mesh, 'neutral', now_landmarks=now_landmarks)


class SVFRSubmissionDataReader(SubmissionDataReader):
    def load_posed_mesh(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> trimesh.Trimesh:
        return self._load_mesh(participant_id, sequence_name, timestep, serial, 'posed')

    def load_neutral_mesh(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> trimesh.Trimesh:
        return self._load_mesh(participant_id, sequence_name, timestep, serial, 'neutral')

    def load_posed_landmarks(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        return self._load_landmarks(participant_id, sequence_name, timestep, serial, 'posed')

    def load_neutral_landmarks(self, participant_id: int, sequence_name: str, timestep: int, serial: str) -> np.ndarray:
        return self._load_landmarks(participant_id, sequence_name, timestep, serial, 'neutral')

    def has_posed_reconstructions(self) -> bool:
        return self._has_reconstructions('posed')

    def has_neutral_reconstructions(self) -> bool:
        return self._has_reconstructions('neutral')

    def validate_submission(self) -> Dict[str, List]:
        actual_files = [file.filename for file in self._zipf.filelist if not file.is_dir()]

        has_posed = False
        has_neutral = False

        missing_posed_meshes = []
        missing_neutral_meshes = []
        empty_posed_meshes = []
        empty_neutral_meshes = []
        missing_posed_landmarks = []
        missing_neutral_landmarks = []
        wrong_posed_landmarks = []
        wrong_neutral_landmarks = []
        all_expected_posed_files = []
        all_expected_neutral_files = []
        unexpected_files = []

        for participant_id, person_keys in BENCHMARK_SVFR_IMAGE_KEYS.items():
            for sequence_name, timestep, serial in person_keys:
                expected_posed_mesh_path = self._get_mesh_path(participant_id, sequence_name, timestep, serial, 'posed')
                expected_posed_landmarks_path = self._get_landmarks_path(participant_id, sequence_name, timestep, serial, 'posed')
                all_expected_posed_files.append(expected_posed_mesh_path)
                all_expected_posed_files.append(expected_posed_landmarks_path)
                if expected_posed_mesh_path in actual_files:
                    has_posed = True

                    mesh = self.load_posed_mesh(participant_id, sequence_name, timestep, serial)
                    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        empty_posed_meshes.append(expected_posed_mesh_path)

                    if len(mesh.vertices) != 5023:
                        if expected_posed_landmarks_path in actual_files:
                            landmarks = self.load_posed_landmarks(participant_id, sequence_name, timestep, serial)
                            if landmarks.shape != (7, 3):
                                wrong_posed_landmarks.append((expected_posed_landmarks_path, landmarks.shape))
                        else:
                            missing_posed_landmarks.append(expected_posed_landmarks_path)
                else:
                    missing_posed_meshes.append(expected_posed_mesh_path)

                expected_neutral_mesh_path = self._get_mesh_path(participant_id, sequence_name, timestep, serial, 'neutral')
                expected_neutral_landmarks_path = self._get_landmarks_path(participant_id, sequence_name, timestep, serial, 'neutral')
                all_expected_neutral_files.append(expected_neutral_mesh_path)
                all_expected_neutral_files.append(expected_neutral_landmarks_path)
                if expected_neutral_mesh_path in actual_files:
                    has_neutral = True

                    mesh = self.load_neutral_mesh(participant_id, sequence_name, timestep, serial)
                    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        empty_neutral_meshes.append(expected_neutral_mesh_path)

                    if len(mesh.vertices) != 5023:
                        if expected_neutral_landmarks_path in actual_files:
                            landmarks = self.load_neutral_landmarks(participant_id, sequence_name, timestep, serial)
                            if landmarks.shape != (7, 3):
                                wrong_neutral_landmarks.append((expected_neutral_landmarks_path, landmarks.shape))
                        else:
                            missing_neutral_landmarks.append(expected_neutral_landmarks_path)
                else:
                    missing_neutral_meshes.append(expected_neutral_mesh_path)

        for actual_file in actual_files:
            if not has_neutral and actual_file not in all_expected_posed_files:
                unexpected_files.append(actual_file)
            if not has_posed and actual_file not in all_expected_neutral_files:
                unexpected_files.append(actual_file)

        submission_issues = dict()
        if has_posed:
            if missing_posed_meshes:
                submission_issues['missing_posed_meshes'] = missing_posed_meshes

            if empty_posed_meshes:
                submission_issues['empty_posed_meshes'] = empty_posed_meshes

            if missing_posed_landmarks:
                submission_issues['missing_posed_landmarks'] = missing_posed_landmarks

            if wrong_posed_landmarks:
                submission_issues['wrong_posed_landmarks'] = wrong_posed_landmarks

        if has_neutral:
            if missing_neutral_meshes:
                submission_issues['missing_neutral_meshes'] = missing_neutral_meshes

            if empty_neutral_meshes:
                submission_issues['empty_neutral_meshes'] = empty_neutral_meshes

            if missing_neutral_landmarks:
                submission_issues['missing_neutral_landmarks'] = missing_neutral_landmarks

            if wrong_neutral_landmarks:
                submission_issues['wrong_neutral_landmarks'] = wrong_neutral_landmarks

        if unexpected_files:
            submission_issues['unexpected_files'] = unexpected_files

        if not has_posed and not has_neutral:
            submission_issues['missing_svfr_tasks'] = ['posed', 'neutral']

        return submission_issues

    def _load_mesh(self, participant_id: int, sequence_name: str, timestep: int, serial: str, svfr_task: str) -> trimesh.Trimesh:
        mesh_path = self._get_mesh_path(participant_id, sequence_name, timestep, serial, svfr_task)

        with self._zipf.open(mesh_path, 'r') as f:
            mesh = trimesh.Trimesh(**load_ply(f))

        return mesh

    def _load_landmarks(self, participant_id: int, sequence_name: str, timestep: int, serial: str, svfr_task: str) -> np.ndarray:
        landmarks_path = self._get_landmarks_path(participant_id, sequence_name, timestep, serial, svfr_task)

        with self._zipf.open(landmarks_path, 'r') as f:
            landmarks = np.load(f)

        return landmarks

    def _get_mesh_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str, svfr_task: str) -> str:
        return f"{participant_id:03d}/{sequence_name}_{timestep:03d}_{serial}/mesh_{svfr_task}.ply"

    def _get_landmarks_path(self, participant_id: int, sequence_name: str, timestep: int, serial: str, svfr_task: str) -> str:
        return f"{participant_id:03d}/{sequence_name}_{timestep:03d}_{serial}/landmarks_{svfr_task}.npy"

    def _has_reconstructions(self, svfr_task: str):
        contained_files = [file.filename for file in self._zipf.filelist]
        has_reconstructions = False
        for participant_id, person_keys in BENCHMARK_SVFR_IMAGE_KEYS.items():
            for sequence_name, timestep, serial in person_keys:
                expected_posed_mesh_path = self._get_mesh_path(participant_id, sequence_name, timestep, serial, svfr_task)
                if expected_posed_mesh_path in contained_files:
                    has_reconstructions = True
                    break

        return has_reconstructions
