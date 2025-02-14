from multiprocessing.pool import ThreadPool
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal, Union, List, Tuple, Optional, Dict

import tyro
from elias.util import load_json
from torch._C import BenchmarkConfig
from tqdm import tqdm

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_TRAIN_SERIALS, ASSETS, BENCHMARK_MONO_FLAME_AVATAR_IDS, \
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TRAIN, BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST, BENCHMARK_MONO_AVATAR_TRAIN_SERIAL, \
    BENCHMARK_MONO_AVATAR_HOLD_OUT_SERIALS
from nersemble_benchmark.env import NERSEMBLE_BENCHMARK_URL_NVS, NERSEMBLE_BENCHMARK_URL
from nersemble_benchmark.util.download import download_file
from nersemble_benchmark.util.metadata import NVSMetadata
from nersemble_benchmark.util.security import validate_nersemble_benchmark_url

BenchmarkType = Literal["nvs", "mono_flame_avatar"]
AssetTypeNvs = Literal["calibration", "images", "alpha_maps", "pointclouds"]
AssetTypeMonoAvatar = Literal['']
# AssetType = Union[AssetTypeNvs, AssetTypeMonoAvatar]
AssetType = AssetTypeNvs
AssetsType = Union[Literal['all'], List[AssetType]]


def main(
        benchmark_folder: Path,
        benchmark_type: BenchmarkType,
        /,
        assets: AssetsType = 'all',
        participant: Union[Literal['all'], List[int]] = 'all',
        pointcloud_frames: Union[Literal['all'], List[int]] = [0],
        n_workers: int = 1,
        overwrite: bool = False):
    """
    Downloads the data for the NeRSemble benchmark.
    This scripts gives various options to select which parts of the benchmark shall be downloaded.
    All downloaded content will be stored in a pre-defined folder structure in the given `benchmark_folder`.

    Parameters
    ----------
    benchmark_folder:
        Where to store the downloaded benchmark data
    benchmark_type:
        There are multiple different benchmarks. Select for which to download the data
    assets:
        Which assets to download
    participant:
        The benchmarks contain multiple participants. Select for which to download the data. Possible values:
            - 'all': download all available participants
            - space-separated list of participant IDs: Download only data for specified participants
    pointcloud_frames:
        Only for NVS benchmark: If the 'pointclouds' asset is selected, specify for which timesteps the pointclouds should be downloaded
            - 'all': download all available pointclouds
            - space-separated list of timesteps: download pointclouds only for specified timestep(s)
    n_workers:
        How many parallel processes should be started to download data. More workers can lead to faster download but potentially overload your system.
    overwrite:
        Whether to overwrite already existing local files
    """

    validate_nersemble_benchmark_url()
    assets = validate_assets(benchmark_type, assets)

    # ----------------------
    # Collect download links
    # ----------------------
    if benchmark_type == 'nvs':
        if participant == 'all':
            benchmark_ids_sequences_and_timesteps = BENCHMARK_NVS_IDS_AND_SEQUENCES
        else:
            benchmark_ids_sequences_and_timesteps = [(p_id, seq_name, BENCHMARK_NVS_TRAIN_SERIALS) for p_id, seq_name in BENCHMARK_NVS_IDS_AND_SEQUENCES if p_id in participant]

        nvs_metadata = NVSMetadata.load()
        benchmark_ids_sequences_and_timesteps = [(p_id, seq_name, serials, nvs_metadata.sequences[p_id].timesteps)
                                                 for p_id, seq_name, serials in benchmark_ids_sequences_and_timesteps]

        relative_urls = collect_relative_urls(benchmark_type, benchmark_ids_sequences_and_timesteps, assets, pointcloud_frames)
        download_urls(benchmark_folder, benchmark_type, relative_urls, overwrite=overwrite, n_workers=n_workers)
    elif benchmark_type == 'mono_flame_avatar':
        if participant == 'all':
            participant_ids = BENCHMARK_MONO_FLAME_AVATAR_IDS
        else:
            participant_ids = participant

        sequences = BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TRAIN
        timesteps = None
        benchmark_ids_sequences_and_timesteps = [(p_id, seq_name, [BENCHMARK_MONO_AVATAR_TRAIN_SERIAL], timesteps) for p_id in participant_ids for seq_name in sequences]

        relative_urls = collect_relative_urls(benchmark_type, benchmark_ids_sequences_and_timesteps, assets)
        download_urls(benchmark_folder, benchmark_type, relative_urls, overwrite=overwrite, n_workers=n_workers)

        # Test inputs for hold-out sequences
        sequences = BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST
        benchmark_ids_sequences_and_timesteps = [(p_id, seq_name, BENCHMARK_MONO_AVATAR_HOLD_OUT_SERIALS, timesteps) for p_id in participant_ids for seq_name in sequences]
        assets_test = [asset for asset in assets if asset in ASSETS[benchmark_type]['test_assets']]

        relative_urls = collect_relative_urls(benchmark_type, benchmark_ids_sequences_and_timesteps, assets_test)
        download_urls(benchmark_folder, benchmark_type, relative_urls, overwrite=overwrite, n_workers=n_workers)


    else:
        raise NotImplementedError(f"Benchmark type {benchmark_type} not implemented")


def validate_assets(benchmark_type: BenchmarkType, assets: AssetsType):
    benchmark_assets = ASSETS[benchmark_type]
    available_asset_keys = []
    for asset_group, asset_set in benchmark_assets.items():
        if asset_group != 'test_assets':
            available_asset_keys.extend(asset_set.keys())
    # available_asset_keys = [asset_name for asset_set in benchmark_assets.values() if asset_set != 'test_assets' for asset_name in asset_set.keys()]
    if assets == 'all':
        assets = available_asset_keys
    else:
        unexpected_assets = [asset_name for asset_name in assets if asset_name not in available_asset_keys]
        assert len(unexpected_assets) == 0, f"Unexpected assets specified for {benchmark_type} benchmark: {unexpected_assets}"

    return assets


def collect_relative_urls(
        benchmarkt_type: BenchmarkType,
        sequence_config: List[Tuple[int, str, List[str], Optional[List[int]]]],
        assets: AssetsType,
        pointcloud_frames: Union[Literal['all'], List[int]] = [0]):
    benchmark_assets = ASSETS[benchmarkt_type]
    relative_urls = []
    for p_id, seq_name, serials, timesteps in sequence_config:
        for asset in assets:
            if asset in benchmark_assets['per_person']:
                relative_url = benchmark_assets['per_person'][asset]
                relative_url = relative_url.format(p_id=p_id)
                relative_urls.append(relative_url)
            elif asset in benchmark_assets['per_cam']:
                for serial in serials:
                    relative_url = benchmark_assets['per_cam'][asset]
                    relative_url = relative_url.format(p_id=p_id, seq_name=seq_name, serial=serial)
                    relative_urls.append(relative_url)
            elif 'per_timestep' in benchmark_assets and asset in benchmark_assets['per_timestep']:
                if asset == 'pointclouds' and pointcloud_frames != 'all':
                    timesteps = pointcloud_frames
                for timestep in timesteps:
                    relative_url = benchmark_assets['per_timestep'][asset]
                    relative_url = relative_url.format(p_id=p_id, seq_name=seq_name, timestep=timestep)
                    relative_urls.append(relative_url)
            elif 'per_sequence' in benchmark_assets and asset in benchmark_assets['per_sequence']:
                relative_url = benchmark_assets['per_sequence'][asset]
                relative_url = relative_url.format(p_id=p_id, seq_name=seq_name)
                relative_urls.append(relative_url)

    return relative_urls


def download_urls(benchmark_folder: Path,
                  benchmark_type: str,
                  relative_urls: List[str],
                  overwrite: bool = False,
                  n_workers: int = 1):

    if n_workers == 1:
        print(f"[Warning] Downloading data with a single worker which may be slow. Consider setting --n_workers to a number greater than 1")
        for relative_url in tqdm(relative_urls):
            absolute_url = f"{NERSEMBLE_BENCHMARK_URL}/{benchmark_type}/{relative_url}"
            target_path = f"{benchmark_folder}/{benchmark_type}/{relative_url}"
            download_file(absolute_url, target_path, overwrite=overwrite)
    else:
        print(f"Downloading data with {n_workers} workers")
        pool = ThreadPool(processes=n_workers)
        futures = []
        for relative_url in relative_urls:
            absolute_url = f"{NERSEMBLE_BENCHMARK_URL}/{benchmark_type}/{relative_url}"
            target_path = f"{benchmark_folder}/{benchmark_type}/{relative_url}"
            future = pool.apply_async(download_file, (absolute_url, target_path, overwrite))
            futures.append(future)

        for future in tqdm(futures):
            future.get()


def main_cli():
    tyro.cli(main)
