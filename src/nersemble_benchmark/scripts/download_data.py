from multiprocessing.pool import ThreadPool
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal, Union, List

import tyro
from elias.util import load_json
from tqdm import tqdm

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_TRAIN_SERIALS, ASSETS
from nersemble_benchmark.env import NERSEMBLE_BENCHMARK_URL_NVS, NERSEMBLE_BENCHMARK_URL
from nersemble_benchmark.util.download import download_file
from nersemble_benchmark.util.metadata import NVSMetadata
from nersemble_benchmark.util.security import validate_nersemble_benchmark_url

BenchmarkType = Literal["nvs", "mono-flame-avatar"]
AssetTypeNvs = Literal["calibration", "images", "alpha_maps", "pointclouds"]
AssetTypeMonoAvatar = Literal['']
# AssetType = Union[AssetTypeNvs, AssetTypeMonoAvatar]
AssetType = AssetTypeNvs


def main(
        benchmark_folder: Path,
        benchmark_type: BenchmarkType,
        /,
        assets: Union[Literal['all'], List[AssetType]] = 'all',
        participant_id: Union[Literal['all'], List[int]] = 'all',
        pointcloud_frames: Union[Literal['all'], List[int]] = [0],
        n_workers: int = 1):
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
    participant_id:
        The benchmarks contain multiple participants. Select for which to download the data
    pointcloud_frames:
        Only for NVS benchmark: If the 'pointclouds' asset is selected, specify for which timesteps the pointclouds should be downloaded
            - 'all': download all available pointclouds
            - space-separated list of timesteps: download pointclouds only for specified timestep(s)
    n_workers:
        How many parallel processes should be started to download data. More workers can lead to faster download but potentially overload your system.
    """

    validate_nersemble_benchmark_url()

    benchmark_assets = ASSETS[benchmark_type]
    available_asset_keys = [asset_name for asset_set in benchmark_assets.values() for asset_name in asset_set.keys()]
    if assets == 'all':
        assets = available_asset_keys
    else:
        unexpected_assets = [asset_name for asset_name in assets if asset_name not in available_asset_keys]
        assert len(unexpected_assets) == 0, f"Unexpected assets specified for {benchmark_type} benchmark: {unexpected_assets}"

    # ----------------------
    # Collect download links
    # ----------------------
    if benchmark_type == 'nvs':
        if participant_id == 'all':
            benchmark_nvs_ids_and_sequences = BENCHMARK_NVS_IDS_AND_SEQUENCES
        else:
            benchmark_nvs_ids_and_sequences = [(p_id, seq_name) for p_id, seq_name in BENCHMARK_NVS_IDS_AND_SEQUENCES if p_id in participant_id]

        relative_urls = []
        nvs_metadata = NVSMetadata.load()
        for p_id, seq_name in benchmark_nvs_ids_and_sequences:
            for asset in assets:
                if asset in benchmark_assets['global']:
                    relative_url = benchmark_assets['global'][asset]
                    relative_url = relative_url.format(p_id=p_id)
                    relative_urls.append(relative_url)
                elif asset in benchmark_assets['per_cam']:
                    for serial in BENCHMARK_NVS_TRAIN_SERIALS:
                        relative_url = benchmark_assets['per_cam'][asset]
                        relative_url = relative_url.format(p_id=p_id, seq_name=seq_name, serial=serial)
                        relative_urls.append(relative_url)
                elif asset in benchmark_assets['per_timestep']:
                    timesteps = nvs_metadata.sequences[p_id].timesteps
                    if asset == 'pointclouds' and pointcloud_frames != 'all':
                        timesteps = pointcloud_frames
                    for timestep in timesteps:
                        relative_url = benchmark_assets['per_timestep'][asset]
                        relative_url = relative_url.format(p_id=p_id, seq_name=seq_name, timestep=timestep)
                        relative_urls.append(relative_url)

    else:
        raise NotImplementedError(f"Benchmark type {benchmark_type} not implemented")

    # -------------
    # Download data
    # -------------
    if n_workers == 1:
        print(f"[Warning] Downloading data with a single worker which may be slow. Consider setting --n_workers to a number greater than 1")
        for relative_url in tqdm(relative_urls):
            absolute_url = f"{NERSEMBLE_BENCHMARK_URL}/{benchmark_type}/{relative_url}"
            target_path = f"{benchmark_folder}/{benchmark_type}/{relative_url}"
            download_file(absolute_url, target_path)
    else:
        print(f"Downloading data with {n_workers} workers")
        pool = ThreadPool(processes=n_workers)
        futures = []
        for relative_url in relative_urls:
            absolute_url = f"{NERSEMBLE_BENCHMARK_URL}/{benchmark_type}/{relative_url}"
            target_path = f"{benchmark_folder}/{benchmark_type}/{relative_url}"
            future = pool.apply_async(download_file, (absolute_url, target_path))
            futures.append(future)

        for future in tqdm(futures):
            future.get()


def main_cli():
    tyro.cli(main)
