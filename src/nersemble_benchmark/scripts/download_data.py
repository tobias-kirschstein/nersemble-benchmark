import logging
import os
import urllib
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal, Union, List
from urllib.error import HTTPError, URLError

import requests
import tyro
from elias.util import ensure_directory_exists_for_file
from tqdm import tqdm

from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_HOLD_OUT_SERIALS, BENCHMARK_NVS_TRAIN_SERIALS, ASSETS

from nersemble_benchmark.env import NERSEMBLE_BENCHMARK_URL_NVS, NERSEMBLE_BENCHMARK_URL

BenchmarkType = Literal["nvs", "mono-avatar"]
AssetTypeNvs = Literal["calibration", "images", "alpha_maps"]
AssetTypeMonoAvatar = Literal['']
# AssetType = Union[AssetTypeNvs, AssetTypeMonoAvatar]
AssetType = AssetTypeNvs

def download_file(url: str, target_path: str) -> None:
    ensure_directory_exists_for_file(target_path)

    if Path(target_path).exists():
        response = requests.head(url)
        download_size = int(response.headers['content-length'])
        local_file_size = os.path.getsize(target_path)

        if download_size == local_file_size:
            print(f"{target_path} already exists, skipping")
            return
        else:
            print(f"{target_path} seems to be incomplete. Re-downloading...")

    # percent_done = 100 * i / total
    # logging.info("[%.2f%%] Downloading link %d / %d from %s to %s", percent_done, i, total, from_url, to_path)
    # logging.info(f"Downloading file from {url} to {target_path}")
    print(f"Downloading file from {url} to {target_path}")

    try:
        urllib.request.urlretrieve(url, target_path)
    except HTTPError as e:
        logging.error("HTTP error occurred reaching %s: %s", url, e)
        raise e
    except URLError as e:
        logging.error("URL error occurred reaching %s: %s", url, e)
        raise e


def main(
        benchmark_folder: Path,
        benchmark_type: BenchmarkType,
        /,
        assets: Union[Literal['all'], List[AssetType]] = 'all',
        participant_id: Union[Literal['all'], List[int]] = 'all',
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
    """

    benchmark_assets = ASSETS[benchmark_type]
    if assets == 'all':
        assets = [asset_name for asset_set in benchmark_assets.values() for asset_name in asset_set.keys()]

    if benchmark_type == 'nvs':
        if participant_id == 'all':
            benchmark_nvs_ids_and_sequences = BENCHMARK_NVS_IDS_AND_SEQUENCES
        else:
            benchmark_nvs_ids_and_sequences = [(p_id, seq_name) for p_id, seq_name in NERSEMBLE_BENCHMARK_URL_NVS if p_id in participant_id]

        download_links_and_target_paths = []
        relative_urls = []
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

    else:
        raise NotImplementedError(f"Benchmark type {benchmark_type} not implemented")

    # Download data
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
