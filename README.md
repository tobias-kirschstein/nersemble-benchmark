# NeRSemble Photorealistic 3D Head Avatar Benchmark

This is the official repository containing the devkit for downloading the data and submitting results to the NeRSemble Photorealistic 3D Head Avatar benchmarks.

## 1. Setup

### Repository installation
```shell
pip install nersemble_benchmark@git+ssh://git@github.com/tobias-kirschstein/nersemble-benchmark.git
```

### Environment variables
Create a file at `~/.config/nersemble_benchmark/.env` with following content:
```python
NERSEMBLE_BENCHMARK_URL="<<<URL YOU GOT WHEN REQUESTING ACCESS TO NERSEMBLE>>>"
```

## 2. Download

After installation of the benchmark repository, a `nersemble-benchmark-download` command will be available in your environment. 
This is the main tool to download the benchmark data. To get a detailed description of download options, run `nersemble-benchmark-download --help`.
In the following, `${benchmark_folder}` denotes the path to your local folder where the benchmark data should be downloaded to. 

### NVS Benchmark download

```shell
nersemble-benchmark-download ${benchmark_folder} nvs 
```

## 3. Usage

### Data Managers

The benchmark repository provides data managers to simplify loading individual assets such as images in Python code. 

```python
from nersemble_benchmark.data.benchmark_data import NVSDataManager
from nersemble_benchmark.constants import BENCHMARK_NVS_IDS_AND_SEQUENCES, BENCHMARK_NVS_TRAIN_SERIALS

benchmark_folder = "path/to/local/benchmark/folder"
participant_id, sequence_name = BENCHMARK_NVS_IDS_AND_SEQUENCES[0]  # <- Use first benchmark subject
serial = BENCHMARK_NVS_TRAIN_SERIALS[0]  # <- Use first train camera

data_manager = NVSDataManager(benchmark_folder, participant_id)
```

#### Load image
```python
image = data_manager.load_image(sequence_name, serial, 0, apply_alpha_map=True)  # <- Load first frame and remove background
```
![static/images/example_image.jpg](static/images/example_image.jpg)

#### Load Alpha Map
```python
image = data_manager.load_alpha_map(sequence_name, serial, 0)  # <- Load alpha map
```
![static/images/example_alpha_map.jpg](static/images/example_alpha_map.jpg)

#### Load cameras

```python
camera_params = data_manager.load_camera_calibration()
world_2_cam_pose = camera_params.world_2_cam[serial]  # <- 4x4 extrinsic matrix in OpenCV camera coordinate convention
intrinsics = camera_params.intrinsics # <- 3x3 intrinsic matrix
```