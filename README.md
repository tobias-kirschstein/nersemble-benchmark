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

