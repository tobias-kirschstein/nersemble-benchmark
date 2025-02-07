import hashlib
from pathlib import Path

from elias.util import ensure_directory_exists_for_file

from nersemble_benchmark.constants import NERSEMBLE_ACCESS_FORM_URL
from nersemble_benchmark.env import NERSEMBLE_BENCHMARK_URL, env, env_file_path


def _prompt_nersemble_benchmark_url():
    print("To download the NeRSemble benchmark data, please do the following:")
    print(f" 1. Request access to the NeRSemble dataset via {NERSEMBLE_ACCESS_FORM_URL}")
    print(f" 2. Once your request was approved, you will receive a mail with the download url for the benchmark. Enter it here")
    nersemble_benchmark_url = input("Please enter the NERSEMBLE_BENCHMARK_URL from your access mail:")
    nersemble_benchmark_url = nersemble_benchmark_url.strip()

    env_dict = env.dump()
    env_dict["NERSEMBLE_BENCHMARK_URL"] = nersemble_benchmark_url
    global NERSEMBLE_BENCHMARK_URL
    NERSEMBLE_BENCHMARK_URL = nersemble_benchmark_url
    ensure_directory_exists_for_file(env_file_path)
    for key, value in env_dict.items():
        with open(env_file_path, "w+") as f:
            f.write(f"{key}=\"{value}\"\n")


def validate_nersemble_benchmark_url():
    env_file_path = Path(f"{Path.home()}/.config/nersemble_benchmark/.env")
    unset_url = f"<<<Define NERSEMBLE_BENCHMARK_URL in {env_file_path}>>>"

    if NERSEMBLE_BENCHMARK_URL == unset_url:
        _prompt_nersemble_benchmark_url()

    while True:
        salt = "aL8jN4%Y1h%9fG7U"
        hash = hashlib.md5(f"{salt}-{NERSEMBLE_BENCHMARK_URL}".encode()).hexdigest()
        if hash == "c0140cfad39e3e15479451c389c71a5b":
            break
        else:
            print("The NERSEMBLE_BENCHMARK_URL that you configured is not correct.")
            _prompt_nersemble_benchmark_url()
