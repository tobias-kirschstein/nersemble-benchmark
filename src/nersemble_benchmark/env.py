from pathlib import Path

from environs import Env

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/nersemble_benchmark/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("NERSEMBLE_BENCHMARK_"):
    NERSEMBLE_BENCHMARK_URL = env("URL", f"<<<Define NERSEMBLE_BENCHMARK_URL in {env_file_path}>>>")

NERSEMBLE_BENCHMARK_URL_NVS = f"{NERSEMBLE_BENCHMARK_URL}/nvs"

REPO_ROOT = f"{Path(__file__).parent.resolve()}/../.."
