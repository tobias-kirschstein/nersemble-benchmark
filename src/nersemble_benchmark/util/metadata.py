import json
from urllib.request import urlopen

from elias.config import Config
from dataclasses import dataclass
from typing import List, Dict

from elias.util import load_json

from nersemble_benchmark.env import NERSEMBLE_BENCHMARK_URL_NVS


@dataclass
class NVSSequenceMetadata(Config):
    sequence_name: str
    timesteps: List[int]


@dataclass
class NVSMetadata(Config):
    sequences: Dict[int, NVSSequenceMetadata]

    @staticmethod
    def load() -> 'NVSMetadata':
        with urlopen(f"{NERSEMBLE_BENCHMARK_URL_NVS}/metadata.json") as f:
            nvs_metadata = json.load(f)
        return NVSMetadata.from_json(nvs_metadata)

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        # Config classes have the issue that dicts with integer keys are not parsed correctly because in JSON they are stored as strings
        loaded_config['sequences'] = {int(p_id): sequences for p_id, sequences in loaded_config['sequences'].items()}
        super()._backward_compatibility(loaded_config)


