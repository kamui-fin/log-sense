from typing import Dict, List
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ServiceConfig:
    is_train: bool
    threshold: float
    coreset_size: int

    enable_trace: bool
    trace_regex: str


@dataclass_json
@dataclass
class GlobalConfig:
    configs: Dict[str, ServiceConfig]
