from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ServiceConfig:
    is_train: bool
    threshold: float
    coreset_size: int


@dataclass_json
@dataclass
class GlobalConfig:
    configs: Dict[str, ServiceConfig]


@dataclass_json
@dataclass
class BaseLogEvent:
    service: str
    node: str
    filename: str
    original_text: str
    timestamp: int
    cleaned_text: str
    hash: str


@dataclass_json
@dataclass
class RapidLogEvent(BaseLogEvent):
    tokens: dict


@dataclass_json
@dataclass
class LogGPTProcessedInput:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


@dataclass_json
@dataclass
class LogSequenceEvent:
    hashes: List[str]
    chunk: List[LogGPTProcessedInput]
    original_logs: List[List[BaseLogEvent]]
