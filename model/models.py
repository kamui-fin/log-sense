import torch
from typing import Dict, List
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from torch.utils.data import Dataset


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
class GptLogEvent(BaseLogEvent):
    train_strategy: str


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
    chunks: List[LogGPTProcessedInput]
    original_logs: List[List[GptLogEvent]]


class ChunkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"]).cuda(),
            "attention_mask": torch.tensor(sample["attention_mask"]).cuda(),
            "labels": torch.tensor(sample["labels"]).cuda(),
        }