from typing import Dict, List, Literal
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    mode: Literal["train", "test"]
    threshold: float
    coreset_number: int

    def from_dict(data):
        return ServiceConfig(
            mode=data["mode"],
            threshold=data["threshold"],
            coreset_number=data["coreset_number"]
        )

@dataclass
class GlobalConfig:
    configs: Dict[str, ServiceConfig]

    def from_dict(data):
        return GlobalConfig(
            configs={name: ServiceConfig.from_dict(config) for name, config in data.items()}
        )


@dataclass
class LogEvent:
    service: str
    node: str
    filename: str
    text: str
    tokens: dict
    hash: int
    timestamp: int

    def from_dict(data):
        return LogEvent(
            service=data["service"],
            node=data["node"],
            filename=data["filename"],
            text=data["text"],
            hash=data["hash"],
            timestamp=data["timestamp"],
            tokens=data['tokens']
        )

    def to_dict(self):
        return {
            "service": self.service,
            "node": self.node,
            "filename": self.filename,
            "text": self.text,
            "hash": self.hash,
            "timestamp": self.timestamp,
            "tokens": self.tokens
        }
