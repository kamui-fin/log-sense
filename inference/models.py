from typing import Dict, List, Literal, Optional
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    is_train: bool
    threshold: float
    coreset_size: int

    def from_dict(data):
        return ServiceConfig(
            is_train=data["is_train"],
            threshold=data["threshold"],
            coreset_size=data["coreset_size"]
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
    original_text: str
    cleaned_text: str
    hash: str
    timestamp: int

    tokens: Optional[dict] = None

    def from_dict(data):
        return LogEvent(
            service=data["service"],
            node=data["node"],
            filename=data["filename"],
            original_text=data["original_text"],
            cleaned_text=data["cleaned_text"],
            hash=data["hash"],
            timestamp=data["timestamp"],
            tokens=data.get("tokens", None)
        )

    def to_dict(self):
        log_dict = {
            "service": self.service,
            "node": self.node,
            "filename": self.filename,
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "hash": self.hash,
            "timestamp": self.timestamp,
        }
        if self.tokens is not None:
            log_dict["tokens"] = self.tokens
        return log_dict
