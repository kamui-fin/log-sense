from typing import Dict, List, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class RegexSub:
    pattern: str
    replacement: str

@dataclass_json
@dataclass
class ServiceConfig:
    is_train: bool
    threshold: float
    coreset_size: int

    enable_trace: bool

    # log-gpt
    top_k: int
    max_pretrain: int
    context_size: int
    lr_pretraining: float
    lr_finetuning: float
    train_batch_size: int
    num_episodes: int
    num_epochs: int
    vocab_size: int


    regex_subs: List[RegexSub]
    trace_regex: Optional[str] = ""


@dataclass_json
@dataclass
class GlobalConfig:
    configs: Dict[str, ServiceConfig]
