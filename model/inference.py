import requests
import sys
import os
from torch.utils.data import DataLoader

from models import ChunkDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from log_gpt import LogGPTInferenceAPI
from models import (
    GlobalConfig,
    LogSequenceEvent,
    ServiceConfig,
    RapidLogEvent,
)
import json
from kafka import KafkaConsumer, KafkaProducer
import argparse
from qdrant_client import QdrantClient
import logging
from rapid import RapidInferenceAPI

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", 6333)
KAFKA_URI = os.getenv("KAFKA_URI", "localhost:9092")
LOGSENSE_BACKEND_URI = os.getenv("LOGSENSE_BACKEND_URI", "http://localhost:3000")

producer = KafkaProducer(bootstrap_servers=KAFKA_URI)
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

logging.basicConfig(level=logging.INFO)


def get_config() -> GlobalConfig:
    response = requests.get(
        f"{LOGSENSE_BACKEND_URI}/api/trpc/config.getServices"
    ).json()
    service_list = response["result"]["data"]["json"]["data"]
    configs = {}
    for service in service_list:
        name = service["name"]
        del service["_id"]
        del service["name"]
        del service["description"]
        del service["__v"]
        configs[name] = ServiceConfig(**service)
    return GlobalConfig(configs=configs)


def deserializer(msg):
    try:
        return json.loads(msg.decode("utf-8"))
    except:
        return {"invalid": True}


def listen_inference_rapid():
    consumer = KafkaConsumer(
        f"rapid-processed",
        "config-change",
        "mark-normal",
        bootstrap_servers=[KAFKA_URI],
        group_id="rapid-inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=deserializer,
    )
    config = get_config()
    services = [svc for svc in config.configs.keys()]
    logging.info(f"RAPID Inference service started")
    logging.info(f"Using previous config {config}")
    inferencers = {svc: RapidInferenceAPI(client, svc, config) for svc in services}
    for message in consumer:
        if message.topic == "config-change":
            new_config = get_config()
            logging.info(f"Received config update: {new_config}")
            for inferencer in inferencers.values():
                inferencer.reload_config(new_config)
        elif message.value.get("invalid"):
            continue
        else:
            event = RapidLogEvent.from_dict(message.value)
            service = event.service
            inferencer = inferencers[service]
            logging.info(f"Received log: {event}")
            if message.topic == "mark-normal":
                inferencer.mark_normal(event)
                continue
            result = inferencer.run_inference(event)
            if result is None:
                # Found in normal db
                continue
            score, is_anomaly = result
            output = event.to_dict()
            del output["tokens"]
            producer.send(
                "predictions",
                value=json.dumps(
                    {
                        "type": "rapid",
                        "score": score,
                        "is_anomaly": is_anomaly,
                        **output,
                    }
                ).encode("utf-8"),
            )


def listen_inference_gpt():
    """
    How training works:
    - When training mode is enabled:
        - Add to pretraining dataset if total_trained <= stop_train_after
        - Otherwise we add to finetuning dataset
    - All interactive feedback results will be finetuned upon with RL
    - At the end of the day, pre-training & finetuning will be performed on datasets. Should be tracked through tensorboard

    How inference works:
    - Every minute, a batch of processed log sequences arrives with accompanying original log data.
    - We feed this batch to the model at once and mark the timestamp range as anomalous (for now).
    - Frontend should prepare for a different kind of model output compared to rapid inference.
    """

    consumer = KafkaConsumer(
        f"gpt-processed",
        "config-change",
        bootstrap_servers=[KAFKA_URI],
        group_id="gpt-inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=deserializer,
    )
    config = get_config()
    logging.info(f"LogGPT Inference service started")
    logging.info(f"Using previous config {config}")
    inferencers = {
        svc: LogGPTInferenceAPI(svc, cfg) for svc, cfg in config.configs.items()
    }
    for message in consumer:
        if message.topic == "config-change":
            config = get_config()
            logging.info(f"Received config update: {config}")
            for inferencer in inferencers.values():
                inferencer.reload_config(config.configs[inferencer.service])
        elif message.value.get("invalid"):
            continue
        else:
            log_batch: LogSequenceEvent = LogSequenceEvent.from_dict(message.value)
            svc = log_batch.original_logs[0][0].service
            num_test_samples = len(log_batch.hashes)
            logging.info(f"Received batch: {log_batch}")

            inferencer = inferencers[svc]
            is_train = config.configs[svc].is_train
            if not is_train:
                data_loader = DataLoader(
                    ChunkDataset([c.to_dict() for c in log_batch.chunks]),
                    batch_size=2,  # TODO:
                    shuffle=True,
                )
                is_anomaly = inferencer.run_inference(data_loader)
                logging.info("Generated predictions!", is_anomaly)

            for i in range(num_test_samples):
                if is_train or is_anomaly[i]:
                    status = False if is_train else bool(is_anomaly[i])
                    producer.send(
                        "predictions",
                        value=json.dumps(
                            {
                                "hash": log_batch.hashes[i],
                                "type": "log_gpt",
                                "is_anomaly": status,
                                "chunk": log_batch.chunks[i].to_dict(),
                                "original_logs": [
                                    l.to_dict() for l in log_batch.original_logs[i]
                                ],
                                "train_strategy": log_batch.original_logs[i][
                                    0
                                ].train_strategy,
                            }
                        ).encode("utf-8"),
                    )


if __name__ == "__main__":
    model = os.getenv("MODEL", "rapid")

    if model == "rapid":
        listen_inference_rapid()
    elif model == "gpt":
        listen_inference_gpt()
    else:
        logging.error(f'Unknown model "{model}"')
        sys.exit(1)
