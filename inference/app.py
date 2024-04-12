import pprint
import pymongo
import requests
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from inference.log_gpt import LogGPTInferenceAPI
from inference.models import (
    GlobalConfig,
    LogSequenceEvent,
    ServiceConfig,
    RapidLogEvent,
)
import json
from typing import Literal, Union
from kafka import KafkaConsumer, KafkaProducer
import argparse
from qdrant_client import QdrantClient
import logging
from inference.rapid import RapidInferenceAPI

producer = KafkaProducer(bootstrap_servers="localhost:9092")
client = QdrantClient("localhost", port=6333)

logging.basicConfig(level=logging.INFO)


def get_config(base_url="http://localhost:3000") -> GlobalConfig:
    # GET request to config microservice
    # for now, let's mock it out
    # return GlobalConfig(configs={'service1': ServiceConfig(mode="test", threshold=-481.175, coreset_number=2)})

    # GET to http://localhost:3000/api/trpc/config.getServices
    response = requests.get(f"{base_url}/api/trpc/config.getServices").json()
    service_list = response["result"]["data"]["json"]["data"]
    configs = {}
    for service in service_list:
        configs[service["name"]] = ServiceConfig(
            is_train=service["isTrain"],
            threshold=service["threshold"],
            coreset_size=service["coresetSize"],
        )
    return GlobalConfig(configs=configs)


def listen_inference_rapid(svc):
    consumer = KafkaConsumer(
        f"{svc}-rapid-processed",
        "config-change",
        bootstrap_servers=["localhost:9092"],
        group_id="inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    config = get_config()
    if svc not in config.configs:
        logging.error(f'No config found for service "{svc}"')
        return
    logging.info(f'RAPID Inference service "{svc}" started')
    logging.info(f"Using previous config {config}")
    inferencer = RapidInferenceAPI(client, svc, config)
    for message in consumer:
        if message.topic == "config-change":
            new_config = get_config()
            logging.info(f"Received config update: {new_config}")
            logging.info(f"Old config: {inferencer.config}")
            inferencer.reload_config(new_config)
            logging.info(f"New config: {inferencer.config}")
            # TODO: handle train mode
        else:
            event = RapidLogEvent.from_json(message.value)
            logging.info(f"Received log: {event}")
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


def listen_inference_gpt(svc):
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
        - In the future, it should also expect different kinds of grouping for inference.
            - Single
            - Window
            - Trace
    """

    consumer = KafkaConsumer(
        f"{svc}-gpt-processed",
        "config-change",
        "mark-normal",
        bootstrap_servers=["localhost:9092"],
        group_id="inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    config = get_config()
    if svc not in config.configs:
        logging.error(f'No config found for service "{svc}"')
        return
    logging.info(f'LogGPT Inference service "{svc}" started')
    logging.info(f"Using previous config {config}")
    inferencer = LogGPTInferenceAPI(client, svc, config)
    # client = pymongo.MongoClient("mongodb://localhost:27017/")
    # db = client['log_sense']
    # collection = client['logs']
    for message in consumer:
        if message.topic == "config-change":
            config = get_config()
            logging.info(f"Received config update: {config}")
            logging.info(f"Old config: {inferencer.config}")
            inferencer.reload_config(config)
            logging.info(f"New config: {inferencer.config}")
        else:
            log_batch = LogSequenceEvent.from_json(message.value)
            num_test_samples = len(log_batch.hashes)
            logging.info(f"Received batch: {log_batch}")
            is_train = config.configs[svc].is_train
            if not is_train:
                is_anomaly = inferencer.run_inference(log_batch)

            for i in range(num_test_samples):
                if is_train or is_anomaly[i]:
                    producer.send(
                        "predictions",
                        value=json.dumps(
                            {
                                "type": "log_gpt",
                                "is_anomaly": False if is_train else is_anomaly[i],
                                **TODO,
                            }
                        ).encode("utf-8"),
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference listener")
    # choose model name between [rapid, gpt]
    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("svc", help="Service name")
    args = parser.parse_args()

    if args.model == "rapid":
        listen_inference_rapid(args.svc)
    elif args.model == "gpt":
        listen_inference_gpt(args.svc)
    else:
        logging.error(f'Unknown model "{args.model}"')
        sys.exit(1)
