import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from inference.models import GlobalConfig, ServiceConfig, LogEvent
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

def get_config(base_url = '') -> GlobalConfig:
    # GET request to config microservice
    # for now, let's mock it out
    return GlobalConfig(configs={'service1': ServiceConfig(mode="test", threshold=-481.175, coreset_number=2)})


def listen_inference(svc):
    consumer = KafkaConsumer(
        f"{svc}-processed", "config-change",
        bootstrap_servers=["localhost:9092"],
        group_id="inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    config = get_config()
    logging.info(f'Inference service "{svc}" started')
    logging.info(f'Using previous config {config}')
    inferencer = RapidInferenceAPI(client, svc, config)
    for message in consumer:
        if message.topic == "config-change":
            config_update = GlobalConfig.from_dict(message.value)
            logging.info(f'Received config update: {config_update}')
            logging.info(f'Old config: {inferencer.config}')
            inferencer.reload_config(config_update)
            logging.info(f'New config: {inferencer.config}')
        else:
            event = LogEvent.from_dict(message.value)
            logging.info(f"Received log: {event}")
            result = inferencer.run_inference(event)
            if result is None:
                continue
            score, is_anomaly = result
            output = event.to_dict()
            del output['tokens']
            print(output, score, is_anomaly)
            producer.send(
                "predictions", value=json.dumps({"score": score, "is_anomaly": is_anomaly, **output}).encode('utf-8')
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference listener")
    parser.add_argument("svc", help="Service name")
    args = parser.parse_args()

    listen_inference(args.svc)