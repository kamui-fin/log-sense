import pprint
import requests
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

def get_config(base_url = 'http://localhost:3000') -> GlobalConfig:
    # GET request to config microservice
    # for now, let's mock it out
    # return GlobalConfig(configs={'service1': ServiceConfig(mode="test", threshold=-481.175, coreset_number=2)})
    
    # GET to http://localhost:3000/api/trpc/config.getServices
    response = requests.get(f'{base_url}/api/trpc/config.getServices').json()
    service_list = response['result']['data']['json']['data']
    configs = {}
    for service in service_list:
        configs[service['name']] = ServiceConfig(
            is_train=service['isTrain'],
            threshold=service['threshold'],
            coreset_size=service['coresetSize']
        )
    return GlobalConfig(configs=configs)


def listen_inference(svc):
    consumer = KafkaConsumer(
        f"{svc}-processed", "config-change", "mark-normal",
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
    logging.info(f'Inference service "{svc}" started')
    logging.info(f'Using previous config {config}')
    inferencer = RapidInferenceAPI(client, svc, config)
    for message in consumer:
        if message.topic == "config-change":
            new_config = get_config()
            logging.info(f'Received config update: {new_config}')
            logging.info(f'Old config: {inferencer.config}')
            inferencer.reload_config(new_config)
            logging.info(f'New config: {inferencer.config}')
        else:
            event = LogEvent.from_dict(message.value)
            logging.info(f"Received log: {event}")
            if message.topic == "mark-normal":
                logging.info(f"Marking event as normal: {event}")
                inferencer.mark_normal(event)
                continue
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