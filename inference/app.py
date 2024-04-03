from dataclasses import dataclass
import json
from typing import Literal, Union
from kafka import KafkaConsumer, KafkaProducer
import argparse

from qdrant_client import QdrantClient

from inference.rapid import RapidInferenceAPI

producer = KafkaProducer(bootstrap_servers="localhost:9092")
client = QdrantClient("localhost", port=6333)

@dataclass
class Config:
    mode: Literal['train', 'test']
    threshold: float
    coreset_number: int


@dataclass
class LogEvent:
    service: str
    node: str
    filename: str
    text: str
    tokens: dict
    hash: int
    timestamp: int

    def from_dict(self, data):
        return LogEvent(
            service=data["service"],
            node=data["node"],
            filename=data["filename"],
            text=data["text"],
            hash=data["hash"],
            timestamp=data["timestamp"],
        )
    
    def to_dict(self):
        return {
            "service": self.service,
            "node": self.node,
            "filename": self.filename,
            "text": self.text,
            "hash": self.hash,
            "timestamp": self.timestamp,
        }

def get_config(base_url) -> Config:
    # GET request to config microservice 
    # for now, let's mock it out
    return Config(mode='test', threshold=-470, coreset_number=2)

def listen_inference(svc):
    consumer = KafkaConsumer(
        [f"{svc}-processed", "config-change"],
        bootstrap_servers=["localhost:9092"],
        group_id="inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    inferencer = RapidInferenceAPI(client, producer, svc)
    config = get_config()
    for message in consumer:
        if message.topic == "config-change":
            config_update = Config.from_dict(message.value)
            inferencer.reload_config(config_update)
        else:
            event = LogEvent.from_dict(message.value)
            print("Received event:", event)
            # score, is_anomaly = inferencer.run_inference(event)
            # producer.send(
            #     "predictions", value={"score": score, "is_anomaly": is_anomaly, **event}
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference listener")
    parser.add_argument("svc", help="Service name")
    args = parser.parse_args()

    listen_inference(args.svc)