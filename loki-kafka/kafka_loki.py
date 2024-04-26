import asyncio
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from typing import List
import requests
import websockets
import json
from urllib.parse import quote
from kafka import KafkaProducer
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)

load_dotenv()

LOKI_ENDPOINT = os.getenv("LOKI_ENDPOINT", "ws://localhost:3100")
KAFKA_BROKER_URI = os.getenv("KAFKA_BROKER_URI", "localhost:9092")

admin_client = KafkaAdminClient(
    bootstrap_servers=KAFKA_BROKER_URI, client_id="topic-init"
)
producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER_URI)

init_topics = [
    "loki",
    "rapid-processed",
    "gpt-processed",
    "predictions",
    "config-change",
    "mark-normal",
]


def create_topic(topic_name=None, num_partitions=1, replication_factor=1):
    try:
        topic_list = []
        topic_list.append(
            NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
            )
        )
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        return True
    except TopicAlreadyExistsError as err:
        logging.error(
            f"Request for topic creation is failed as {topic_name} is already created due to {err}"
        )
        return False
    except Exception as err:
        logging.error(f"Request for topic creation is failing due to {err}")
        return False


def shorten_timestamp(timestamp):
    return int(str(timestamp)[:15])


async def tail_loki(uri):
    # FIXME: Tail timeouts after 1hr
    async with websockets.connect(uri) as websocket:
        logging.info(f"Successfully connected to {uri}")
        while True:
            logs = json.loads(await websocket.recv())
            streams = logs["streams"]
            for stream in streams:
                meta = stream["stream"]
                filename, node, service = (
                    meta["filename"],
                    meta["node"],
                    meta["service"],
                )
                for timestamp, log in stream["values"]:
                    timestamp = shorten_timestamp(timestamp)
                    kafka_entry = {
                        "node": node,
                        "filename": filename,
                        "timestamp": timestamp,
                        "service": service,
                        "original_text": log,
                    }
                    logging.info(f"Recieved loki entry: {kafka_entry}")
                    kafka_entry_ser = json.dumps(kafka_entry).encode("utf-8")
                    producer.send("loki", kafka_entry_ser)
                    logging.info(f"Sent to kafka!")


# Create topics if they don't exist
for topic in init_topics:
    create_topic(topic)


query = quote(f'{{service=~".+"}}')
uri = f"{LOKI_ENDPOINT}/loki/api/v1/tail?query={query}&limit=1&delay_for=0"
asyncio.run(tail_loki(uri))
producer.close()
