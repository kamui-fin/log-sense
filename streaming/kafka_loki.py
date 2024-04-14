import asyncio
import websockets
import requests
import json
from urllib.parse import quote
from kafka import KafkaProducer
import logging

LOKI_ENDPOINT = "ws://localhost:3100/loki/api/v1/tail"
SERVICES = ["service1"]

bootstrap_servers = "localhost:9092"
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

def shorten_timestamp(timestamp):
    return int(str(timestamp)[:14])


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
                    print(f"Recieved loki entry: {kafka_entry}")
                    kafka_entry_ser = json.dumps(kafka_entry).encode("utf-8")
                    producer.send("loki", kafka_entry_ser)
                    print(f"Sent to kafka!")


async def main():
    async with asyncio.TaskGroup() as tg:
        for service in SERVICES:
            query = quote(f'{{service="{service}"}}')
            uri = f"{LOKI_ENDPOINT}?query={query}&limit=1&delay_for=0"
            tg.create_task(tail_loki(uri))


asyncio.run(main())
producer.close()
