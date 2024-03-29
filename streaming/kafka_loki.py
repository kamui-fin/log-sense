import asyncio
import websockets
import json
import threading
from datetime import datetime, timedelta
from pprint import pprint
from urllib.parse import quote
from kafka import KafkaProducer

LOKI_ENDPOINT = "ws://localhost:3100/loki/api/v1/tail"
SERVICES = [
    'test',
    'lol'
]

bootstrap_servers = 'localhost:9092'
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

async def tail_loki(uri):
    async with websockets.connect(uri) as websocket:
        while True:
            logs = json.loads(await websocket.recv())
            streams = logs["streams"]
            for stream in streams:
                meta = stream['stream']
                filename, node, service = meta['filename'], meta['node'], meta['service']
                for timestamp, log in stream['values']:
                    kafka_entry = {
                        'node': node,
                        'filename': filename,
                        'timestamp': timestamp,
                        'log': log
                    }
                    kafka_entry_ser = json.dumps(kafka_entry).encode('utf-8')
                    producer.send(service, kafka_entry_ser)

async def main():
    async with asyncio.TaskGroup() as tg:
        for service in SERVICES:
            query = quote(f'{{service="{service}"}}')
            uri = f'{LOKI_ENDPOINT}?query={query}&limit=100&delay_for=0'
            tg.create_task(tail_loki(uri))
            
asyncio.run(main())
producer.close()