import json
from kafka import KafkaConsumer
import argparse

def process_event(event):
    # Replace this with your own logic to process the event
    print("Received event:", event)

def kafka_consumer(topic_name):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=['localhost:9092'],
        group_id='inference-group',
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    for message in consumer:
        process_event(message.value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference listener")
    parser.add_argument("svc", help="Service name")
    args = parser.parse_args()

    kafka_consumer(f'{args.svc}-processed')