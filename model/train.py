from collections import defaultdict
from pathlib import Path
from pprint import pprint
from kafka import KafkaProducer
from minio import Minio
import torch
import time
from models import ChunkDataset, GlobalConfig
import schedule
from log_gpt import LogGPTInferenceAPI
from pymongo import MongoClient
from torch.utils.data import DataLoader
from models import get_config
import logging
from torch.utils.tensorboard import SummaryWriter
import os

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", 27017)

KAFKA_URI = os.getenv("KAFKA_URI", "localhost:9092")

MINIO_URI = os.getenv("MINIO_URI", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", None)
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", None)

if MINIO_SECRET_KEY is None or MINIO_ACCESS_KEY is None:
    logging.error("Minio credentials not found")
    quit(1)

minio_client = Minio(
    MINIO_URI,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

producer = KafkaProducer(bootstrap_servers=KAFKA_URI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache_path = Path(os.getenv("CACHE_PATH", "/tmp"))

if not cache_path.exists():
    logging.error("Cache path does not exist")
    quit(1)

logging.info(f"Using cache path {cache_path}")

writer = SummaryWriter("/runs")


def task():
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client.log_sense
    collection = db.gptlogs
    pipeline = [
        # {"$match": {"is_anomaly": False}},
        {"$unwind": "$original_logs"},
        {
            "$group": {
                "_id": {
                    "train_strategy": "$train_strategy",
                    "service": "$original_logs.service",
                },
                "chunks": {"$push": "$chunk"},
            }
        },
    ]

    documents = collection.aggregate(pipeline)
    chunks_by_strat = defaultdict(lambda: defaultdict(list))
    for document in documents:
        group_key = document.get("_id")
        service = group_key["service"]
        train_strategy = group_key["train_strategy"]

        chunks = document.get("chunks")
        chunks_by_strat[train_strategy][service] = chunks

    if not len(chunks_by_strat):
        logging.error("No data to train for today")
        return

    config = get_config()
    print(config.configs)
    # pre-train
    pre_train_chunks = chunks_by_strat["pre-train"]
    begin_pretraining(pre_train_chunks, config)

    # fine-tune
    fine_tune_chunks = chunks_by_strat["finetune"]
    begin_finetuning(fine_tune_chunks, config)

    producer.flush()
    writer.flush()


def begin_pretraining(chunks_grouped, config: GlobalConfig):
    for service, chunks in chunks_grouped.items():
        train_loader = DataLoader(ChunkDataset(chunks), batch_size=2, shuffle=True)
        model = LogGPTInferenceAPI(
            service, config.configs[service], cache_path, minio_client
        )
        model.pretrain_model(train_loader, writer)

    producer.send("model-update", value={"msg": "NEW_MODEL_PRETRAINED"})


def begin_finetuning(sequences_grouped, config):
    for service, seqs in sequences_grouped.items():
        sequences = torch.tensor([chunk["input_ids"] for chunk in seqs], device=device)
        model = LogGPTInferenceAPI(
            service, config.configs[service], cache_path, minio_client
        )
        model.finetune(sequences, writer)

    producer.send("model-update", value={"msg": "NEW_MODEL_FINETUNED"})


# Schedule the task to run every day at 12 am
schedule.every().day.at("00:00").do(task)

task()
producer.close()

# try:
#     while True:
#         schedule.run_pending()
#         time.sleep(1)
# except:
#     producer.close()
#     writer.close()
