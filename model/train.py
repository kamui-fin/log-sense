from pprint import pprint
import torch
import time
from models import ChunkDataset
import schedule
from log_gpt import LogGPTInferenceAPI
from pymongo import MongoClient
from torch.utils.data import DataLoader
import logging
import os

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", 27017)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def task():
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client.log_sense
    collection = db.gptlogs
    pipeline = [
        {"$match": {"is_anomaly": False}},
        {"$group": {"_id": "$train_strategy", "chunks": {"$push": "$chunk"}}},
    ]

    documents = collection.aggregate(pipeline)
    chunks_by_strat = {}
    for document in documents:
        train_strategy = document.get("_id")
        chunks = document.get("chunks")
        chunks_by_strat[train_strategy] = chunks

    if not len(chunks_by_strat):
        logging.error('No data to train for today')
        return

    # pre-train
    pre_train_chunks = chunks_by_strat["pre-train"]
    pprint(pre_train_chunks)
    # begin_pretraining(pre_train_chunks)

    # fine-tune
    fine_tune_chunks = chunks_by_strat["finetune"]
    pprint(fine_tune_chunks)
    # begin_finetuning(fine_tune_chunks)


def begin_pretraining(chunks):
    train_loader = DataLoader(ChunkDataset(chunks), batch_size=2, shuffle=True)
    model = LogGPTInferenceAPI(None)
    model.pretrain_model(train_loader)


def begin_finetuning(sequences):
    sequences = torch.tensor([chunk["input_ids"] for chunk in sequences], device=device)
    model = LogGPTInferenceAPI(None)
    model.finetune(sequences)


# Schedule the task to run every day at 12 am
schedule.every().day.at("00:00").do(task)

task()
while True:
    schedule.run_pending()
    time.sleep(1)