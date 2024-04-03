from typing import Union
import pandas as pd
from kafka import KafkaProducer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from transformers import BertTokenizer, BertModel
import torch
import hashlib
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from tqdm.notebook import tqdm

from inference.app import Config, LogEvent

model = BertModel.from_pretrained("bert-base-uncased").to("cuda:0")

def distance(query_emb, train_embedding):
    dot_products = query_emb @ train_embedding.T
    query_normalized = query_emb.square().sum(dim=-1).sqrt()
    train_normalized = train_embedding.square().sum(dim=-1).sqrt()
    normalized_pairs = query_normalized.unsqueeze(0).T * train_normalized.unsqueeze(0)
    max_similarity = (dot_products / normalized_pairs).max(dim=-1).values.sum(dim=-1)
    return 1 - max_similarity

def get_sentence_embedding(log: LogEvent):
    output = model(**log.tokens)
    embeddings = output.last_hidden_state
    # we must deallocate from gpu mem!
    del encoded_input
    return embeddings.detach().cpu()  # use first for [CLS]

class RapidInferenceAPI:
    def __init__(self, client, service, config):
        self.client = client
        self.service = service
        self.config = config

    def reload_config(self, config: Config):
        self.config = config

    def init_qdrant(self, client):
        for collection_name in [f'normal-{self.service}', 'test']:
            if collection_name not in client.list_collections():
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=model.config.hidden_size, distance=Distance.COSINE
                    ),
                )

    def generate_point(self, log_event: LogEvent):
        embedding = get_sentence_embedding(log_event)
        cls_embedding = embedding[0, 0].numpy()
        token_embeddings = embedding[0, 1:].numpy()
        point = PointStruct(
            id=hash,
            vector=cls_embedding,
            payload={"tokens": token_embeddings.tolist()},
        )
        return point

    def add_log_to_collection(self, log: LogEvent, collection='normal'):
        point = self.generate_point(log)
        self.client.upsert(collection_name=collection, points=[point])
        return point


    def get_embedding_test(self, test_log: LogEvent):
        point = self.get_point_from_test_col(test_log.hash)
        if not point:
            point = self.add_log_to_collection(test_log, 'test')
        else:
            print("Fetching embedding from cache!")
            point = point[0]
        return point


    def get_score(self, log: LogEvent, core_set_size = 2):
        embedding = get_sentence_embedding(log)[0]
        core_set = self.client.search(
            collection_name="normal",
            query_vector=embedding[0].numpy(),
            limit=core_set_size,
            with_payload=["tokens"],
            with_vectors=False,
        )
        anomaly_score = min(
            distance(embedding, torch.tensor(c.payload["tokens"])) for c in core_set
        )
        return anomaly_score


    def log_exists_in_normal_col(self, hash: int):
        return (
            self.client.retrieve(
                collection_name="normal", ids=[hash], with_payload=False, with_vectors=False
            )
            != []
        )


    def get_point_from_test_col(self, hash: int):
        return self.client.retrieve(
            collection_name="test", ids=[hash], with_payload=True, with_vectors=True
        )

    # Entrypoint
    def run_inference(self, test_log: LogEvent):
        if self.log_exists_in_normal_col(test_log.hash):
            print("Log exists in normal db!")
            return
        point = self.get_embedding_test(test_log)
        if not is_anomaly or self.config.mode == "train":
            print("Adding to normal db..")
            self.client.upsert(
                collection_name="normal", points=[point]
            )  # NOTE: will exist in both test and normal collection (optimize later?)

        score = self.get_score(test_log)
        print(f"Score: {score}")
        is_anomaly = score < self.config.threshold
        print(f"Is anomaly: {is_anomaly}")

        return score, is_anomaly

# BGL TESTING
# def experiment_bgl():
#     bgl = pd.read_csv("../data/bgl/bgl_cleaned.csv", compression="gzip").drop(columns=["Unnamed: 0"])
#     unique_normal_logs = bgl[bgl["is_anomaly"] == False]["line"].unique()
#     for text in tqdm(unique_normal_logs):
#         add_log_to_collection(text, 'normal')
#     test_data = pd.concat( [ bgl[bgl["is_anomaly"] == False].sample(879_910), bgl[bgl["is_anomaly"] == True].sample(348_460) ])
#     unique_test_logs = test_data["line"].unique()
#     scores = [get_score(log).item() for log in tqdm(unique_test_logs)]
#     unique_scores = pd.DataFrame({"line": unique_test_logs, "score": scores})
#     test_data = test_data.merge(unique_scores, on="line", how="left")
#     evaluate(test_data["is_anomaly"], test_data["score"].values)