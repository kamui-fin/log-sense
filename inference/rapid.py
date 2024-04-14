import logging
from typing import Union
import pandas as pd
from kafka import KafkaProducer
import qdrant_client
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
from . import models

from inference.models import GlobalConfig, RapidLogEvent


def distance(query_emb, train_embedding):
    dot_products = query_emb @ train_embedding.T
    query_normalized = query_emb.square().sum(dim=-1).sqrt()
    train_normalized = train_embedding.square().sum(dim=-1).sqrt()
    normalized_pairs = query_normalized.unsqueeze(0).T * train_normalized.unsqueeze(0)
    max_similarity = (dot_products / normalized_pairs).max(dim=-1).values.sum(dim=-1)
    return 1 - max_similarity


class RapidInferenceAPI:
    def __init__(self, client, service, config):
        self.client = client
        self.service = service
        self.config = config
        self.model = BertModel.from_pretrained("bert-base-uncased").to("cuda:0")

        self.normal_collection_name = f"normal-{self.service}"
        self.init_qdrant()

    def reload_config(self, config: GlobalConfig):
        self.config = config

    def init_qdrant(self):
        for collection_name in [self.normal_collection_name, "test"]:
            if not self.client.collection_exists(collection_name):
                logging.info(f'Creating collection "{collection_name}"')
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.model.config.hidden_size, distance=Distance.COSINE
                    ),
                )

    def get_sentence_embedding(self, log: RapidLogEvent):
        logging.info(f"Getting embedding..")
        # map each list within log.tokens into a tensor and move to cuda
        for key, value in log.tokens.items():
            if isinstance(value, list) or value.device.type == "cpu":
                log.tokens[key] = torch.tensor(value).unsqueeze(0).to("cuda:0")
        output = self.model(**log.tokens)
        embeddings = output.last_hidden_state
        # we must deallocate from gpu mem!
        for key, value in log.tokens.items():
            log.tokens[key] = value.detach().cpu().squeeze(0).tolist()
        return embeddings.detach().cpu()  # use first for [CLS]

    def generate_point(self, log_event: RapidLogEvent):
        embedding = self.get_sentence_embedding(log_event)
        cls_embedding = embedding[0, 0].numpy()
        token_embeddings = embedding[0, 1:].numpy().tolist()
        point = PointStruct(
            id=int(log_event.hash),
            vector=cls_embedding,
            payload={"tokens": token_embeddings},
        )
        return point

    def add_log_to_collection(self, log: RapidLogEvent, collection):
        logging.info(f'Adding log "{log}" to collection {collection}..')
        point = self.generate_point(log)
        self.client.upsert(collection_name=collection, points=[point])
        return point

    def get_embedding_test(self, test_log: RapidLogEvent):
        point = self.get_point_from_test_col(test_log.hash)
        if not point:
            logging.info(
                f"Could not find log in test collection. Generating and adding..."
            )
            point = self.add_log_to_collection(test_log, "test")
        else:
            logging.info("Fetching embedding from cache!")
        return point

    def get_score(self, log: RapidLogEvent):
        logging.info("Scoring log..")
        embedding = self.get_sentence_embedding(log)[0]
        logging.info(f"Generating coreset..")
        core_set = self.client.search(
            collection_name=self.normal_collection_name,
            query_vector=embedding[0].numpy(),
            limit=self.config.configs[self.service].coreset_size,
            with_payload=["cleaned_text", "tokens"],
            with_vectors=False,
        )
        # logging.info(f'Found coreset: {" ".join([c.payload["text"] for c in core_set])}')
        logging.info(f"Found core set of length {len(core_set)}")
        logging.info("Calculating anomaly score..")
        anomaly_score = min(
            distance(embedding, torch.tensor(c.payload["tokens"])) for c in core_set
        )
        return anomaly_score

    def log_exists_in_normal_col(self, hash: str):
        return (
            self.client.retrieve(
                collection_name=self.normal_collection_name,
                ids=[int(hash)],
                with_payload=False,
                with_vectors=False,
            )
            != []
        )

    def get_point_from_test_col(self, hash: str):
        record = self.client.retrieve(
            collection_name="test",
            ids=[int(hash)],
            with_payload=True,
            with_vectors=True,
        )
        if record:
            record = record[0]
            point = PointStruct(
                id=record.id,
                vector=record.vector,
                payload=record.payload,
            )
            return point

    def mark_normal(self, test_log: RapidLogEvent):
        point = self.get_embedding_test(test_log)
        self.client.upsert(collection_name="normal", points=[point])

    # Entrypoint
    def run_inference(self, test_log: RapidLogEvent):
        logging.info('Running inference on "{test_log}"..')
        if self.log_exists_in_normal_col(test_log.hash):
            logging.info("Log already exists in normal db!")
            return
        point = self.get_embedding_test(test_log)
        if self.config.configs[self.service].is_train:
            logging.info("Adding to normal db..")
            self.client.upsert(collection_name="normal", points=[point])
            return

        score = self.get_score(test_log)
        logging.info(f"Score: {score}")
        is_anomaly = score > self.config.configs[self.service].threshold
        logging.info(f"Is anomaly: {is_anomaly}")

        if not is_anomaly:
            logging.info("Adding to normal db..")
            self.client.upsert(
                collection_name="normal", points=[point]
            )  # NOTE: will exist in both test and normal collection (optimize later?)

        return score.item(), is_anomaly.item()


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
