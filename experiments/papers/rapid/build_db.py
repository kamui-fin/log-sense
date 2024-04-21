from dataclasses import dataclass
from pathlib import Path
import torch
import math
from transformers import BertTokenizer, BertModel
from typing import List
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from collections.abc import Mapping
import pickle

from rapid.utils import yield_batches

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to("cuda:0")


def save_dict_as_pickle(file_path, dictionary):
    with open(file_path, "wb") as f:
        pickle.dump(dictionary, f)


def load_dict_from_pickle(file_path):
    with open(file_path, "rb") as f:
        dictionary = pickle.load(f)
    return dictionary


@dataclass
class RapidDB:
    num_unique_logs: int  # more readable over len(embedding_db)
    # Maps a unique log seq index to its embedding
    # Tensor of shape (num_embeddings, max_seq_len, embed_dim)
    embedding_db: torch.Tensor
    # Maps a unique log seq index to its actual text
    index_to_unique_log: Mapping[int, str]
    # Maps a timestamp to the appropriate unique log seq index
    lookup_db: Mapping[int, int]


class DBCache:
    def __init__(self, output_dir, key):
        self.embedding_db_path = output_dir / f"{key}-embedding_db.pt"
        self.index_to_unique_path = output_dir / f"{key}-index_to_unique.pkl"
        self.lookup_db_path = output_dir / f"{key}-lookup_db.pkl"

    def cache_exists(self):
        return all(
            p.exists()
            for p in [
                self.embedding_db_path,
                self.index_to_unique_path,
                self.lookup_db_path,
            ]
        )

    def load_if_exists(self):
        if self.cache_exists():
            embedding_db = torch.load(self.embedding_db_path)
            return RapidDB(
                embedding_db=embedding_db,
                index_to_unique_log=load_dict_from_pickle(self.index_to_unique_path),
                lookup_db=load_dict_from_pickle(self.lookup_db_path),
                num_unique_logs=len(embedding_db),
            )

    def save(self, db: RapidDB):
        torch.save(db.embedding_db, self.embedding_db_path)
        save_dict_as_pickle(self.index_to_unique_path, db.index_to_unique_log)
        save_dict_as_pickle(self.lookup_db_path, db.lookup_db)


@dataclass
class RapidKNN:
    k_nearest_neighbors: NearestNeighbors


class KNNCache:
    def __init__(self, output_dir):
        self.knn_path = output_dir / f"knn.pkl"

    def load_if_exists(self):
        if self.knn_path.exists():
            return RapidKNN(k_nearest_neighbors=load_dict_from_pickle(self.knn_path))

    def save(self, knn: RapidKNN):
        save_dict_as_pickle(self.knn_path, knn.k_nearest_neighbors)


def get_sentence_embedding(text):
    encoded_input = tokenizer(
        text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda:0")
    output = model(**encoded_input)
    embeddings = output.last_hidden_state
    # we must deallocate from gpu mem!
    del encoded_input
    return embeddings.detach().cpu()  # use first for [CLS]


def build_db(log_sequences: List[str], db_key: str, output_dir: Path):
    cache = DBCache(output_dir, db_key)
    if db := cache.load_if_exists():
        print(f"Cache found! Pulling {db_key} DB from cache...")
        return db

    print(f"Building {db_key} database...")
    unique_log_to_index = {l: i for i, l in enumerate(list(set(log_sequences)))}
    num_unique_logs = len(unique_log_to_index)
    index_to_unique_log = {i: l for l, i in unique_log_to_index.items()}
    timestamp_to_unique_index = {
        i: unique_log_to_index[line] for i, line in enumerate(log_sequences)
    }

    batch_size = 10
    log_batches = yield_batches(
        [index_to_unique_log[i] for i in range(num_unique_logs)], batch_size
    )

    embedding_db = torch.zeros(
        (
            num_unique_logs,
            model.config.max_position_embeddings,
            model.config.hidden_size,
        )
    )
    idx = iter(range(num_unique_logs))
    for batch in tqdm(log_batches, total=math.ceil(num_unique_logs / batch_size)):
        embedding_batch = get_sentence_embedding(batch)
        for emb in embedding_batch:
            embedding_db[next(idx)] = emb

    db = RapidDB(
        num_unique_logs=num_unique_logs,
        embedding_db=embedding_db,
        lookup_db=timestamp_to_unique_index,
        index_to_unique_log=index_to_unique_log,
    )

    cache.save(db)

    return db


def distance(query_emb, train_embedding):
    dot_products = query_emb @ train_embedding.T
    query_normalized = query_emb.square().sum(dim=-1).sqrt()
    train_normalized = train_embedding.square().sum(dim=-1).sqrt()
    normalized_pairs = query_normalized.unsqueeze(0).T * train_normalized.unsqueeze(0)
    max_similarity = (dot_products / normalized_pairs).max(dim=-1).values.sum(dim=-1)
    return 1 - max_similarity


def get_core_set(query, knn):
    return knn.k_nearest_neighbors.kneighbors(query.detach().numpy().reshape(1, -1))[
        -1
    ][0]


def build_knn(db, output_dir):
    cache = KNNCache(output_dir)
    if knn := cache.load_if_exists():
        print("Cache found! Pulling normal KNN from cache...")
        return knn

    print("Building KNN...")
    # Build KNN
    num_neighbors = max(2, round(0.01 * len(db.index_to_unique_log)))
    k_nearest_neighbors = NearestNeighbors(n_neighbors=num_neighbors)
    k_nearest_neighbors.fit(
        [
            db.embedding_db[index][0].detach().numpy()
            for index in range(len(db.embedding_db))
        ]
    )

    knn = RapidKNN(k_nearest_neighbors)
    cache.save(knn)
    return knn


def build_normal_db(train_X, output_dir):
    # Build DB
    train_log_seqs = train_X.to_list()
    db = build_db(train_log_seqs, "normal", output_dir)
    knn = build_knn(db, output_dir)
    return db, knn
