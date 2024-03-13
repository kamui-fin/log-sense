from dataclasses import dataclass
import math
from transformers import BertTokenizer, BertModel
from typing import List
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from src.rapid.utils import yield_batches

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to('cuda:0')

# TODO: implement save & load for data classes

@dataclass
class RapidDB:
    # Embedding DB
    num_unique_logs: int # more readable over len(embedding_db)
    embedding_db: dict
    lookup_db: dict
    index_to_unique_log: dict

@dataclass
class RapidKNN:
    k_nearest_neighbors: NearestNeighbors
    num_neighbors: int

def get_sentence_embedding(text):
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to('cuda:0')
    output = model(**encoded_input)
    embeddings = output.last_hidden_state
    # we must deallocate from gpu mem!
    del encoded_input 
    return embeddings.detach().cpu() # use first for [CLS]

def build_db(log_sequences: List[str]):
    unique_log_to_index = { l: i for i, l in enumerate(list(set(log_sequences))) }
    num_unique_logs = len(unique_log_to_index)
    index_to_unique_log = {i: l for l, i in unique_log_to_index.items()}
    timestamp_to_unique_index = {i: unique_log_to_index[line] for i, line in enumerate(log_sequences)}

    batch_size = 10
    log_batches = yield_batches([index_to_unique_log[i] for i in range(num_unique_logs)], batch_size)

    embedding_db = []
    idx = iter(range(num_unique_logs))
    for batch in tqdm(log_batches, total = math.ceil(num_unique_logs / batch_size)):
        embedding_batch = get_sentence_embedding(batch)
        embedding_db += [(next(idx), emb) for emb in embedding_batch]

    embedding_db = dict(embedding_db)

    db = RapidDB(
        num_unique_logs=num_unique_logs,
        embedding_db=embedding_db,
        lookup_db=timestamp_to_unique_index,
        index_to_unique_log=index_to_unique_log
    )

    return db

def distance(query_emb, train_embedding):
    dot_products = query_emb @ train_embedding.T
    query_normalized = query_emb.square().sum(dim = -1).sqrt()
    train_normalized = train_embedding.square().sum(dim = -1).sqrt()
    normalized_pairs = query_normalized.unsqueeze(0).T * train_normalized.unsqueeze(0)
    max_similarity = (dot_products / normalized_pairs).max(dim=-1).values.sum(dim=-1)
    return 1 - max_similarity

def get_core_set(query, knn):
    return knn.kneighbors(query.detach().numpy().reshape(1, -1))[-1][0]

def build_normal_db(train_X):
    # Build DB
    train_log_seqs = train_X.to_list()
    db = build_db(train_log_seqs)

    # Build KNN
    num_neighbors = round(0.01 * len(db.index_to_unique_log)) # or simply 2
    k_nearest_neighbors = NearestNeighbors(n_neighbors=num_neighbors)
    k_nearest_neighbors.fit([db.embedding_db[index][0].detach().numpy() for index in range(len(db.embedding_db))])

    knn = RapidKNN(k_nearest_neighbors=k_nearest_neighbors, num_neighbors=num_neighbors)
    
    return db, knn