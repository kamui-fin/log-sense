import gc
import math
import torch
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from typing import List
from sklearn.metrics import classification_report
from tqdm import tqdm
from src.dataset import Hdfs

## Load data and create train-test split
hdfs = Hdfs(Path('../../data/hdfsv1.log'), Path('../../data/preprocessed/anomaly_label.csv'))
train_X, test_X, test_y = hdfs.train_test_split()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to('cuda:0')

def clear_hw_cache():
    del model
    gc.collect()
    torch.cuda.empty_cache()

def yield_batches(data, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def get_sentence_embedding(text):
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to('cuda:0')
    output = model(**encoded_input)
    embeddings = output.last_hidden_state
    del encoded_input
    return embeddings.detach().cpu() # use first for [CLS]

def build_db(log_sequences: List[str]):
    unique_log_to_index = { l: i for i, l in enumerate(list(set(log_sequences))) }
    num_unique_logs = len(unique_log_to_index)
    index_to_unique_log = {i: l for l, i in unique_log_to_index.items()}
    timestamp_to_unique_index = {i: unique_log_to_index[line] for i, line in enumerate(log_sequences)}

    batch_size = 5
    log_batches = yield_batches([index_to_unique_log[i] for i in range(num_unique_logs)], batch_size)

    embedding_db = []
    idx = iter(range(num_unique_logs))
    for batch in tqdm(log_batches, total = math.ceil(num_unique_logs / batch_size)):
        embedding_batch = get_sentence_embedding(batch)
        embedding_db += [(next(idx), emb) for emb in embedding_batch]

    embedding_db = dict(embedding_db)
    return embedding_db, timestamp_to_unique_index, index_to_unique_log

def distance(query_emb, train_embedding):
    dot_products = query_emb @ train_embedding.T
    query_normalized = query_emb.square().sum(dim = -1).sqrt()
    train_normalized = train_embedding.square().sum(dim = -1).sqrt()
    normalized_pairs = query_normalized.unsqueeze(0).T * train_normalized.unsqueeze(0)
    max_similarity = (dot_products / normalized_pairs).max(dim=-1).values.sum(dim=-1)
    return 1 - max_similarity

def get_core_set(query, knn):
    return knn.kneighbors(query.detach().numpy().reshape(1, -1))[-1][0]

train_log_seqs = train_X.to_list()
normal_embedding_db, normal_lookup_db, normal_index_to_unique_log = build_db(train_log_seqs)

test_log_seqs = test_X.to_list()
test_embedding_db, test_lookup_db, _ = build_db(test_log_seqs)
test_num_unique = len(test_embedding_db)

## INFERENCE
num_neighbors = round(0.01 * len(normal_index_to_unique_log)) # or simply 2
k_nearest_neighbors = NearestNeighbors(n_neighbors=num_neighbors)
k_nearest_neighbors.fit([normal_embedding_db[index][0].detach().numpy() for index in range(len(normal_embedding_db))])

# TODO: save database and knn to disk

abnormal_scores = [0] * test_num_unique
for index in range(test_num_unique):
    core_set = get_core_set(test_embedding_db[index][0])
    abnormal_scores[index] = min(distance(test_embedding_db[index], normal_embedding_db[c_index]) for c_index in core_set[1:])

final_pred = [0] * len(test_lookup_db)
for orig, uniq in test_lookup_db.items():
    final_pred[orig] = abnormal_scores[uniq].item()
final_pred = pd.Series(final_pred)

## ANALYSIS
precision, recall, thresholds = precision_recall_curve(test_y, final_pred)
f_score = (2 * precision * recall) / (precision + recall)
ix = np.argmax(f_score)
print(f'Best Threshold = {thresholds[ix]}')
print(classification_report(test_y, final_pred > thresholds[ix], target_names=['Normal', 'Anomaly']))

f1 = f1_score(test_y, final_pred > thresholds[ix])
print(f'F1 = {f1}')

auc = roc_auc_score(test_y, final_pred)
print(f'AUC = {auc}')