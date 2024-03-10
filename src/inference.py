import torch
import pandas as pd
import numpy as np
import pathlib
import re
from typing import List
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertForMaskedLM, AutoTokenizer

from src.config import HDFS_UNCLEANED_PATH, LAST_CHECKPOINT, MODEL_CHECKPOINTS
from src.preprocess import regex_clean

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(MODEL_CHECKPOINTS / LAST_CHECKPOINT)

# Caching
score_cache = {}

def top_k(arr: List[int], k: int, strategy = 'max'):
    # FIXME: potential overhead with converting to tensor and back?
    return torch.topk(torch.tensor(arr), k, dim = -1, largest = strategy == 'max').values.mean().item()


def calculate_abnormal_scores(logs: List[str]):
    """
    Inputs
    --------
    logs: List of log sequences
    s_len: sequence length of every log
    model: BERT model

    Outputs
    -------
    List of anomaly loss and probability score pairs for every log seq
    """

    # TODO: refactor and comment

    K = 5
    scores = []
    for log in logs:
        log = regex_clean(log)

        if log not in score_cache:
            log_tokens = tokenizer([log], return_tensors='pt')
            input_ids = log_tokens['input_ids']
            batch_labels, batch_tokens = [], []

            # Collecting masked sequences into a batch for parallelism
            for mask_token_index in range(1, log_tokens['input_ids'].shape[1] - 1):
                labels = input_ids.detach().clone()
                curr_mask_seq = input_ids.detach().clone()
                curr_mask_seq[0, mask_token_index] = tokenizer.mask_token_id
                labels[curr_mask_seq != tokenizer.mask_token_id] = -100

                batch_tokens.append(curr_mask_seq)
                batch_labels.append(labels)

            s_len = log_tokens['input_ids'].shape[1] # number of tokens

            # Spliting up and inferencing in batches for speed-up
            batch_size = 50
            batch_tokens = torch.concat(batch_tokens).split(batch_size)
            batch_labels = torch.concat(batch_labels).split(batch_size)

            all_probs, all_losses = [], []
            for current_batch in range(len(batch_tokens)):
                curr_label_batch = batch_labels[current_batch]
                curr_batch_size = curr_label_batch.shape[0]

                attention_mask = torch.ones((curr_batch_size, s_len), dtype=torch.long)
                token_type_ids = torch.zeros(curr_batch_size, s_len, dtype=torch.long)

                output = model(input_ids=batch_tokens[current_batch], attention_mask=attention_mask, token_type_ids=token_type_ids, labels = curr_label_batch)
                logits = output.logits

                # reduction=none to avoid agg loss, we want the loss per-sample
                loss_function = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_function(logits.view(-1, tokenizer.vocab_size), curr_label_batch.view(-1)).view(logits.size(0), -1).sum(dim=1)

                probabilities = F.softmax(logits, dim = -1)
                values, indices = torch.max(curr_label_batch, dim = 1)
                prob = probabilities[torch.arange(indices.size(0)), indices, values]

                all_probs.extend(prob.tolist())
                all_losses.extend(loss.tolist())

            abnormal_loss = top_k(all_losses, K)
            abnormal_prob = top_k(all_probs, K, 'min')
            score_cache[log] = (abnormal_loss, abnormal_prob)
        else:
            abnormal_loss, abnormal_prob = score_cache[log]
        
        scores.append((abnormal_loss, abnormal_prob))

    return scores

# TODO: Dataset classes for more generic testing

raw_hdfs_logs = pathlib.Path(HDFS_UNCLEANED_PATH).read_text().splitlines()

true_labels = pd.read_csv('./data/preprocessed/anomaly_label.csv')
true_labels['Label'] = (true_labels['Label'] == 'Anomaly').astype('int64')

test_logs = raw_hdfs_logs[5000:5500] # Random chunk to test out inference
scores = np.array(calculate_abnormal_scores(test_logs))

# Place into dataframe to examine results
eval_df = pd.DataFrame()
eval_df['block_id'] = [re.findall('blk_-?\d+', l)[0] for l in test_logs]
eval_df['seq'] = test_logs
eval_df['loss'] = scores[:, 0]
eval_df['prob'] = scores[:, 1]
eval_df = pd.merge(eval_df, true_labels, left_on='block_id', right_on='BlockId').drop(columns=['BlockId'])
eval_df = eval_df.rename(columns={'Label': 'is_anomaly'})

# TODO: Choose best threshold and calculate F1