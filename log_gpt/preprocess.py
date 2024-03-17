import ast
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dataset import Hdfs
from torch.utils.data import Dataset
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner, TemplateMinerConfig
import torch
import torch.nn.functional as F
from log_gpt.config import *

import sys

sys.path.insert(0, "..")

persistence = FilePersistence("drain3_state.bin")
config = TemplateMinerConfig()
config.profiling_enabled = True
template_miner = TemplateMiner(persistence_handler=persistence, config=config)

class LogDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.cat((torch.tensor([bos_token_id]), torch.tensor(self.data.iloc[idx]), torch.tensor([eos_token_id])))

        if len(input_ids) > context_size:
            input_ids = torch.cat(
                (
                    input_ids.narrow(-1, 0, context_size - 1),
                    torch.tensor([eos_token_id]),
                ),
                dim=-1,
            )
        else:
            input_ids = F.pad(
                input_ids, (0, context_size - input_ids.shape[0]), value=pad_token_id
            )

        attention_mask = (input_ids != pad_token_id).float()

        return {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
        }

def drain_cluster(log_lines):
    for log_line in tqdm(log_lines):
        template_miner.add_log_message(log_line)

    num_clusters = len(template_miner.drain.clusters)
    print(f"Found {num_clusters} clusters!")
    sorted_clusters = sorted(
        template_miner.drain.clusters, key=lambda it: it.size, reverse=True
    )
    print('Top 5 clusters:')
    for cluster in sorted_clusters[:5]:
        print(cluster)
    return num_clusters

def load_hdfs(hdfs_data_dir, cache_file: Path):
    print('Loading HDFS dataset...')
    if cache_file.exists():
        df = pd.read_csv(cache_file, compression="gzip", converters={'line': ast.literal_eval })
        vocab_size = df['line'].apply(max).max()
        return df, vocab_size

    hdfs = Hdfs(
        hdfs_data_dir / "hdfsv1.log",
        hdfs_data_dir / "anomaly_label.csv",
        hdfs_data_dir / "cache_no_regex.csv",
        True,
    )
    df = hdfs.df.drop(columns=["BlockId"]) # TODO: refactor
    num_clusters = drain_cluster(df['line'])
    # Concatenate log keys per block
    grouped_log_keys = df.groupby("block_id")["line"].apply(
        lambda logs: [template_miner.match(log).cluster_id + num_special_tokens for log in logs]
    )
    # Add ground truths to each block
    grouped_anomaly_labels = df.groupby("block_id")["is_anomaly"].first()
    df = pd.merge(
        grouped_anomaly_labels, grouped_log_keys, left_index=True, right_index=True
    )

    df.to_csv(cache_file, compression="gzip")
    return df, num_clusters + num_special_tokens

def train_test_split(df):
    print('Creating train test split...')
    normal_df = df[df["is_anomaly"] == 0].sample(frac=1)
    abnormal_df = df[df['is_anomaly'] == 1].sample(frac=1)
    train_df = normal_df[:train_samples]
    test_df = pd.concat([normal_df[train_samples:], abnormal_df])
    return train_df, test_df

def train_val_split(train_df, torch_dataset=False):
    # We split a train_df further, usually with a 0.9 ratio for quick validation within training loops
    idx = int(train_val_ratio * len(train_df))
    train_set = train_df[:idx]
    val_set = train_df[idx:]
    if torch_dataset:
        train_set = LogDataset(train_set["line"])
        val_set = LogDataset(val_set["line"])
    return train_set, val_set