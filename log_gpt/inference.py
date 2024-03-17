from torch.distributions.categorical import Categorical
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from log_gpt.config import *
from log_gpt.preprocess import LogDataset


def evaluate_topk(val_df, model):
    print('Beginning evaluation...')
    val_dataset = LogDataset(val_df["line"])
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16) # experiment with this batch size
    ground_truths = val_df["is_anomaly"]

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, "Batch: "):
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(**batch).logits
            for batch_num, labels in enumerate(batch["input_ids"]):
                    is_anomaly = False
                    dist = Categorical(logits=logits[batch_num])
                    for token_id in range(len(labels) - 1):
                        softmax_top = set(torch.topk(dist.logits[token_id], top_k).indices) # TODO: extract this top k 
                        ground_truth = labels[token_id + 1]
                        if ground_truth not in softmax_top:
                            is_anomaly = True
                            break
                    predictions.append(is_anomaly)

    print(classification_report(ground_truths, predictions))
    return predictions