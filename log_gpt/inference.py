from torch.distributions.categorical import Categorical
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from log_gpt.config import *
from log_gpt.preprocess import LogDataset


# Top K prediction
# TODO: speed up
def predict(df, model):
    normal_df = df[df["is_anomaly"] == 0]
    abnormal_df = df[df["is_anomaly"] == 1]
    val_df = pd.concat([normal_df[train_samples:], abnormal_df])
    val_dataset = LogDataset(val_df["line"])
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16) # experiment with this batch size
    ground_truths = val_df["is_anomaly"]

    model.eval()
    predictions = []
    for batch in tqdm(val_dataloader, "Batch: "):
        batch = {k: v.cuda() for k, v in batch.items()}
        logits = model(**batch).logits
        for batch_num, labels in enumerate(batch["input_ids"]):
            with torch.no_grad():
                is_anomaly = False
                dist = Categorical(logits=logits[batch_num])
                for token_id in range(len(labels) - 1):
                    softmax_top = set(torch.topk(dist.logits[token_id], 5).indices) # TODO: extract this top k 
                    ground_truth = labels[token_id + 1]
                    if ground_truth not in softmax_top:
                        is_anomaly = True
                        break
                predictions.append(is_anomaly)

    print(classification_report(ground_truths, predictions))
