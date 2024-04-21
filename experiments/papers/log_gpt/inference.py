from torch.distributions.categorical import Categorical
import torch
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from log_gpt.config import *
from log_gpt.preprocess import LogDataset


def predict_topk(val_df, model, top_k=top_k, sliding_window=False, trim=False):
    print("Beginning evaluation...")
    if trim:
        val_df = val_df.sample(100)
    val_dataset = LogDataset(val_df["line"], sliding_window=sliding_window)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=16
    )  # experiment with this batch size
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
                    softmax_top = set(
                        [
                            i.detach().cpu().item()
                            for i in torch.topk(dist.logits[token_id], top_k).indices
                        ]
                    )  # TODO: extract this top k
                    actual_token = labels[token_id + 1].detach().cpu().item()
                    if actual_token == pad_token_id:
                        break
                    if actual_token not in softmax_top:
                        is_anomaly = True
                        break
                predictions.append(is_anomaly)

    print(classification_report(ground_truths, predictions))
    print(confusion_matrix(y_true=ground_truths, y_pred=predictions))
    print(f"AUC ROC: {roc_auc_score(y_true=ground_truths, y_score=predictions)}")
    print(
        f"AUC PR: {average_precision_score(y_true=ground_truths, y_score=predictions)}"
    )
    return predictions
