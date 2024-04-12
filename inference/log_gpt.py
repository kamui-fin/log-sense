from math import floor
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config
import torch
from torch.distributions import Categorical

from inference.models import GlobalConfig

model_output_dir = Path("../../output/log_gpt")

embed_dim = 60
layers = 6
heads = 6

context_size = 256
lr_pretraining = 1e-4
lr_finetuning = 1e-6
batch_size = 16

# HDFS: 7
# BGL: 80
# TB: 690
top_k = 80

num_episodes = 1
num_epochs = 10

loss_improv_epsilon = 0.01

num_special_tokens = 3
bos_token_id = 0
eos_token_id = 1
pad_token_id = 2

num_beams = 5
cut = 0.2

# TODO: We need vocab_size


class LogGPTInferenceAPI:
    def __init__(self, client, service, config):
        self.client = client
        self.service = service
        self.config = config
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda:0")

    def reload_config(self, config: GlobalConfig):
        self.config = config

    def predict_topk(self, val_df, model, top_k=top_k, sliding_window=False):
        print("Beginning evaluation...")
        val_dataset = LogDataset(val_df["line"])
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
                                for i in torch.topk(
                                    dist.logits[token_id], top_k
                                ).indices
                            ]
                        )
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

    def run_inference(self, log_batch):
        pass

    def get_data_loaders(self, train_df):
        train_dataset, val_dataset = train_val_split(train_df, torch_dataset=True)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size
        )
        return train_dataloader, val_dataloader

    def setup_model_optimizer(self, vocab_size, cache_path=None):
        configuration = GPT2Config(
            n_layer=layers,
            n_head=heads,
            n_embd=embed_dim,
            vocab_size=vocab_size + 1,  # for padding token
            n_positions=context_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        model = GPT2LMHeadModel(configuration).cuda()
        optimizer = AdamW(model.parameters(), lr=lr_pretraining)
        if cache_path and cache_path.exists():
            self.load_model(model, optimizer, cache_path)
        return model, optimizer

    def pretrain_model(self, train_df, vocab_size, output_path):
        print("Beginning model pre-training...")
        model, optimizer = self.setup_model_optimizer(
            vocab_size, output_path
        )  # resume from last trained if possible
        train_dataloader, val_dataloader = self.get_data_loaders(train_df)
        J = []
        for epoch in tqdm(range(num_epochs), desc="Epoch: "):
            model.train()
            train_loss = 0
            for batch in tqdm(train_dataloader, desc="Training: "):
                optimizer.zero_grad()
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            train_loss /= len(val_dataloader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(batch)}"
            )

            model.eval()
            val_loss = 0
            for batch in tqdm(val_dataloader, desc="Evaluating: "):
                with torch.no_grad():
                    batch = {k: v.cuda() for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()

            val_loss /= len(val_dataloader)
            J.append((train_loss, val_loss))
            print(
                f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(batch)}"
            )
            print("Saving model...")
            self.save_model(model, optimizer, output_path)

        return model, optimizer, J

    def save_model(self, model, optimizer, saved_model_path):
        state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, saved_model_path)

    def load_model(self, model, optimizer, saved_model_path):
        saved_model = torch.load(saved_model_path)
        model.load_state_dict(saved_model["state_dict"], strict=False)
        optimizer.load_state_dict(saved_model["optimizer"])

    def compute_loss(self, input_ids, logits, start_gen_pos):
        reward = log_prob = 0
        for i in range(start_gen_pos - 1, len(input_ids) - 1):
            softmax = Categorical(logits=logits[i])
            next_token_id = input_ids[i + 1]
            log_prob += softmax.log_prob(torch.tensor(next_token_id).cuda())
            top_next_tokens = torch.topk(logits[i], k=top_k).indices
            reward += 1 if next_token_id in top_next_tokens else -1
        cost = -reward * log_prob
        return cost

    def step(self, model, optimizer, sequence, val=False, sliding_window=False):
        if not sliding_window:
            sequence = [bos_token_id] + sequence + [eos_token_id]
        start_gen_pos = floor(cut * len(sequence))
        if sliding_window:
            if start_gen_pos < 1:
                input_ids = torch.tensor([sequence[:1]])
            elif start_gen_pos >= len(sequence) - 1:
                input_ids = torch.tensor([sequence[:-1]])
            else:
                input_ids = torch.tensor([sequence[:start_gen_pos]])
        else:
            if start_gen_pos < 2:
                input_ids = torch.tensor([sequence[:2]])
            elif start_gen_pos >= len(sequence) - 2:
                input_ids = torch.tensor([sequence[:-2]])
            else:
                input_ids = torch.tensor([sequence[:start_gen_pos]])
        input_ids = input_ids.cuda()
        prompt = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != pad_token_id).float(),
        }
        gen_seq = model.generate(
            **prompt,
            min_length=len(sequence),
            max_length=len(sequence),
            num_beams=num_beams,
            early_stopping=False,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        logits = model(gen_seq).logits[0]

        loss = self.compute_loss(sequence, logits, start_gen_pos)
        if not val:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def finetune(
        self, model, optimizer, train_normal_df, cache_path=None, sliding_window=False
    ):
        print("Beginning model finetuning...")
        last_episode_loss = 0
        count = 0

        best_loss = float("inf")

        train_set, val_set = train_val_split(train_normal_df)

        J = []
        for episode in tqdm(range(num_episodes), "Episode: "):
            model.train()
            finetune_trainset = train_set["line"].sample(frac=1)
            train_loss = 0
            for sequence in tqdm(finetune_trainset, "Finetuning: "):
                train_loss += self.step(
                    model, optimizer, sequence, sliding_window=sliding_window
                )
            train_loss /= len(finetune_trainset)

            print(f"Episode {episode}/{num_episodes} (finetune): {train_loss}")

            model.eval()
            val_loss = 0
            for sequence in tqdm(val_set["line"], "Evaluating: "):
                with torch.no_grad():
                    val_loss += self.step(
                        model,
                        optimizer,
                        sequence,
                        val=True,
                        sliding_window=sliding_window,
                    )
            val_loss /= len(val_set)

            if last_episode_loss - val_loss <= loss_improv_epsilon:
                print(f"Loss barely changed {last_episode_loss} to {val_loss}")
                count += 1

            if count > 3:
                print("Early stop!")
                break

            if val_loss < best_loss or episode == 0:
                print("New best model! Saving...")
                self.save_model(model, optimizer, cache_path)
                best_loss = val_loss

            print(f"Episode {episode}/{num_episodes} (eval): {val_loss}")
            last_episode_loss = val_loss
            J.append((train_loss, val_loss))
        return J
