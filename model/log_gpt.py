from math import floor
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config
import torch
from torch.distributions import Categorical

from models import ServiceConfig

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

num_episodes = 10
num_epochs = 10

loss_improv_epsilon = 0.01

num_special_tokens = 3
bos_token_id = 0
eos_token_id = 1
pad_token_id = 2

num_beams = 5
cut = 0.8

vocab_size = 195


class LogGPTInferenceAPI:
    def __init__(self, service: str, config: ServiceConfig, cache_path=Path("cache")):
        self.service = service
        self.config = config
        self.cache_path = cache_path
        self.model_output_path = self.cache_path / "model.pt"
        self.setup_model_optimizer(vocab_size, self.cache_path)

    def reload_config(self, config: ServiceConfig):
        self.config = config

    def run_inference(self, val_dataloader):
        print("Beginning evaluation...")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, "Batch: "):
                batch = {k: v.cuda() for k, v in batch.items()}
                logits = self.model(**batch).logits
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
        return predictions

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

        self.model = GPT2LMHeadModel(configuration).cuda()
        self.optimizer = AdamW(self.model.parameters(), lr=lr_pretraining)
        if self.model_output_path and self.model_output_path.exists():
            self.load_model(self.model_output_path)

    def pretrain_model(self, train_dataloader):
        print("Beginning model pre-training...")
        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Epoch: "):
            train_loss = 0
            for batch in tqdm(train_dataloader, desc="Training: "):
                self.optimizer.zero_grad()
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

            train_loss /= len(train_dataloader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(batch)}"
            )

            print("Saving model...")
            self.save_model(self.model_output_path)

    def save_model(self, saved_model_path):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_path)

    def load_model(self, saved_model_path):
        saved_model = torch.load(saved_model_path)
        self.model.load_state_dict(saved_model["state_dict"], strict=False)
        self.optimizer.load_state_dict(saved_model["optimizer"])

    def compute_loss(self, input_ids, logits, start_gen_pos):
        reward = log_prob = 0
        for i in range(start_gen_pos - 1, len(input_ids) - 1):
            softmax = Categorical(logits=logits[i])
            next_token_id = input_ids[i + 1]
            log_prob += softmax.log_prob(next_token_id.clone())
            top_next_tokens = torch.topk(logits[i], k=top_k).indices
            reward += 1 if next_token_id in top_next_tokens else -1
        cost = -reward * log_prob
        return cost

    def step(self, sequence):
        start_gen_pos = floor(cut * len(sequence))
        if start_gen_pos < 2:
            input_ids = sequence[:2].unsqueeze(0)
        elif start_gen_pos >= len(sequence) - 2:
            input_ids = sequence[:-2].unsqueeze(0)
        else:
            input_ids = sequence[:start_gen_pos].unsqueeze(0)
        prompt = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != pad_token_id).float(),
        }
        print(prompt)
        gen_seq = self.model.generate(
            **prompt,
            min_length=len(sequence),
            max_length=len(sequence),
            num_beams=num_beams,
            early_stopping=False,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        logits = self.model(gen_seq).logits[0]
        loss = self.compute_loss(sequence, logits, start_gen_pos)
        return loss

    def finetune(self, train_set):
        print("Beginning model finetuning...")
        last_episode_loss = 0
        count = 0
        best_loss = float("inf")
        self.model.train()
        for episode in tqdm(range(num_episodes), "Episode: "):
            num_rows = train_set.size(0)
            permuted_indices = torch.randperm(num_rows)
            finetune_trainset = train_set[permuted_indices]

            train_loss = 0
            for sequence in tqdm(finetune_trainset, "Finetuning: "):
                train_loss += self.step(sequence)
            train_loss /= len(finetune_trainset)

            print(f"Episode {episode}/{num_episodes} (finetune): {train_loss}")
            if last_episode_loss - train_loss <= loss_improv_epsilon:
                print(f"Loss barely changed {last_episode_loss} to {train_loss}")
                count += 1
            if count > 3:
                print("Early stop!")
                break

            if train_loss < best_loss or episode == 0:
                print("New best model! Saving...")
                self.save_model(self.model_output_path)
                best_loss = train_loss

            last_episode_loss = train_loss
