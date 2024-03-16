import torch
from transformers import GPT2Config, GPT2LMHeadModel, AdamW
from tqdm import tqdm

from src.log_gpt.preprocess import LogDataset
from src.log_gpt.config import *

def get_data_loaders(df):
    normal_df = df[df["is_anomaly"] == 0]
    abnormal_df = df[df["is_anomaly"] == 1]
    train_normal_df = normal_df.iloc[:5000]
    val_df = normal_df.iloc[5000:8000]

    train_dataset = LogDataset(train_normal_df["line"])
    val_dataset = LogDataset(val_df["line"])

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader

def setup_model_optimizer(vocab_size):
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
    return model, optimizer

def pretrain_model(model, optimizer, train_dataloader, val_dataloader):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(batch)}")

        model.eval()
        val_loss = 0
        for batch in tqdm(val_dataloader, desc="Evaluating: "):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(batch)}")


def save_model(model, optimizer):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, "model.pt")


def load_model(model, optimizer, saved_model_path):
    saved_model = torch.load(saved_model_path)
    model.load_state_dict(saved_model["state_dict"], strict=False)
    optimizer.load_state_dict(saved_model["optimizer"])
