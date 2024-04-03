import torch
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from tqdm import tqdm

from log_gpt.preprocess import train_val_split
from log_gpt.config import *


def get_data_loaders(train_df):
    train_dataset, val_dataset = train_val_split(train_df, torch_dataset=True)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size
    )
    return train_dataloader, val_dataloader


def setup_model_optimizer(vocab_size, cache_path=None):
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
        load_model(model, optimizer, cache_path)
    return model, optimizer


def pretrain_model(train_df, vocab_size, output_path):
    print("Beginning model pre-training...")
    model, optimizer = setup_model_optimizer(
        vocab_size, output_path
    )  # resume from last trained if possible
    train_dataloader, val_dataloader = get_data_loaders(train_df)
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
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(batch)}")

        model.eval()
        val_loss = 0
        for batch in tqdm(val_dataloader, desc="Evaluating: "):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        J.append((train_loss, val_loss))
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(batch)}")
        print("Saving model...")
        save_model(model, optimizer, output_path)

    return model, optimizer, J


def save_model(model, optimizer, saved_model_path):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, saved_model_path)


def load_model(model, optimizer, saved_model_path):
    saved_model = torch.load(saved_model_path)
    model.load_state_dict(saved_model["state_dict"], strict=False)
    optimizer.load_state_dict(saved_model["optimizer"])
