import torch
from torch.distributions import Categorical
from math import floor
from tqdm import tqdm
from src.log_gpt.config import *
from src.log_gpt.pretrain import save_model


def compute_loss(input_ids, logits, t):
    reward = log_prob = 0
    for i in range(t, len(input_ids) - 1):
        softmax = Categorical(logits=logits[i])
        next_token_id = input_ids[i + 1]
        log_prob += softmax.log_prob(torch.tensor(next_token_id).cuda())
        top_next_tokens = torch.topk(logits[i], k=5).indices
        reward += 1 if next_token_id in top_next_tokens else -1
    cost = -reward * log_prob
    return cost


def step(model, optimizer, sequence, cut=0.2, val=False):
    t = floor(cut * len(sequence))
    input_ids = torch.tensor(sequence[:t]).unsqueeze(0).cuda()
    prompt = {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_token_id).float(),
    }
    gen_seq = model.generate(
        **prompt,
        min_length=len(sequence),
        max_length=len(sequence),
        num_beams=10,
        early_stopping=False,
        num_return_sequences=1,
        pad_token_id=pad_token_id,
    )
    logits = model(gen_seq).logits[0]

    loss = compute_loss(sequence, logits, t)
    if not val:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss


def finetune(model, optimizer, train_normal_df):
    last_episode_loss = 0
    count = 0

    best_loss = float("inf")

    finetune_normal_data = train_normal_df.iloc[:100]
    idx = floor(0.9 * len(finetune_normal_data))
    train_set = finetune_normal_data[:idx]
    val_set = finetune_normal_data[idx:]

    for episode in tqdm(range(num_episodes), "Episode: "):
        model.train()
        finetune_trainset = train_set["line"].sample(frac=1)
        episode_loss = 0
        for sequence in tqdm(finetune_trainset, "Finetuning: "):
            episode_loss += step(sequence)
        episode_loss /= len(finetune_trainset)

        print(f"Episode {episode}/{num_episodes} (finetune): {episode_loss}")

        model.eval()
        episode_loss = 0
        for sequence in tqdm(val_set["line"], "Evaluating: "):
            with torch.no_grad():
                episode_loss += step(sequence, val=True)
        episode_loss /= len(val_set)

        if last_episode_loss - episode_loss <= loss_improv_epsilon:
            print(f"Loss barely changed {last_episode_loss} to {episode_loss}")
            count += 1

        if count > 3:
            print("Early stop!")
            break

        if episode_loss < best_loss or episode == 0:
            print("New best model! Saving...")
            save_model(model, optimizer)
            best_loss = episode_loss

        print(f"Episode {episode}/{num_episodes} (eval): {episode_loss}")
        last_episode_loss = episode_loss