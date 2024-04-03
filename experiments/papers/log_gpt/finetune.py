import torch
from torch.distributions import Categorical
from math import floor
from tqdm import tqdm
from log_gpt.config import *
from log_gpt.preprocess import train_val_split
from log_gpt.pretrain import save_model


def compute_loss(input_ids, logits, start_gen_pos):
    reward = log_prob = 0
    for i in range(start_gen_pos - 1, len(input_ids) - 1):
        softmax = Categorical(logits=logits[i])
        next_token_id = input_ids[i + 1]
        log_prob += softmax.log_prob(torch.tensor(next_token_id).cuda())
        top_next_tokens = torch.topk(logits[i], k=top_k).indices
        reward += 1 if next_token_id in top_next_tokens else -1
    cost = -reward * log_prob
    return cost


def step(model, optimizer, sequence, val=False, sliding_window=False):
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

    loss = compute_loss(sequence, logits, start_gen_pos)
    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def finetune(model, optimizer, train_normal_df, cache_path=None, sliding_window=False):
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
            train_loss += step(
                model, optimizer, sequence, sliding_window=sliding_window
            )
        train_loss /= len(finetune_trainset)

        print(f"Episode {episode}/{num_episodes} (finetune): {train_loss}")

        model.eval()
        val_loss = 0
        for sequence in tqdm(val_set["line"], "Evaluating: "):
            with torch.no_grad():
                val_loss += step(
                    model, optimizer, sequence, val=True, sliding_window=sliding_window
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
            save_model(model, optimizer, cache_path)
            best_loss = val_loss

        print(f"Episode {episode}/{num_episodes} (eval): {val_loss}")
        last_episode_loss = val_loss
        J.append((train_loss, val_loss))
    return J
