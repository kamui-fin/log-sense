from pathlib import Path

model_output_dir = Path("../../output/log_gpt")

embed_dim = 60
layers = 6
heads = 6

context_size = 256
lr_pretraining = 1e-4
lr_finetuning = 1e-6
batch_size = 16

top_k_percentage = 0.5

num_episodes = 20
num_epochs = 10

loss_improv_epsilon = 0.01

bos_token_id = 0
eos_token_id = 1
pad_token_id = 2