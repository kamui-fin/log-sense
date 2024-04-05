from pathlib import Path

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

train_samples = 5000
train_val_ratio = 0.9
