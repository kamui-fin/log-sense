from log_gpt.pretrain import *
from log_gpt.finetune import *
from log_gpt.preprocess import *
from log_gpt.inference import *

df, vocab_size = load_hdfs(Path('data/hdfs'), Path('output/log_gpt/drain_parsed.csv'))
train_df, test_df = train_test_split(df)

model, optimizer, pretrain_losses = pretrain_model(train_df, vocab_size, Path('output/log_gpt/model.pt'))
finetune_losses = finetune(model, optimizer, train_df)

evaluate_topk(test_df, model)