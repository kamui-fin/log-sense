from log_gpt.pretrain import *
from log_gpt.finetune import *
from log_gpt.preprocess import *
from log_gpt.inference import *

df, vocab_size = load_bgl(Path('data/bgl'), Path('data/bgl/bgl_windows.csv'))
train_df, test_df = train_test_split(df)

# model, optimizer, pretrain_losses = pretrain_model(train_df, vocab_size, Path('output/log_gpt/model_bgl_pre.pt'))
model, optimizer = setup_model_optimizer(vocab_size, cache_path=Path('output/log_gpt/model_bgl_pre.pt'))
finetune_losses = finetune(model, optimizer, train_df, cache_path=Path('output/log_gpt/model_bgl_fine.pt'), sliding_window=True)
evaluate_topk(test_df, model, sliding_window=True, trim = True)