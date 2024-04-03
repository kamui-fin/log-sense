from log_gpt.pretrain import *
from log_gpt.finetune import *
from log_gpt.preprocess import *
from log_gpt.inference import *
from log_gpt.model import *

df, vocab_size = load_bgl(Path("data/bgl"), Path("data/bgl/bgl_windows.csv"))
train_df, test_df = train_test_split(df)

log_gpt = LogGPT(Path("output/log_gpt/model_bgl_pre.pt"), vocab_size)
log_gpt.pre_train(train_df)
log_gpt.finetune(train_df)
log_gpt.predict(test_df)
