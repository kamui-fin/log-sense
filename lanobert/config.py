from pathlib import Path

BASE_MODEL = "google-bert/bert-base-uncased"

DATA_DIR = Path("data")

HDFS_UNCLEANED_PATH = DATA_DIR / "hdfsv1.log"
HDFS_CLEANED_PATH = DATA_DIR / "hdfsv1_regex.log"
MODEL_CHECKPOINTS = DATA_DIR / "hdfs_bert"
LAST_CHECKPOINT = "checkpoint-99"

TRAIN_SET_OUTPUT = DATA_DIR / 'torch_dataset' / 'train'
TEST_SET_OUTPUT = DATA_DIR / 'torch_dataset' / 'test'

CONTEXT_LENGTH = 128

# Only for pre-training
# TOKENIZER_PATH = f'data/hdfs_tokenized'