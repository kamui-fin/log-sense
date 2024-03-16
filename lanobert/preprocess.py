import re
import pathlib
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset

from config import BASE_MODEL, HDFS_CLEANED_PATH, HDFS_UNCLEANED_PATH, CONTEXT_LENGTH, TRAIN_SET_OUTPUT, TEST_SET_OUTPUT

ipv4_pattern = '(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])'
ipv6_pattern = '(?:(?:[0-9A-Fa-f]{1,4}:){6}(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|::(?:[0-9A-Fa-f]{1,4}:){5}(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|(?:[0-9A-Fa-f]{1,4})?::(?:[0-9A-Fa-f]{1,4}:){4}(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4})?::(?:[0-9A-Fa-f]{1,4}:){3}(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|(?:(?:[0-9A-Fa-f]{1,4}:){,2}[0-9A-Fa-f]{1,4})?::(?:[0-9A-Fa-f]{1,4}:){2}(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|(?:(?:[0-9A-Fa-f]{1,4}:){,3}[0-9A-Fa-f]{1,4})?::[0-9A-Fa-f]{1,4}:(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|(?:(?:[0-9A-Fa-f]{1,4}:){,4}[0-9A-Fa-f]{1,4})?::(?:[0-9A-Fa-f]{1,4}:[0-9A-Fa-f]{1,4}|(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))|(?:(?:[0-9A-Fa-f]{1,4}:){,5}[0-9A-Fa-f]{1,4})?::[0-9A-Fa-f]{1,4}|(?:(?:[0-9A-Fa-f]{1,4}:){,6}[0-9A-Fa-f]{1,4})?::)(?:%25(?:[A-Za-z0-9\\-._~]|%[0-9A-Fa-f]{2})+)?'
number_pattern = '-?\d+'
datetime_pattern = '\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}'

def regex_clean(text):
    text = re.sub(ipv4_pattern, 'IP', text)
    text = re.sub(ipv6_pattern, 'IP', text)
    text = re.sub(datetime_pattern, 'DATE', text)
    text = re.sub(number_pattern, 'NUM', text)
    return text

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
# grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= CONTEXT_LENGTH:
        total_length = (total_length // CONTEXT_LENGTH) * CONTEXT_LENGTH
    result = {
        k: [t[i : i + CONTEXT_LENGTH] for i in range(0, total_length, CONTEXT_LENGTH)]
        for k, t in concatenated_examples.items()
    }
    return result


#####
## Preliminary REGEX substitution on dataset
#####

hdfs_log_file_path = pathlib.Path(HDFS_UNCLEANED_PATH)
hdfs_log = hdfs_log_file_path.read_text()
hdfs_log_file_path.write_text(regex_clean(hdfs_log))

######
## Tokenizing
######

# To train a new word-piece tokenizer from scratch:
# tokenizer = BertWordPieceTokenizer()
# tokenizer.train(files)
# tokenizer.save_model(TOKENIZER_PATH)

# Using existing one for finetuning
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

#######
## Train-test split + process
#######

def encode(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=CONTEXT_LENGTH, return_special_tokens_mask=True)

dataset = load_dataset('text', data_files=[HDFS_CLEANED_PATH], split='train')
train_dataset = dataset.select(range(0, 1000))
test_dataset = dataset.select(range(1000, 2000))

train_dataset = train_dataset.map(encode, batched=True)
test_dataset = test_dataset.map(encode, batched=True)

train_dataset = train_dataset.map(group_texts, batched=True,
                                desc=f"Grouping texts in chunks of {CONTEXT_LENGTH}")
test_dataset = test_dataset.map(group_texts, batched=True,
                                desc=f"Grouping texts in chunks of {CONTEXT_LENGTH}")

# Save pytorch datasets for faster loading
train_dataset.set_format("torch")
test_dataset.set_format("torch")

train_dataset.save_to_disk(TRAIN_SET_OUTPUT)
test_dataset.save_to_disk(TEST_SET_OUTPUT)