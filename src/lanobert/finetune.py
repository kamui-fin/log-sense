from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.config import BASE_MODEL, MODEL_CHECKPOINTS, TRAIN_SET_OUTPUT, TEST_SET_OUTPUT

train_dataset = load_from_disk(TRAIN_SET_OUTPUT)
test_dataset = load_from_disk(TEST_SET_OUTPUT)

# If pre-training from scratch only:
# config = BertConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=MAX_LENGTH)
# model = BertForMaskedLM(config=model_config)

config = AutoConfig.from_pretrained(BASE_MODEL)
model = AutoModelForMaskedLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.2)

training_args = TrainingArguments(
    output_dir=MODEL_CHECKPOINTS,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs =1,
    eval_accumulation_steps=2,
    # per_device_train_batch_size=5,
    # gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    # per_device_eval_batch_size=64,
    logging_steps=1,             # evaluate, log and save model checkpoints every 1000 step
    save_strategy='epoch',
    load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()