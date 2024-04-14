import torch
from log_gpt import LogGPTInferenceAPI
from models import ServiceConfig
import schedule
import time
from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader

class ChunkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids']).cuda(),
            'attention_mask': torch.tensor(sample['attention_mask']).cuda(),
            'labels': torch.tensor(sample['labels']).cuda(),
        }

def task():
    print("Running the task")
    client = MongoClient('localhost', 27017)
    db = client.log_sense
    collection = db.gptlogs
    # find where train strategy = "pre-train" & is_anomaly = False & retrieve the chunks property for each document
    
    pipeline = [
        {"$match": {"is_anomaly": False}},
        {"$group": {"_id": "$train_strategy", "chunks": {"$push": "$chunk"}}}
    ]
    
    documents = collection.aggregate(pipeline)
    chunks_by_strat = {}
    for document in documents:
        train_strategy = document.get("_id")
        chunks = document.get("chunks") 
        chunks_by_strat[train_strategy] = chunks
    
    # pre-train
    pre_train_chunks = chunks_by_strat[ "pre-train" ]
    begin_pretraining(pre_train_chunks)
        
    # fine-tune
    # fine_tune_chunks = chunks_by_strat[ "finetune" ]
    # begin_finetuning(fine_tune_chunks)

def begin_pretraining(chunks):
    train_loader = DataLoader(ChunkDataset(chunks), batch_size=2, shuffle=True)
    model = LogGPTInferenceAPI(None)
    print(model.run_inference(train_loader))
    # model.pretrain_model(train_loader)

def begin_finetuning(sequences):
    sequences = torch.tensor([chunk['input_ids'] for chunk in sequences]).cuda()
    model = LogGPTInferenceAPI(None)
    model.finetune(sequences)

# Schedule the task to run every day at 12 am
# schedule.every().day.at("00:00").do(task)

# Keep the script running
# while True:
#     schedule.run_pending()
#     time.sleep(1)

# task()