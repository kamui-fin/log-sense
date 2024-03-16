import gc
import torch
import numpy as np

def clear_hw_cache():
    del model
    gc.collect()
    torch.cuda.empty_cache()

def yield_batches(data, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]
