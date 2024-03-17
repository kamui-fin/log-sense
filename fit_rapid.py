from pathlib import Path
from dataset import Hdfs
from rapid.model import Rapid, evaluate

log_file = Path("data/hdfs/hdfsv1_head.log")
labels_file = Path("data/hdfs/anomaly_label.csv")
cache_file = Path('data/hdfs/cache.csv')

dataset = Hdfs(log_file, labels_file, cache_file)
train_X, test_X, test_y = dataset.train_test_split(100)

output_dir = Path('output/rapid')
model = Rapid(output_dir)
model.fit(train_X)

pred = model.predict(test_X)
evaluate(test_y, pred)