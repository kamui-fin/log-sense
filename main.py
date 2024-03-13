from pathlib import Path
from src.dataset import Hdfs
from src.rapid.model import Rapid, evaluate

# Example usage with hdfs dataset

log_file = Path("data/hdfs/hdfsv1.log")
labels_file = Path("data/hdfs/anomaly_label.csv")

dataset = Hdfs(log_file, labels_file)
train_X, test_X, test_y = dataset.train_test_split()

model = Rapid()
model.fit(train_X)

pred = model.predict(test_y)
evaluate(test_y, pred)