from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

from rapid.build_db import build_db, build_normal_db, distance, get_core_set

def predict_test(test_X, normal_db, output_dir, knn):
    test_log_seqs = test_X.to_list()
    db = build_db(test_log_seqs, 'test', output_dir)

    ## INFERENCE
    print('Calculating abnormal scores...')
    abnormal_scores = [0] * db.num_unique_logs
    for index in tqdm(range(db.num_unique_logs)):
        core_set = get_core_set(db.embedding_db[index][0], knn)
        abnormal_scores[index] = min(distance(db.embedding_db[index], normal_db.embedding_db[c_index]) for c_index in core_set[1:])

    print(f'Generating final predictions...')
    final_pred = [0] * len(db.lookup_db)
    for orig, uniq in db.lookup_db.items():
        final_pred[orig] = abnormal_scores[uniq].item()

    final_pred = pd.Series(final_pred)
    return final_pred

def evaluate(test_y, final_pred):
    print('Done!')
    precision, recall, thresholds = precision_recall_curve(test_y, final_pred)
    f_score = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(f_score)

    print(f'Best Threshold = {thresholds[ix]}')
    print(classification_report(test_y, final_pred > thresholds[ix], target_names=['Normal', 'Anomaly']))

    f1 = f1_score(test_y, final_pred > thresholds[ix])
    print(f'F1 = {f1}')

    auc = roc_auc_score(test_y, final_pred)
    print(f'AUC = {auc}')

class Rapid:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def fit(self, normal_logs) -> None:
        self.normal_db, self.knn = build_normal_db(normal_logs, self.output_dir)

    def predict(self, test_logs) -> pd.Series:
        return predict_test(test_logs, self.normal_db, self.output_dir, self.knn)