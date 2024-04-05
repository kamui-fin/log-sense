import pandas as pd
from log_gpt.finetune import finetune
from log_gpt.inference import predict_topk
from log_gpt.pretrain import pretrain_model, setup_model_optimizer
from rapid.build_db import build_normal_db
from rapid.model import predict_test


class Rapid:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def fit(self, normal_logs) -> None:
        self.normal_db, self.knn = build_normal_db(normal_logs, self.output_dir)

    def predict(self, test_logs) -> pd.Series:
        return predict_test(test_logs, self.normal_db, self.output_dir, self.knn)


class LogGPT:
    def __init__(self, model_path, vocab_size):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.model, self.optimizer = setup_model_optimizer(
            self.vocab_size, cache_path=self.model_path
        )

    def pre_train(self, normal_logs):
        self.model, self.optimizer, pretrain_losses = pretrain_model(
            normal_logs, self.vocab_size, output_path=self.model_path
        )
        return pretrain_losses

    def finetune(self, normal_logs) -> None:
        finetune_losses = finetune(
            self.model,
            self.optimizer,
            normal_logs,
            cache_path=self.model_path,
            sliding_window=True,
        )
        return finetune_losses

    def predict(self, test_logs):
        predictions = predict_topk(test_logs, self.model, sliding_window=True)
        return predictions
