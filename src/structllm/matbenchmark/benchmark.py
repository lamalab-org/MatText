import torch 
import os
import pandas as pd
import hydra
import wandb
from omegaconf import DictConfig
from matbench.bench import MatbenchBenchmark
from structllm.models.predict import Inference
from structllm.models.finetune import FinetuneModel

class Matbenchmark:
    """
    Class to perform predictions on Matbench datasets.

    Args:
    - task_cfg (DictConfig): Configuration dictionary containing task parameters.
    """

    def __init__(self, task_cfg: DictConfig):
        self.task_cfg = task_cfg
        self.benchmark = self.task_cfg.matbench.record_test.benchmark_dataset
        self.exp_names = self.task_cfg.model.finetune.exp_name
        self.test_exp_names = self.task_cfg.model.inference.exp_name
        self.train_data = task_cfg.model.finetune.path.finetune_traindata
        self.test_data = task_cfg.model.inference.path.test_data
        self.benchmark_save_path = self.task_cfg.model.inference.benchmark_save_file


    def run_benchmarking(self, local_rank=None) -> None:

        mb = MatbenchBenchmark(autoload=False)
        benchmark = getattr(mb, self.benchmark)
        benchmark.load()

        for i, (exp_name, test_name, train_data_path, test_data_path) in enumerate(
            zip(self.exp_names, self.test_exp_names, self.train_data, self.test_data)
        ):

            print("Running training on {}, and testing on {}".format(train_data_path, test_data_path))
            wandb.init(
                config=dict(self.task_cfg.model.finetune), 
                project=self.task_cfg.logging.wandb_project, name=exp_name)
            
            exp_cfg = self.task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path
            

            finetuner = FinetuneModel(exp_cfg,local_rank)
            ckpt = finetuner.finetune()

            wandb.init(
                config=dict(self.task_cfg.model.inference), 
                project=self.task_cfg.logging.wandb_project, name=test_name)
            
            exp_cfg.model.inference.path.test_data = test_data_path
            exp_cfg.model.inference.path.pretrained_checkpoint = ckpt

            predict = Inference(exp_cfg)
            predictions = predict.predict()
            benchmark.record(i, predictions)

        file_name = f"{self.benchmark}.json.gz"


        os.mkdir(self.benchmark_save_path,if_not_exists=True)
        file_name = f"{self.benchmark_save_path}/{self.benchmark}.json.gz"

        benchmark.to_file(self.benchmark_save_path)