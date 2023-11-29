import torch 
import pandas as pd
import hydra
import wandb
from omegaconf import DictConfig
from matbench.bench import MatbenchBenchmark
from structllm.models.predict import Inference

class MatbenchPredict:
    """
    Class to perform predictions on Matbench datasets.

    Args:
    - task_cfg (DictConfig): Configuration dictionary containing task parameters.
    """

    def __init__(self, task_cfg: DictConfig):
        self.task_cfg = task_cfg
        self.exp_names = self.task_cfg.matbench.record_test.exp_name
        self.test_datasets = self.task_cfg.matbench.record_test.list_test_data
        self.pretrained_checkpoints = self.task_cfg.matbench.record_test.checkpoints
        self.benchmark = self.task_cfg.matbench.record_test.benchmark_dataset
        self.benchmark_save_path = self.task_cfg.matbench.record_test.benchmark_save_file

    def run_benchmark(self):
        """
        Runs predictions on Matbench datasets using the provided configurations.
        """
        mb = MatbenchBenchmark(autoload=False)
        benchmark = getattr(mb, self.benchmark)
        benchmark.load()

        for i, (exp_name, test_data_path, ckpt) in enumerate(
            zip(self.exp_names, self.test_datasets, self.pretrained_checkpoints)
        ):
            # Initialize Weights and Biases run
            wandb.init(
                config=dict(self.task_cfg.matbench),
                project=self.task_cfg.logging.wandb_project,
                name=exp_name
            )
            
            exp_cfg = self.task_cfg.copy()
            exp_cfg.model.inference.exp_name = exp_name
            exp_cfg.model.inference.path.test_data = test_data_path
            exp_cfg.model.inference.path.pretrained_checkpoint = ckpt

            print(exp_cfg.model.inference.path.test_data)
            
            # Perform inference and record predictions
            predict = Inference(exp_cfg)
            predictions = predict.predict()
            wandb.finish()
            benchmark.record(i, predictions)

        benchmark.to_file(self.benchmark_save_path)
