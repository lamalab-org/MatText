import os
import traceback

import wandb
from matbench.bench import MatbenchBenchmark
from omegaconf import DictConfig

from mattext.models.finetune import FinetuneModel
from mattext.models.predict import Inference

def fold_key_namer(fold_key):
    return f"fold_{fold_key}"

class Matbenchmark:
    """
    Class to perform predictions on Matbench datasets.

    Args:
    - task_cfg (DictConfig): Configuration dictionary containing task parameters.
    """

    def __init__(self, task_cfg: DictConfig):
        """
        Initializes the object with the given task configuration.

        Parameters:
            task_cfg (DictConfig): The configuration dictionary containing task parameters.

        Returns:
            None
        """
        self.representation = self.task_cfg.model.representation
        self.benchmark = self.task_cfg.model.inference.benchmark_dataset
        self.exp_names = self.task_cfg.model.finetune.exp_name
        self.test_exp_names = self.task_cfg.model.inference.exp_name
        self.train_data = task_cfg.model.finetune.path.finetune_traindata
        self.test_data = task_cfg.model.inference.path.test_data
        self.benchmark_save_path = self.task_cfg.model.inference.benchmark_save_file

        # override wandb project name & tokenizer
        self.wandb_project = self.task_cfg.model.logging.wandb_project

    def run_benchmarking(self, local_rank=None) -> None:
        """
        Runs benchmarking on the specified dataset.

        Args:
            local_rank (int, optional): The local rank for distributed training. Defaults to None.

        Returns:
            None

        Raises:
            Exception: If an error occurs during inference for a finetuned checkpoint.

        """
        mb = MatbenchBenchmark(autoload=False)
        benchmark = getattr(mb, self.benchmark)
        benchmark.load()

        for i, (exp_name, test_name, train_data_path, test_data_path) in enumerate(
            zip(self.exp_names, self.test_exp_names, self.train_data, self.test_data)
        ):
            print(
                f"Running training on {train_data_path}, and testing on {test_data_path}"
            )
            wandb.init(
                config=dict(self.task_cfg.model.finetune),
                project=self.task_cfg.model.logging.wandb_project,
                name=exp_name,
            )

            exp_cfg = self.task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path
            fold = fold_key_namer(i)

            finetuner = FinetuneModel(exp_cfg, local_rank,fold)
            ckpt = finetuner.finetune()
            print("-------------------------")
            print(ckpt)
            print("-------------------------")

            wandb.init(
                config=dict(self.task_cfg.model.inference),
                project=self.task_cfg.model.logging.wandb_project,
                name=test_name,
            )

            exp_cfg.model.inference.path.test_data = test_data_path
            exp_cfg.model.inference.path.pretrained_checkpoint = ckpt

            try:
                predict = Inference(exp_cfg,fold=fold)
                predictions = predict.predict()

                #benchmark.record(i, predictions)
            except Exception as e:
                print(
                    f"Error occurred during inference for finetuned checkpoint '{exp_name}':"
                )
                print(traceback.format_exc())

        if not os.path.exists(self.benchmark_save_path):
            os.makedirs(self.benchmark_save_path)

        file_name = os.path.join(
            self.benchmark_save_path, f"{self.representation}_{self.benchmark}.json.gz"
        )
        benchmark.to_file(file_name)

    def run_qmof(self, local_rank=None) -> None:

        for i, (exp_name, test_name, train_data_path, test_data_path) in enumerate(
            zip(self.exp_names, self.test_exp_names, self.train_data, self.test_data)
        ):
            print(
                f"Running training on {train_data_path}, and testing on {test_data_path}"
            )
            wandb.init(
                config=dict(self.task_cfg.model.finetune),
                project=self.task_cfg.model.logging.wandb_project,
                name=exp_name,
            )

            exp_cfg = self.task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path

            finetuner = FinetuneModel(exp_cfg, local_rank)
            ckpt = finetuner.finetune()
            print("-------------------------")
            print(ckpt)
            print("-------------------------")

            wandb.init(
                config=dict(self.task_cfg.model.inference),
                project=self.task_cfg.model.logging.wandb_project,
                name=test_name,
            )

            exp_cfg.model.inference.path.test_data = test_data_path
            exp_cfg.model.inference.path.pretrained_checkpoint = ckpt

            try:
                predict = Inference(exp_cfg)
                predictions = predict.predict()

            except Exception as e:
                print(
                    f"Error occurred during inference for finetuned checkpoint '{exp_name}':"
                )
                print(traceback.format_exc())

        if not os.path.exists(self.benchmark_save_path):
            os.makedirs(self.benchmark_save_path)
        import numpy as np

        np.save(self.benchmark_save_path, predictions.predictions)
