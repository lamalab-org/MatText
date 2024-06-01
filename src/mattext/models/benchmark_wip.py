import os
import traceback

import wandb
from matbench.bench import MatbenchBenchmark
from omegaconf import DictConfig

from mattext.models.finetune_wip import FinetuneModel
from mattext.models.predict_wip import Inference
from mattext.models.score import Task


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
        self.task_cfg = task_cfg
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
        # mb = MatbenchBenchmark(autoload=False)
        # benchmark = getattr(mb, self.benchmark)
        # benchmark.load()
        task = Task(task_name="kvrh", metric="mae")

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
            fold_name = fold_key_namer(i)
            print(fold_name)

            try:
                predict = Inference(exp_cfg)
                predictions = predict.predict()
                prediction_ids = predict.prediction_ids
                task.record_fold(fold=i, prediction_ids=prediction_ids, predictions=predictions)

                #benchmark.record(i, predictions)
            except Exception as e:
                print(
                    f"Error occurred during inference for finetuned checkpoint '{exp_name}':"
                )
                print(traceback.format_exc())

        if not os.path.exists(self.benchmark_save_path):
            os.makedirs(self.benchmark_save_path)

        file_name = os.path.join(
            self.benchmark_save_path, f"{self.representation}_{self.benchmark}.json"
        )
        task.to_file(file_name)
        # Get final results after recording all folds
        final_results = task.get_final_results()
        print(final_results)
