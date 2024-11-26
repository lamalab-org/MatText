import os
import traceback

import wandb
from loguru import logger
from matbench.bench import MatbenchBenchmark
from omegaconf import DictConfig

from mattext.models.predict import Inference


class Benchmark:
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
        self.wndb = task_cfg.model.logging.wandb_project
        self.benchmark_save_path = self.task_cfg.model.inference.benchmark_save_file

        # override wandb project name & tokenizer
        self.wandb_project = self.task_cfg.model.logging.wandb_project

    def run_benchmarking(self, local_rank=None) -> None:
        """
        Runs benchmarking on the specified dataset.

        This function loads the MatbenchBenchmark dataset and iterates over the exp_names, test_exp_names, train_data, and test_data lists.
        For each iteration, it prints the training and testing data paths.
        It then initializes the experiment configuration, sets the exp_name, train_data_path, and test_data_path in the configuration.
        It prints the finetuned model name and initializes the wandb configuration for the inference.
        It sets the test_data_path and pretrained_checkpoint in the configuration.
        It then tries to predict using the Inference class and records the predictions using the MatbenchBenchmark class.
        If an exception occurs during inference, it prints the error message and traceback.
        Finally, it saves the benchmark results to a JSON file.

        Parameters:
            local_rank (int, optional): The local rank for distributed training. Defaults to None.

        Returns:
            None
        """

        mb = MatbenchBenchmark(autoload=False)
        benchmark = getattr(mb, self.benchmark)
        benchmark.load()

        for i, (exp_name, test_name, train_data_path, test_data_path) in enumerate(
            zip(self.exp_names, self.test_exp_names, self.train_data, self.test_data)
        ):
            logger.info(
                f"Running training on {train_data_path}, and testing on {test_data_path}"
            )
            exp_cfg = self.task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path

            ckpt = exp_cfg.model.finetune.path.finetuned_modelname
            logger.info("Checkpoint: ", ckpt)

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
                benchmark.record(i, predictions)
            except Exception as e:
                logger.error(
                    f"Error occurred during inference for finetuned checkpoint '{exp_name}':"
                )
                logger.error(traceback.format_exc())

        if not os.path.exists(self.benchmark_save_path):
            os.makedirs(self.benchmark_save_path)

        file_name = os.path.join(
            self.benchmark_save_path,
            f"{self.representation}_{self.wndb}_{self.benchmark}.json.gz",
        )
        benchmark.to_file(file_name)
