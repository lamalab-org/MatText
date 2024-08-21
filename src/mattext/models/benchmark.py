import os
import traceback
from abc import ABC, abstractmethod

import wandb
from matbench.bench import MatbenchBenchmark
from omegaconf import DictConfig

from mattext.models.finetune import FinetuneModel, FinetuneClassificationModel
from mattext.models.predict import Inference, InferenceClassification
from mattext.models.score import (
    MATTEXT_MATBENCH,
    MatTextTask,
)
from mattext.models.utils import fold_key_namer


class BaseBenchmark(ABC):
    def __init__(self, task_cfg: DictConfig):
        self.task_cfg = task_cfg
        self.representation = self.task_cfg.model.representation
        self.task = self.task_cfg.model.dataset
        self.task_type = self.task_cfg.model.dataset_type
        self.benchmark = self.task_cfg.model.inference.benchmark_dataset
        self.exp_names = self.task_cfg.model.finetune.exp_name
        self.test_exp_names = self.task_cfg.model.inference.exp_name
        self.train_data = self.task_cfg.model.finetune.dataset_name
        self.test_data = self.task_cfg.model.inference.benchmark_dataset
        self.benchmark_save_path = self.task_cfg.model.inference.benchmark_save_file
        self.wandb_project = self.task_cfg.model.logging.wandb_project

    @abstractmethod
    def run_benchmarking(self, local_rank=None) -> None:
        pass

    def _initialize_task(self):
        if self.task_type == "matbench":
            mb = MatbenchBenchmark(autoload=False)
            task = getattr(mb, MATTEXT_MATBENCH[self.task])
            task.load()
        else:
            task = MatTextTask(task_name=self.task)
        return task

    def _run_experiment(self, task, i, exp_name, test_name, local_rank):
        fold_name = fold_key_namer(i)
        print(
            f"Running training on {self.train_data}, and testing on {self.test_data} for fold {i}"
        )
        print("-------------------------")
        print(fold_name)
        print("-------------------------")

        exp_cfg = self.task_cfg.copy()
        exp_cfg.model.finetune.exp_name = exp_name
        exp_cfg.model.finetune.path.finetune_traindata = self.train_data

        finetuner = self._get_finetuner(exp_cfg, local_rank, fold_name)
        ckpt = finetuner.finetune()
        print("-------------------------")
        print(ckpt)
        print("-------------------------")

        wandb.init(
            config=dict(self.task_cfg.model.inference),
            project=self.task_cfg.model.logging.wandb_project,
            name=test_name,
        )

        exp_cfg.model.inference.path.test_data = self.test_data
        exp_cfg.model.inference.path.pretrained_checkpoint = ckpt

        try:
            predict = self._get_inference(exp_cfg, fold_name)
            predictions, prediction_ids = predict.predict()
            self._record_predictions(task, i, predictions, prediction_ids)
        except Exception as e:
            print(
                f"Error occurred during inference for finetuned checkpoint '{exp_name}': {str(e)}"
            )
            if isinstance(e, (ValueError, TypeError)):
                    raise
            print(traceback.format_exc())

    @abstractmethod
    def _get_finetuner(self, exp_cfg, local_rank, fold_name):
        pass

    @abstractmethod
    def _get_inference(self, exp_cfg, fold_name):
        pass

    @abstractmethod
    def _record_predictions(self, task, fold, predictions, prediction_ids):
        pass

    def _save_results(self, task):
        if not os.path.exists(self.benchmark_save_path):
            os.makedirs(self.benchmark_save_path)

        file_name = os.path.join(
            self.benchmark_save_path,
            f"mattext_benchmark_{self.representation}_{self.benchmark}.json",
        )
        task.to_file(file_name)


class Matbenchmark(BaseBenchmark):
    def run_benchmarking(self, local_rank=None) -> None:
        task = self._initialize_task()

        for i, (exp_name, test_name) in enumerate(
            zip(self.exp_names, self.test_exp_names)
        ):
            wandb.init(
                config=dict(self.task_cfg.model.finetune),
                project=self.task_cfg.model.logging.wandb_project,
                name=exp_name,
            )
            self._run_experiment(task, i, exp_name, test_name, local_rank)

        self._save_results(task)

    def _get_finetuner(self, exp_cfg, local_rank, fold_name):
        return FinetuneModel(exp_cfg, local_rank, fold=fold_name)

    def _get_inference(self, exp_cfg, fold_name):
        return Inference(exp_cfg, fold=fold_name)

    def _record_predictions(self, task, fold, predictions, prediction_ids):
        if self.task_type == "matbench":
            task.record(fold, predictions)
        else:
            task.record_fold(
                fold=fold, prediction_ids=prediction_ids, predictions=predictions
            )


class MatbenchmarkClassification(BaseBenchmark):
    def run_benchmarking(self, local_rank=None) -> None:
        task = self._initialize_task()

        for i, (exp_name, test_name) in enumerate(
            zip(self.exp_names, self.test_exp_names)
        ):
            wandb.init(
                config=dict(self.task_cfg.model.finetune),
                project=self.task_cfg.model.logging.wandb_project,
                name=exp_name,
            )
            self._run_experiment(task, i, exp_name, test_name, local_rank)

        self._save_results(task)

    def _initialize_task(self):
        return MatTextTask(task_name=self.task, is_classification=True)

    def _get_finetuner(self, exp_cfg, local_rank, fold_name):
        return FinetuneClassificationModel(exp_cfg, local_rank, fold=fold_name)

    def _get_inference(self, exp_cfg, fold_name):
        return InferenceClassification(exp_cfg, fold=fold_name)

    def _record_predictions(self, task, fold, predictions, prediction_ids):
        task.record_fold(
            fold=fold, prediction_ids=prediction_ids, predictions=predictions.values
        )
