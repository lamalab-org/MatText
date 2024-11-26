import os
from typing import Callable, Union

import hydra
import wandb
from hydra import main as hydra_main
from hydra import utils
from omegaconf import DictConfig

from mattext.models.benchmark import Matbenchmark, MatbenchmarkClassification
from mattext.models.finetune import FinetuneModel
from mattext.models.inference import Benchmark
from mattext.models.llama import FinetuneLLama
from mattext.models.llama_sft import FinetuneLLamaSFT
from mattext.models.potential import PotentialModel
from mattext.models.pretrain import PretrainModel


class TaskRunner:
    def __init__(self):
        self.wandb_api_key = os.environ.get("WANDB_API_KEY")
        self.task_map = {
            "benchmark": self.run_benchmarking,
            "classification": self.run_classification,
            "inference": self.run_inference,
            "finetune": self.run_finetuning,
            "pretrain": self.run_pretraining,
            "qmof": self.run_qmof,
            "llama": self.run_llama,
            "llama_sft": self.run_llama_sft,
            "potential": self.run_potential,
        }

    def run_task(self, run: list, task_cfg: DictConfig, local_rank=None) -> None:
        for task in run:
            if task in self.task_map:
                self.task_map[task](task_cfg, local_rank)
            else:
                print(f"Unknown task: {task}")

    def _run_experiment(
        self,
        task_cfg: DictConfig,
        local_rank: Union[int, None],
        model_class: Callable,
        experiment_type: str,
        use_folds: bool = False,
        use_train_data_path: bool = False,
    ):
        if use_folds:
            iterations = range(task_cfg.model.fold)
        elif use_train_data_path:
            iterations = zip(
                task_cfg.model.finetune.exp_name,
                task_cfg.model.finetune.path.finetune_traindata,
            )
        else:
            iterations = [None]

        for item in iterations:
            if use_folds:
                exp_name = f"{task_cfg.model.finetune.exp_name}_fold_{item}"
                fold = f"fold_{item}"
            elif use_train_data_path:
                exp_name, train_data_path = item
                fold = None
            else:
                exp_name = task_cfg.model[experiment_type].exp_name
                fold = None

            wandb.init(
                config=dict(task_cfg.model[experiment_type]),
                project=task_cfg.model.logging.wandb_project,
                name=exp_name,
            )

            exp_cfg = task_cfg.copy()
            exp_cfg.model[experiment_type].exp_name = exp_name
            if use_train_data_path:
                exp_cfg.model.finetune.path.finetune_traindata = train_data_path

            if fold:
                model = model_class(exp_cfg, local_rank, fold=fold)
            else:
                model = model_class(exp_cfg, local_rank)

            result = (
                model.finetune() if hasattr(model, "finetune") else model.pretrain_mlm()
            )
            print(result)
            wandb.finish()

    def run_benchmarking(self, task_cfg: DictConfig, local_rank=None) -> None:
        print("Benchmarking")
        benchmark = Matbenchmark(task_cfg)
        benchmark.run_benchmarking(local_rank=local_rank)

    def run_classification(self, task_cfg: DictConfig, local_rank=None) -> None:
        print("Benchmarking Classification")
        benchmark = MatbenchmarkClassification(task_cfg)
        benchmark.run_benchmarking(local_rank=local_rank)

    def run_qmof(self, task_cfg: DictConfig, local_rank=None) -> None:
        print("Finetuning on qmof")
        matbench_predictor = Matbenchmark(task_cfg)
        matbench_predictor.run_qmof(local_rank=local_rank)

    def run_inference(self, task_cfg: DictConfig, local_rank=None) -> None:
        print("Testing on matbench dataset")
        matbench_predictor = Benchmark(task_cfg)
        matbench_predictor.run_benchmarking(local_rank=local_rank)

    def run_llama(self, task_cfg: DictConfig, local_rank=None) -> None:
        self._run_experiment(
            task_cfg, local_rank, FinetuneLLama, "finetune", use_train_data_path=True
        )

    def run_llama_sft(self, task_cfg: DictConfig, local_rank=None) -> None:
        self._run_experiment(
            task_cfg, local_rank, FinetuneLLamaSFT, "finetune", use_folds=True
        )

    def run_finetuning(self, task_cfg: DictConfig, local_rank=None) -> None:
        self._run_experiment(
            task_cfg, local_rank, FinetuneModel, "finetune", use_train_data_path=True
        )

    def run_potential(self, task_cfg: DictConfig, local_rank=None) -> None:
        self._run_experiment(
            task_cfg, local_rank, PotentialModel, "finetune", use_train_data_path=True
        )

    def run_pretraining(self, task_cfg: DictConfig, local_rank=None) -> None:
        self._run_experiment(task_cfg, local_rank, PretrainModel, "pretrain")

    def initialize_wandb(self):
        if self.wandb_api_key:
            wandb.login(key=self.wandb_api_key)
            print("W&B API key found")
        else:
            print(
                "W&B API key not found. Please set the WANDB_API_KEY environment variable."
            )


@hydra_main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    task_runner = TaskRunner()
    task_runner.initialize_wandb()

    if cfg.runs:
        print(cfg)
        runs = utils.instantiate(cfg.runs)
        print(runs)
        for run in runs:
            print(run)
            task_runner.run_task(run["tasks"], task_cfg=cfg, local_rank=local_rank)


if __name__ == "__main__":
    main()
