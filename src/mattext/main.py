import os
from collections.abc import Callable

import hydra
import torch
import wandb
from hydra import main as hydra_main
from hydra import utils
from loguru import logger
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
                logger.info(f"Unknown task: {task}")

    def _run_experiment(
        self,
        task_cfg: DictConfig,
        local_rank: int | None,
        model_class: Callable,
        experiment_type: str,
        use_folds: bool = False,
        use_train_data_path: bool = False,
    ):
        if use_folds:
            skip_folds = set(task_cfg.model.get("skip_folds", []))
            iterations = [i for i in range(task_cfg.model.fold) if i not in skip_folds]
        elif use_train_data_path:
            iterations = zip(
                task_cfg.model.finetune.exp_name,
                task_cfg.model.finetune.path.finetune_traindata,
                strict=False,
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

            is_main = local_rank is None or local_rank == 0
            if is_main:
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
            logger.info(result)

            if is_main:
                wandb.finish()

    def run_benchmarking(self, task_cfg: DictConfig, local_rank=None) -> None:
        logger.info("Benchmarking")
        benchmark = Matbenchmark(task_cfg)
        benchmark.run_benchmarking(local_rank=local_rank)

    def run_classification(self, task_cfg: DictConfig, local_rank=None) -> None:
        logger.info("Benchmarking Classification")
        benchmark = MatbenchmarkClassification(task_cfg)
        benchmark.run_benchmarking(local_rank=local_rank)

    def run_qmof(self, task_cfg: DictConfig, local_rank=None) -> None:
        logger.info("Finetuning on qmof")
        matbench_predictor = Matbenchmark(task_cfg)
        matbench_predictor.run_qmof(local_rank=local_rank)

    def run_inference(self, task_cfg: DictConfig, local_rank=None) -> None:
        logger.info("Testing on matbench dataset")
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
            logger.info("W&B API key found")
        else:
            logger.info(
                "W&B API key not found. Please set the WANDB_API_KEY environment variable."
            )


@hydra_main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    # Get local rank from environment (set by torchrun/accelerate)
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))

    # If LOCAL_RANK is not set, we're not in DDP mode
    if local_rank == -1:
        local_rank = None
        logger.info("Running in single-process mode")
    else:
        logger.info(f"Running in DDP mode with LOCAL_RANK={local_rank}")

        # Validate that distributed is properly initialized
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            logger.info(f"Distributed initialized: rank={rank}/{world_size}")
        else:
            logger.info(
                "WARNING: LOCAL_RANK is set but torch.distributed is not initialized!"
            )
            logger.info(
                "DDP may not work correctly. Please ensure you're using torchrun or accelerate."
            )

    task_runner = TaskRunner()

    # Only initialize wandb on main process
    is_main = local_rank is None or local_rank == 0
    if is_main:
        task_runner.initialize_wandb()

    if cfg.runs:
        logger.info(cfg)
        runs = utils.instantiate(cfg.runs)
        logger.info(runs)
        for run in runs:
            logger.info(run)
            task_runner.run_task(run["tasks"], task_cfg=cfg, local_rank=local_rank)


if __name__ == "__main__":
    main()
