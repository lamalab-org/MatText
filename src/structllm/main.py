import os

import hydra
import wandb
from hydra import main as hydra_main
from hydra import utils
from omegaconf import DictConfig

from structllm.models.benchmark import Matbenchmark
from structllm.models.finetune import FinetuneModel
from structllm.models.llama import FinetuneLLama
from structllm.models.llama_sft import FinetuneLLamaSFT
from structllm.models.inference import Benchmark
from structllm.models.pretrain import PretrainModel


class TaskRunner:
    def __init__(self):
        self.wandb_api_key = os.environ.get("WANDB_API_KEY")

    def run_task(self, run: list , task_cfg: DictConfig, local_rank=None) -> None:

        if "benchmark" in run:
            self.run_benchmarking(task_cfg)

        if "inference" in run:
            self.run_inference(task_cfg)

        if "finetune" in run:
            self.run_finetuning(task_cfg)

        if "pretrain" in run:
            self.run_pretraining(task_cfg)

        if "qmof" in run:
            self.run_qmof(task_cfg)

        if "llama" in run:
            self.run_llama(task_cfg,local_rank=local_rank)

        if "llama_sft" in run:
            self.run_llama_sft(task_cfg,local_rank=local_rank)


    def run_benchmarking(self, task_cfg: DictConfig,local_rank=None) -> None:
        print("Finetuning and testing on matbench dataset")
        matbench_predictor = Matbenchmark(task_cfg)
        matbench_predictor.run_benchmarking(local_rank=local_rank)

    def run_qmof(self, task_cfg: DictConfig,local_rank=None) -> None:
        print("Finetuning on qmof")
        matbench_predictor = Matbenchmark(task_cfg)
        matbench_predictor.run_qmof(local_rank=local_rank)

    def run_inference(self, task_cfg: DictConfig,local_rank=None) -> None:
        print("Testing on matbench dataset")
        matbench_predictor = Benchmark(task_cfg)
        matbench_predictor.run_benchmarking(local_rank=local_rank)

    def run_llama(self, task_cfg: DictConfig,local_rank=None) -> None:
        for exp_name, train_data_path in zip(task_cfg.model.finetune.exp_name, task_cfg.model.finetune.path.finetune_traindata):
            wandb.init(
                config=dict(task_cfg.model.finetune),
                project=task_cfg.model.logging.wandb_project, name=exp_name)

            exp_cfg = task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path


            finetuner = FinetuneLLama(exp_cfg,local_rank)
            f = finetuner.finetune()
            print(f)
            wandb.finish()

    def run_llama_sft(self, task_cfg: DictConfig,local_rank=None) -> None:
        for exp_name, train_data_path in zip(task_cfg.model.finetune.exp_name, task_cfg.model.finetune.path.finetune_traindata):
            wandb.init(
                config=dict(task_cfg.model.finetune),
                project=task_cfg.model.logging.wandb_project, name=exp_name)

            exp_cfg = task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path


            finetuner = FinetuneLLamaSFT(exp_cfg,local_rank)
            f = finetuner.finetune()
            print(f)
            wandb.finish()


    def run_finetuning(self, task_cfg: DictConfig,local_rank=None) -> None:
        for exp_name, train_data_path in zip(task_cfg.model.finetune.exp_name, task_cfg.model.finetune.path.finetune_traindata):
            wandb.init(
                config=dict(task_cfg.model.finetune),
                project=task_cfg.logging.wandb_project, name=exp_name)

            exp_cfg = task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path


            finetuner = FinetuneModel(exp_cfg,local_rank)
            finetuner.finetune()
            wandb.finish()

    def run_pretraining(self, task_cfg: DictConfig,local_rank=None) -> None:

        wandb.init(
                config=dict(task_cfg.model.pretrain),
                project=task_cfg.model.logging.wandb_project,
                name=task_cfg.model.pretrain.exp_name
                    )
        print(task_cfg)
        pretrainer = PretrainModel(task_cfg,local_rank)
        pretrainer.pretrain_mlm()

    def initialize_wandb(self):
        if self.wandb_api_key:
            wandb.login(key=self.wandb_api_key)
            print("W&B API key found")
        else:
            print("W&B API key not found. Please set the WANDB_API_KEY environment variable.")




@hydra_main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    task_runner = TaskRunner()
    task_runner.initialize_wandb()

    if cfg.runs:
        print(cfg)
        runs = utils.instantiate(cfg.runs)
        print(runs)
        for run in runs:
            print(run)
            task_runner.run_task(run['tasks'],task_cfg=cfg,local_rank=local_rank)

if __name__ == "__main__":
    main()


