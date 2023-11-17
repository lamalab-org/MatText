import os
import hydra
from omegaconf import DictConfig
from structllm.models.finetune import FinetuneModel
from structllm.models.pretrain import PretrainModel
import wandb

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ["WANDB_PROJECT"] = cfg.logging.wandb_project
    os.environ["WANDB_LOG_MODEL"] = cfg.logging.wandb_log_model

    # Initialize W&B session
    wandb.init(config=dict(cfg.model.finetune),
               project=cfg.logging.wandb_project,
               name=cfg.model.finetune.exp_name)

    tasks = cfg.tasks

    if "pretrain" in tasks:
        pretrainer = PretrainModel(cfg)
        pretrainer.pretrain_mlm()

    if "finetune" in tasks:
        finetuner = FinetuneModel(cfg)
        finetuner.finetune()

if __name__ == "__main__":
    # Retrieve the API key from the environment variable
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        print("W&B API key found")
    else:
        print("W&B API key not found. Please set the WANDB_API_KEY environment variable.")
    main()

#run as python main.py task=finetune