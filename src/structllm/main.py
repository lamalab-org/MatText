import os
from hydra import main as hydra_main, utils
from omegaconf import DictConfig
from structllm.models.finetune_val import FinetuneModelwithEval
from structllm.models.pretrain import PretrainModel
from structllm.models.predict import Inference
from structllm.matbenchmark.matbench_test import MatbenchPredict
import wandb


class TaskRunner:
    def __init__(self):
        self.wandb_api_key = os.environ.get("WANDB_API_KEY")

    def run_task(self, run: list , task_cfg: DictConfig) -> None:

        if "matbench_predict" in run: 
            self.run_matbench_prediction(task_cfg)

        if "predict" in run:
            print(task_cfg)
            self.run_predictions(task_cfg)

        if "finetune" in run:
            self.run_finetuning(task_cfg)

        if "pretrain" in run:
            self.run_pretraining(task_cfg)

    def run_matbench_prediction(self, task_cfg: DictConfig) -> None:
        print("performing benchmaking")
        matbench_predictor = MatbenchPredict(task_cfg)
        matbench_predictor.run_benchmark()

    def run_predictions(self,task_cfg: DictConfig) -> None:
        for exp_name, test_data_path, ckpt in zip(task_cfg.model.inference.exp_name, task_cfg.model.inference.path.test_data, task_cfg.model.inference.path.pretrained_checkpoint):
            wandb.init(
                config=dict(task_cfg.model.inference), 
                project=task_cfg.logging.wandb_project, 
                name=exp_name
                    )
            
            exp_cfg = task_cfg.copy()
            exp_cfg.model.inference.exp_name = exp_name
            exp_cfg.model.inference.path.test_data = test_data_path
            exp_cfg.model.inference.path.pretrained_checkpoint = ckpt
            print(exp_cfg.model.inference.path.test_data)

            predict = Inference(exp_cfg)
            print(predict.predict())
            wandb.finish()


    def run_finetuning(self, task_cfg: DictConfig) -> None:
        for exp_name, train_data_path in zip(task_cfg.model.finetune.exp_name, task_cfg.model.finetune.path.finetune_traindata):
            wandb.init(
                config=dict(task_cfg.model.finetune), 
                project=task_cfg.logging.wandb_project, name=exp_name)
            
            exp_cfg = task_cfg.copy()
            exp_cfg.model.finetune.exp_name = exp_name
            exp_cfg.model.finetune.path.finetune_traindata = train_data_path

            
            finetuner = FinetuneModelwithEval(exp_cfg)
            finetuner.finetune()
            wandb.finish()

    def run_pretraining(self, task_cfg: DictConfig) -> None:

        wandb.init(
                config=dict(task_cfg.model.pretrain), 
                project=task_cfg.logging.wandb_project, 
                name=task_cfg.model.pretrain.exp_name
                    )
        pretrainer = PretrainModel(task_cfg)
        pretrainer.pretrain_mlm()

    def initialize_wandb(self):
        if self.wandb_api_key:
            wandb.login(key=self.wandb_api_key)
            print("W&B API key found")
        else:
            print("W&B API key not found. Please set the WANDB_API_KEY environment variable.")

@hydra_main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    task_runner = TaskRunner()
    task_runner.initialize_wandb()

    if cfg.runs:
        print(cfg)
        runs = utils.instantiate(cfg.runs)
        print(runs)
        for run in runs:
            print(run)
            task_runner.run_task(run['tasks'],task_cfg=cfg)


if __name__ == "__main__":
    main()

    
