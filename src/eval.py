import os

from munch import Munch
import pandas as pd
import torch
import yaml

import models

def get_model_from_run(run_path, step=-1, only_conf=False, disentangled = False, weights = False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    if disentangled: conf.model.family = "gpt2_disent"
    elif weights: conf.model.family = "gpt2_wei"
    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        if torch.cuda.is_available(): state = torch.load(state_path)
        else: state = torch.load(state_path, map_location=torch.device('cpu'))
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        if torch.cuda.is_available(): state_dict = torch.load(model_path)
        else: state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    return model, conf


def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    # assert len(df) == len(df.run_name.unique())
    return df