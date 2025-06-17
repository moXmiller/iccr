from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm", "gpt2_sde", "gpt2_ao", "gpt2_mlp", "rnn", 
                                      "rnn_sde", "lstm_sde", "gru_sde", "gru", "gpt2_ao_sde"])),
    "n_positions": merge(tinteger, required),   # maximum context length
    "n_dims": merge(tinteger, required),        # dimension of theta
    "n_vars": merge(tinteger, required),        # number of variables: X and Y
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "head_dim": merge(tinteger, default(64)),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),         # initial parameter
    "end": merge(tinteger, required),           # limit of final value
    "inc": merge(tinteger, required),
    "interval": merge(tinteger, required),
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
    "vars": stdict(curriculum_base_schema),     # include vars schedule in curriculum
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "counterfactual_regression",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "data_kwargs": merge(tdict, required),                  # data_kwargs for DAG type: not required for this project
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian", "sde"])),
    "batch_size": merge(tinteger, default(64)),
    "continuation": merge(tinteger, default(0)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(200000)),
    "eval_steps": merge(tinteger, default(1000)),
    "diversity": merge(tinteger, default(128000000)),       # 200000 * 64 * 5
    "theta_dist": merge(tstring, default("uniform")),
    "transformation": merge(tstring, default("addlin")),    # one of [addlin, mullin, tanh, sigmoid]
    "lamb": merge(tinteger, default(200)),                  # lambda for poisson process in sdes
    "max_time": merge(tfloat, default(0.1)),                # max_time for poisson process in sdes
    "save_every_steps": merge(tinteger, default(1000)),     # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),       # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
