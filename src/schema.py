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
                                      "rnn_sde", "lstm_sde", "gru_sde", "gru", "gpt2_ao_sde",
                                      "gpt2_cont", "gpt2_ao_cont"])),
    "n_positions": merge(tinteger, required),   # maximum context length
    "n_dims": merge(tinteger, required),        # dimension of theta
    "n_vars": merge(tinteger, required),        # number of variables: X and Y
    "n_embd": merge(tinteger, required),
    "n_bins": merge(tinteger, default(1)),      # number of bins
    "n_layer": merge(tinteger, required),
    "block_setup": merge(tinteger, default(1)), # whether to use block setup for counterfactual regression
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
    "predict_y": merge(tinteger, default(0)),
    "predict_x": merge(tinteger, default(0)),
    "predict_beta": merge(tinteger, default(0)),
    "learning_rate": merge(tfloat, default(0.0001)),
    "train_steps": merge(tinteger, default(50000)),
    "eval_steps": merge(tinteger, default(10000)),
    "training_loss": merge(tstring, default("mse")),
    "diversity": merge(tinteger, default(16000000)),        # 50000 * 64 * 5
    "theta_dist": merge(tstring, default("uniform")),
    "transformation": merge(tstring, default("addlin")),    # one of [addlin, mullin, tanh, sigmoid]
    "constant_z": merge(tinteger, default(-1)),             # whether to use constant z index for delimiter
    "randomize_labels": merge(tinteger, default(0)),        # whether to use randomize labels
    "lamb": merge(tfloat, default(200)),                    # lambda for poisson process in sdes
    "max_time": merge(tfloat, default(0.1)),                # max_time for poisson process in sdes
    "poisson": merge(tinteger, default(1)),                 # time steps by poisson process (1) or uniform time steps (0)
    "ode": merge(tinteger, default(0)),                     # ode (1) or sde (0) for diffusion of sdeint under g()
    "diffusion": merge(tinteger, default(20)),              # diffusion parameter for distribution of sigma in SDE
    "number_events": merge(tinteger, default(None)),        # fixed number of events for poisson 1
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
