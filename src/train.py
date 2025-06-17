import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from dataset import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, data, o_vars, optimizer, loss_func):
    optimizer.zero_grad()
    output, gt = model(data, o_vars = o_vars)
    loss = loss_func(output, gt)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach(), gt.detach()
    

def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    continuation = bool(args.training.continuation)
    n_dims = model.n_dims
    bsize = args.training.batch_size
    
    dag_type = args.training.data_kwargs["dag_type"]
    max_o_dims = n_dims

    if args.training.data == "sde":
        from sde import get_sde_data_sampler
        data_sampler = get_sde_data_sampler(args.training, o_dims=max_o_dims,
                                            **args.training.data_kwargs)
    else:
        data_sampler = get_data_sampler(args.training,
                                        o_dims = max_o_dims,                # this calls the data_sampler LinearAssignments
                                        **args.training.data_kwargs)

    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,                                                             # we do not require any of this for our purposes
        bsize,                                                              # we simply have it to agree with Garg's structure
        num_tasks=args.training.num_tasks,                                  # required for pool_dict, which we do not use
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    if dag_type == "only_parent": block_setup = True
    else: block_setup = False

    transformation = args.training.transformation

    for i in pbar:
        task_sampler_args = {}        
        
        if args.training.data == "sde":
            assert "sde" in args.model.family
            data = data_sampler.complete_sde_dataset(n_thetas = bsize,
                                                     o_vars = curriculum.n_vars_truncated,
                                                     lamb=args.training.lamb,
                                                     max_time = args.training.max_time,
                                                     n_points= args.model.n_positions,
                                                     itr = i,
                                                     split="train")
            o_points = data_sampler.o_points
        else:
            o_points = curriculum.n_points
            data = data_sampler.complete_dataset(n_thetas = bsize,                
                                                o_points = o_points,
                                                o_vars = curriculum.n_vars_truncated,
                                                itr = i,
                                                continuation = continuation,
                                                block_setup=block_setup,
                                                transformation=transformation)

        task = task_sampler(**task_sampler_args)
        
        ess = data_sampler.ess

        loss_func = task.get_training_metric()

        if dag_type not in ["only_parent", "one_parent", "any"]:
            print("Transformer not defined for this dag_type: modify models.py")
            raise NotImplementedError
        
        loss, output, gt = train_step(model, data.cuda(), o_vars = curriculum.n_vars_truncated,
                                      optimizer = optimizer, loss_func = loss_func)

        #####

        # # what is point_wise stuff?
        # point_wise_tags = list(range(o_points))
        # point_wise_loss_func = task.get_metric()
        # point_wise_loss = point_wise_loss_func(output, gt.cuda()).mean(dim=0)
        
        # baseline_loss = (
        #     sum(
        #         max(curriculum.n_dims_truncated - ii, 0)
        #         for ii in range(o_points)
        #     )
        #     / o_points
        # )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    # "excess_loss": loss / baseline_loss,
                    "effective_support_size": ess,
                    "n_points": o_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "gpt2_sde", "gpt2_ao", "gpt2_mlp", "rnn_sde", "rnn", "lstm_sde", "gru_sde", "gru", "gpt2_ao_sde"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
