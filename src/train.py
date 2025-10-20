import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
import datetime
from tqdm import tqdm
import torch
import yaml
from torch.nn.init import xavier_uniform_

from tasks import get_task_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from eval import get_model_from_run

import wandb

torch.backends.cudnn.benchmark = True


def binning(n_bins, ys, low = -100, high = 100):
    assert high == -low
    mid_edges = torch.linspace(low, high, n_bins - 1)
    bin_edges = mid_edges
    if ys != None: 
        assert torch.equal(ys, ys.contiguous())
        bin_indices = torch.bucketize(ys.contiguous(), bin_edges.cuda())
    else: bin_indices = None
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_indices, bin_centers


def train_step(model, data, o_vars, optimizer, loss_func, n_bins = 1, itr = None, beta = None, loss_weights = [1,0,0,0,0,0,0]):
    optimizer.zero_grad()
    assert beta is None
    
    if beta is None:
        if loss_func.__name__ in ["masked_mean_squared_error", "masked_squared_error"]:
            assert n_bins == 1
            output, gt, gt_mask = model(data, o_vars = o_vars)
            loss = loss_func(output, gt, gt_mask)
        else: 
            output, gt = model(data, o_vars = o_vars)
        
            if n_bins == 1: loss = loss_func(output, gt)
            else: 
                gt_bins, _ = binning(n_bins, gt)
                if args.training.training_loss == "cross_entropy":
                    loss = loss_func(gt_bins.view(-1), output.reshape(-1, n_bins).cuda())
                else:
                    ys_large = torch.nn.functional.one_hot(gt_bins.view(-1).long(), num_classes = n_bins).float()
                    loss = loss_func(ys_large, output.reshape(-1, n_bins).cuda())
        
    # 21.08.
    else:
        assert loss_func.__name__ not in ["masked_mean_squared_error", "masked_squared_error"]
        assert data.shape[1] % 2 != 0
        output, gt, probes = model(data, o_vars = o_vars, output_probes=True)
        z_index = data[0, -3, 0].int()
        assert torch.equal(data[:, -3, :], torch.full_like(data[:, -3, :], z_index))

        assert len(beta.shape) == 2
        assert beta.shape[0] == data.shape[0]
        assert beta.shape[1] == data.shape[2]

        xdiff = data[:, -2, :] - data[:, 2 * z_index, :]
        assert len(xdiff.shape) == 2
        bx = beta.cuda() * xdiff.cuda()

        gts = {
            "beta": beta.cuda(),
            "y": data[:, 2 * z_index + 1, :].cuda(),
            "x": data[:, 2 * z_index, :].cuda(),
            "xcf": data[:, -2, :].cuda(),
            "xdiff": xdiff.cuda(),
            "bx": bx.cuda(),
        }

        assert probes.keys() == gts.keys(), f"Probes keys: {probes.keys()}, GT keys: {gts.keys()}"

        loss = loss_weights[0] * loss_func(output, gt)

        for i, k in enumerate(probes.keys()):
            loss += loss_weights[i + 1] * loss_func(probes[k], gts[k])

    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach(), gt.detach()
    

def train(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)
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
    if "cont" in args.model.family: assert continuation == 1
    else: assert continuation == 0

    n_dims = model.n_dims
    n_bins = args.model.n_bins
    bsize = args.training.batch_size
    
    dag_type = args.training.data_kwargs["dag_type"]
    max_o_dims = n_dims

    if args.training.data == "sde":
        from sde import get_sde_data_sampler
        data_sampler = get_sde_data_sampler(args.training, o_dims=max_o_dims,
                                            **args.training.data_kwargs)
    else:
        from dataset import get_data_sampler
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

    for i in pbar: 
        task_sampler_args = {}
        
        if args.training.data == "sde":
            assert "sde" in args.model.family

            parts = [f"data_{args.training.train_steps}"]
            if args.training.lamb != 5:
                parts.append(f"lambda_{int(args.training.lamb)}")
            if args.training.ode:
                parts.append("ode")
            if args.model.n_dims != 5:
                parts.append(f"{args.model.n_dims}dim")
            if args.training.diffusion != 20:
                parts.append(f"diffusion{args.training.diffusion}")
            if args.training.number_events != None:
                parts.append(f"{args.training.number_events}events")

            if args.training.train_steps % 2500 == 0:
                dataset_itr = i // 2500
                filename = f"sde/{'_'.join(parts)}_{dataset_itr}.pt"
                current_data = torch.load(filename)
                data = current_data[ (bsize * (i % 2500)) : (bsize * (i % 2500) + bsize), :, : ]
            else:
                filename = f"sde/{'_'.join(parts)}.pt"
                data = torch.load(filename)
            assert ((data.shape[1] - 1) / 4).is_integer(), f"data shape: {data.shape}"
            o_points = (data.shape[1] - 1) // 4

        else:
            o_points = curriculum.n_points
            data = data_sampler.complete_dataset(n_thetas = bsize,                
                                                o_points = o_points,
                                                o_vars = curriculum.n_vars_truncated,
                                                itr = i,
                                                continuation = continuation,
                                                block_setup=args.model.block_setup,
                                                transformation=args.training.transformation,
                                                constant_z = args.training.constant_z,
                                                randomize_labels= args.training.randomize_labels)

        task = task_sampler(**task_sampler_args)
        
        ess = data_sampler.ess

        loss_func = task.get_training_metric(training_loss=args.training.training_loss)

        if args.training.data == "sde": assert loss_func.__name__ == "masked_mean_squared_error"

        if dag_type not in ["only_parent", "one_parent", "any"]:
            print("Transformer not defined for this dag_type: modify models.py")
            raise NotImplementedError
        
        loss, output, gt = train_step(model, data.cuda(), o_vars = curriculum.n_vars_truncated,
                                      optimizer = optimizer, loss_func = loss_func, n_bins = n_bins)

        if dag_type == "only_parent" and args.training.data == "gaussian":
            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                if args.model.block_setup:
                    z_index = data_sampler.z_index
                    wandb.log(
                        {
                            "overall_loss": loss,
                            "effective_support_size": ess,
                            "n_points": o_points,
                            "z_index": z_index,
                            "n_dims": curriculum.n_dims_truncated,
                        },
                        step=i,
                    )
                else:
                    wandb.log({"overall_loss": loss,
                               "effective_support_size": ess,
                               "n_points": o_points,
                               "n_dims": curriculum.n_dims_truncated},
                               step=i)

        elif i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
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
    assert args.model.family in ["gpt2", "lstm", "gpt2_sde", "gpt2_ao", "gpt2_mlp", "rnn_sde", "rnn", "lstm_sde", "gru_sde", "gru", "gpt2_ao_sde", "gpt2_cont", "gpt2_ao_cont"]
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