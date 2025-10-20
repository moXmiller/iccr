print("Started")
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import statistics
from decimal import Decimal
import os
import csv
print("torch imported", flush = True)

from train import binning
from eval import get_model_from_run, read_run_dir
from dataset import get_data_sampler
from tasks import mean_squared_error, mean_absolute_error, masked_mean_squared_error, brier, rps, hhi, cross_entropy, r2
from sde import get_sde_data_sampler
from tqdm import tqdm
import random
print("functions imported")

def retrieve_model(args, continuation = True):
    run_dir = "/cluster/scratch/millerm/models_cf"
    conference = args.conference
    assert conference in ["neurips", "iclr"]
    if not os.path.exists(run_dir + "/" + conference):
        os.makedirs(run_dir + "/" + conference)
    # df = read_run_dir(run_dir)

    assert conference == "iclr"
    if conference == "iclr":
        run_id = nonnaive_retrieve_model(args)
    print(f"run_id: {run_id}")

    # task_name = "counterfactual_regression"
    run_path = run_dir + "/" + conference + "/" + run_id
    run_path = str(run_path)
    print(f"run_path: {run_path}")

    model, _ = get_model_from_run(run_path)
    
    print("model read in")
    return model


def model_to_device(model):
    if torch.cuda.is_available(): device = "cuda"
    else:
        device = "cpu"
    print(f"torch device: {device}")

    model.to(device)
    return model, device
    

def write_mse_sde(data_sampler, args):
    n_thetas = args.n_thetas
    o_vars = args.o_vars
    family = args.family
    ao = args.ao
    lamb, max_time = args.lamb, args.max_time
    n_points = args.n_points
    print("assigned arguments")

    model = retrieve_model(args, continuation = args.continuation)
    print("model retrieved")
    model, device = model_to_device(model)
    model.eval()
    print("model in eval mode")
    
    mean_row = {"stat": "mean", "family": family}
    if args.poisson == 1: opts_row = {"stat": "o_points", "family": family}

    all_means = []

    if args.conference == "neurips" or args.poisson: eval_steps = 1000
    elif args.conference == "iclr": eval_steps = 100

    for itr in tqdm(range(eval_steps)):
        xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, lamb, max_time, n_points, itr = itr, split = 'val', poisson = args.poisson, ode = args.ode, number_events = args.number_events)
        with torch.no_grad():
            pred, gt, mask = model(xs.to(device), o_vars = o_vars)
        mse_thetas = torch.zeros(n_thetas)
        for theta_idx in range(n_thetas):
            if args.conference == "neurips": 
                pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
                mse_thetas[theta_idx] = mean_squared_error(pred[theta_idx,:,:].cpu().detach(), gt[theta_idx,:,:].cpu().detach()) # theta_hat^*
            elif args.conference == "iclr": mse_thetas[theta_idx] = masked_mean_squared_error(pred[theta_idx,:,:].cpu().detach(), 
                                                                                              gt[theta_idx,:,:].cpu().detach(),
                                                                                              mask[theta_idx,:,:].cpu().detach()) # theta_hat^*

        mean_row[str(itr)] = torch.mean(mse_thetas).cpu().detach().item()
        all_means.append(torch.mean(mse_thetas).cpu().detach().item())
        if args.poisson == 1: opts_row[str(itr)] = data_sampler.o_points

    csv_field_names = [str(e) for e in range(eval_steps)]
    if args.poisson == 1: mse_rows = [mean_row, opts_row]
    else: 
        mse_rows = [mean_row]
        mean_row["mean"] = np.mean(all_means)
        csv_field_names = csv_field_names + ["mean"]
    csv_field_names = ["stat", "family"] + csv_field_names
    if args.conference == "neurips": filename = f"eval/mse/context_length_sde_{family}{'_ao' if ao else ''}.csv"
    elif args.conference == "iclr":
        parts = [f"eval/iclr/sde/", f"{str(args.train_steps)}"]
        if args.lamb != 5:
            parts.append(f"{int(args.lamb)}lamb")
        if args.ode:
            parts.append("ode")
        if args.o_dims != 5:
            parts.append(f"{args.o_dims}dim")
        if args.family != "gpt2_sde":
            parts.append(args.family)
        if args.poisson != 1:
            parts.append(f"poisson{args.poisson}")
        if args.n_layer != 8:
            parts.append(f"{args.n_layer}l")
        if args.n_head != 1:
            parts.append(f"{args.n_head}h")
        if args.diffusion != 20:
            parts.append(f"diffusion{args.diffusion}")
        if args.number_events != None:
            parts.append(f"{args.number_events}events")
        filename = "_".join(parts) + ".csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(mse_rows)
    print(f"Successfully written .csv file for {family}!")
    print(filename)


def perfectly_correlated_prediction(data_sampler, args):
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    o_dims = args.o_dims
    model_size = args.model_size
    ao = args.ao
    transformation = args.transformation
    continuation = args.continuation
    training_loss = args.training_loss
    constant_z = args.constant_z

    model = retrieve_model(args, continuation = continuation)

    model, device = model_to_device(model)
    model.eval()

    n_points = o_points

    eval_steps = 1

    for itr in tqdm(range(eval_steps)):
        mae_itr = torch.zeros(eval_steps)

        for x_cf in [1]:
            p = 34
            xs = data_sampler.complete_dataset(n_thetas, p, o_vars, itr = itr, split = 'val', continuation = continuation, transformation = args.transformation, constant_z = constant_z)
            
            correlated_xs = torch.ones_like(xs)
            correlated_xs[:, 0:int(p * o_vars):(o_vars * 2), :] = 3
            correlated_xs[:, 1:int(p * o_vars):(o_vars * 2), :] = 3
            correlated_xs[:, (-2):, :] = x_cf


            print(correlated_xs.shape)

            correlated_xs[:, -3, :] = xs[:, -3, :]
            
            z_index = data_sampler.z_index

            correlated_xs[:, int(o_vars * z_index + 3), :] = 20
            mu = correlated_xs[:, 1:int(o_vars * p): 2, :].mean()

            print(((correlated_xs[:, :int(o_vars * p): 2, :] - 2)*(correlated_xs[:, 1:int(o_vars * p): 2, :] - mu)).sum(dim = 1) / ((correlated_xs[:, :int(o_vars * p): 2, :] - 2)**2).sum(dim = 1))

            with torch.no_grad():
                pred, gt = model(correlated_xs.to(device), o_vars = o_vars)
            pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
            mae_thetas = torch.zeros(n_thetas)
            for theta_idx in range(n_thetas):
                mae_thetas[theta_idx] = (pred[theta_idx,:,:].cpu().detach() - gt[theta_idx,:,:].cpu().detach()).abs().mean()

            mae_itr[itr] = torch.mean(mae_thetas)

            print(f"predicted: {pred}")

        mean_mae = torch.mean(mae_itr).item()

    print(correlated_xs)

    print(pred.mean())
    print(f"Perfectly correlated prediction MAE: {mean_mae}")


def predict_bins(data_sampler, args):
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    o_dims = args.o_dims
    ao = args.ao
    n_bins = args.n_bins
    transformation = args.transformation
    continuation = args.continuation
    training_loss = args.training_loss
    constant_z = args.constant_z
    assert n_bins != 1
    if training_loss == "ce": training_loss = "cross_entropy"
    assert training_loss in ["brier", "rps", "cross_entropy"]

    model = retrieve_model(args, continuation = continuation)
    model, device = model_to_device(model)
    model.eval()

    n_points = o_points
    eval_steps = 100

    confusion = torch.zeros((n_bins, n_bins))

    brr = 0
    rpss = 0
    ce = 0
    hhii = 0
    right_probs = 0
    window_size = 1
    window_probs = 0

    for itr in tqdm(range(eval_steps)):
        xs = data_sampler.complete_dataset(n_thetas, n_points, o_vars, itr = itr, split = 'val', continuation = continuation, transformation = args.transformation, constant_z = constant_z, block_setup = args.block_setup)

        with torch.no_grad():
           pred, gt = model(xs.to(device), o_vars = o_vars)

        gt_bins, gt_centers = binning(n_bins, gt)

        one_hot_bins = torch.nn.functional.one_hot(gt_bins.view(-1).long(), num_classes = n_bins).float()

        softmaxxx = torch.nn.functional.softmax(pred.reshape(-1, n_bins).cuda(), dim = 1)

        assert softmaxxx.shape == (n_thetas, n_bins)

        for theta_idx in range(n_thetas):
            right_probs += softmaxxx[theta_idx, gt_bins[theta_idx]].cpu().detach().item()
            window_probs += softmaxxx[theta_idx, (gt_bins[theta_idx] - window_size) : (gt_bins[theta_idx] + window_size + 1)].sum().cpu().detach().item()

        ce += cross_entropy(gt_bins.view(-1), pred.reshape(-1, n_bins).cuda())
        hhii += hhi(pred.reshape(-1, n_bins).cuda())
        brr += brier(one_hot_bins, pred.reshape(-1, n_bins).cuda())
        rpss += rps(one_hot_bins, pred.reshape(-1, n_bins).cuda())

        for theta_idx in range(n_thetas):
            maxx = pred[theta_idx, :, :].argmax(dim = 1)[0]
            gt_max = gt_bins[theta_idx]
            confusion[maxx, gt_max] += 1

    confusion = confusion / (eval_steps * n_thetas)
    b = (brr / eval_steps).item()
    print(f"brier: {b}")
    r = (rpss / eval_steps).item()
    print(f"rps: {r}")
    c = (ce / eval_steps).item()
    print(f"cross_entropy: {c}")
    h = (hhii / eval_steps).item()
    print(f"hhi: {h}")
    right_probs = right_probs / (eval_steps * n_thetas)
    print(f"right_probs: {right_probs}")
    window_probs = window_probs / (eval_steps * n_thetas)
    print(f"window_probs (+-{window_size}): {window_probs}")

    conf_indices = (n_bins // 2 - 3), (n_bins // 2 + 3)
    conf_center = confusion[conf_indices[0]:conf_indices[1], conf_indices[0]:conf_indices[1]]
    print({conf_center})
    coverage = conf_center.sum().item()
    print(f"confusion center coverage {coverage}")
    accuracy = confusion.trace().item()
    print(f"accuracy: {accuracy}")

    csv_field_names = ["n_bins", "block_setup", "brier", "rps", "cross_entropy", "hhi", "right_probs", f"window_probs (+-{window_size})", "confusion_center_coverage (+-3)", "accuracy"]
    csv_row = {"n_bins": args.n_bins, "block_setup": args.block_setup, "brier": b, "rps": r, "cross_entropy": c, "hhi": h, "right_probs": right_probs, f"window_probs (+-{window_size})": window_probs, "confusion_center_coverage (+-3)": coverage, "accuracy": accuracy}

    parts = [f"eval/{args.conference}/proper/statistics"]
    parts.append(args.training_loss)
    if not args.block_setup:
        parts.append("noblock")
    assert args.constant_z == 19
    assert args.min_examples == 30
    assert args.o_dims == 1

    filename = "_".join(parts) + ".csv"

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        if os.stat(filename).st_size == 0:
            writer.writeheader() 
        writer.writerow(csv_row)

    return confusion


def write_mse(data_sampler, args):
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    o_dims = args.o_dims
    family = args.family
    ao = args.ao
    theta_dist = args.theta_dist
    eval_theta_dist = args.eval_theta_dist
    transformation = args.transformation
    train_steps = args.train_steps
    continuation = args.continuation
    predict_y = args.predict_y
    predict_x = args.predict_x
    predict_beta = args.predict_beta
    learning_rate = args.learning_rate
    training_loss = args.training_loss
    n_bins = args.n_bins
    constant_z = args.constant_z
    with_pe = args.with_pe
    min_examples = args.min_examples
    disabled_layers = args.disabled_layers

    model = retrieve_model(args, continuation = continuation)
    
    assert disabled_layers == -1
    assert not predict_x and not predict_beta and not predict_y
    
    model, device = model_to_device(model)
    model.eval()

    n_points = o_points
    
    mean_row = {"stat": "mean", "family": family, "attention_only": ao}
    q025_row = {"stat": "q025", "family": family, "attention_only": ao}
    q975_row = {"stat": "q975", "family": family, "attention_only": ao}

    eval_steps = 100

    if training_loss in ["brier", "rps"]:
        assert n_bins != 1
        _, bin_centers = binning(n_bins, None)

    for p in tqdm(range(min_examples, n_points + 1)):
        mse_itr = torch.zeros(eval_steps)
        for itr in range(eval_steps):

            xs = data_sampler.complete_dataset(n_thetas, p, o_vars, itr = itr, split = 'val', continuation = continuation, transformation = args.transformation,
                                               block_setup = args.block_setup, constant_z = constant_z)
            with torch.no_grad():
                pred, gt = model(xs.to(device), o_vars = o_vars)
            if training_loss in ["brier", "rps"]:
                pred = torch.softmax(pred, dim = 2)
                pred_bins = pred.argmax(dim = 2)
                pred = bin_centers[pred_bins]

            pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
            mse_thetas = torch.zeros(n_thetas)
            for theta_idx in range(n_thetas):
                if args.eval_loss == "mae": mse_thetas[theta_idx] = mean_absolute_error(pred[theta_idx,:,:].cpu().detach(), gt[theta_idx,:,:].cpu().detach())
                else: mse_thetas[theta_idx] = mean_squared_error(pred[theta_idx,:,:].cpu().detach(), gt[theta_idx,:,:].cpu().detach()) # theta_hat^*
            mse_itr[itr] = torch.mean(mse_thetas)
        
        quantile_indices = random.choices(range(eval_steps), k = eval_steps)
        quantiles = torch.tensor(np.quantile(mse_itr[quantile_indices].numpy(), q=[0.025, 0.975]))

        mean_mse = torch.mean(mse_itr).item()
        lower_q = max(0, 2 * mean_mse - torch.mean(quantiles[1]).item())
        upper_q = max(0, 2 * mean_mse - torch.mean(quantiles[0]).item())

        mean_row[str(p)] = mean_mse
        q025_row[str(p)] = lower_q
        q975_row[str(p)] = upper_q

    mse_rows = [mean_row, q025_row, q975_row]
    csv_field_names = [str(p) for p in range(2, n_points + 1)]
    csv_field_names = ["stat", "family", "attention_only"] + csv_field_names

    if ao: assert "ao" in family

    parts = [f"context_length"]
    if args.family != "gpt2":
        parts.append(args.family)
    if args.n_layer != 8:
        parts.append(f"{args.n_layer}l")
    if args.n_head != 1:
        parts.append(f"{args.n_head}h")
    if args.n_embd != 256:
        parts.append(f"{args.n_embd}embd")
    if continuation:
        parts.append("cont")
    if predict_y:
        parts.append("y")
    if predict_x:
        parts.append("x")
    if predict_beta:
        parts.append("beta")
    if transformation != "addlin":
        parts.append(transformation)
    if o_dims != 5:
        parts.append(str(o_dims))
    if theta_dist == "norm":
        parts.append("normtrain")
    if eval_theta_dist == "norm":
        parts.append("normeval")
    if args.diversity <= 1000:
        parts.append(f"dvrsty{args.diversity}")
    if not args.block_setup:
        parts.append("noblock")
    if n_bins != 1:
        parts.append(f"{n_bins}bins")
    if learning_rate != 0.0001:
        lr = abs(Decimal(str(learning_rate)).as_tuple().exponent)
        parts.append(f"lr{lr}")
    if training_loss != "mse":
        parts.append(f"tloss{training_loss}")
    if args.eval_loss != "mse":
        parts.append(f"eloss{args.eval_loss}")
    if constant_z != -1:
        parts.append("constz" + str(constant_z))
    if not with_pe:
        parts.append("nopos")
    if min_examples != 2:
        parts.append("minex" + str(min_examples))
    if train_steps != 50000:
        parts.append(f"{train_steps}steps")
    if disabled_layers != -1:
        enabled_layers = "".join(enabled_layers)
        parts.append(f"enabled_{enabled_layers}")
    if args.randomize_labels:
        parts.append("randlabels")

    filename = f"eval/mse_new/{'_'.join(parts)}.csv"
    if args.conference == "iclr": filename = filename.replace("eval/mse_new", "eval/iclr/mse")

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(mse_rows)
    print(f"Successfully written .csv file for {family}!\n{filename}")


def write_mse_complexity(data_sampler, args):
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    model_size = args.model_size
    ao = args.ao
    theta_dist = args.theta_dist
    transformation = args.transformation
    train_steps = args.train_steps
    
    assert args.dag_type == "any"
    
    model = retrieve_model(args, continuation = False)
    
    model, device = model_to_device(model)
    model.eval()
    
    n_points = o_points
    
    mean_row = {"stat": "mean", "model_size": model_size, "attention_only": ao}
    dags_row = {"stat": "DAG", "model_size": model_size, "attention_only": ao}

    eval_steps = 1000
    for p in range(1, n_points + 1):
    # p = random.choice(range(1, n_points + 1))
        for itr in range(eval_steps):
            xs = data_sampler.complete_dataset(n_thetas, p, o_vars, itr = itr, split = 'val', transformation = args.transformation, block_setup = False)
            with torch.no_grad():
                pred, gt = model(xs.to(device), o_vars = o_vars)
            pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
            mse_thetas = torch.zeros(n_thetas)

            for theta_idx in range(n_thetas):
                mse_thetas[theta_idx] = mean_squared_error(pred[theta_idx,:,:].cpu().detach(), gt[theta_idx,:,:].cpu().detach()) # theta_hat^*
            mean_row[str(itr)] = torch.mean(mse_thetas)
            dags_row[str(itr)] = data_sampler.concat_indices
            # compute mean for iteration
            # group by DAG realization and take again the mean
            # add row as column to the final dataset which has DAGs as rows and sequence length and columns
        
    mse_rows = [mean_row, dags_row]
    csv_field_names = [str(e) for e in range(eval_steps)]
    csv_field_names = ["stat", "model_size"] + csv_field_names
    filename = f"eval/mse/context_length_complexity_{model_size}{'_ao' if ao else ''}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(mse_rows)
    print(f"Successfully written .csv file for {model_size}!")


def write_ess_file(sampler_seed = 6052025, filepath = "eval/wandb/iclr_ess.csv"):
    from dataset import effective_support_size
    csv_field_names = ["distribution", "diversity", "ess"]
    for dist in ["norm", "uniform"]:
        for diversity in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000]:
            ess, _, _ = effective_support_size(diversity, dist = dist, seed = sampler_seed, lwr = -6, upr = 6)
            row = {"distribution": dist, "diversity": diversity, "ess": ess.item()}
            with open(filepath, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
                if os.stat(filepath).st_size == 0:
                    writer.writeheader() 
                writer.writerow(row)
            print(dist, diversity, ess)


def extract_residual_stream(data_sampler, args):
    assert args.conference == "iclr"
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    continuation = args.continuation
    min_examples = args.min_examples

    model = retrieve_model(args, continuation = continuation)
    
    model, device = model_to_device(model)
    model.eval()
    
    n_points = o_points

    probing_dir = "/cluster/scratch/millerm/datasets_cf/iclr/probing/"
    
    assert args.ao
    assert args.n_head == 1, "adapt residuals vector, see for loop below"
    
    residuals = {hidden_idx: [] for hidden_idx in range(args.n_layer + 1)}   # 8 + 1 including the embedding dimension
    probes = []
    if args.data == "gaussian":
        eval_steps = 100
        for itr in range(eval_steps):
            p = random.choice(range(min_examples, n_points + 1))
            xs = data_sampler.complete_dataset(n_thetas, p, o_vars, itr = itr, split = 'val', continuation = continuation, transformation = args.transformation, block_setup = args.block_setup,
                                            constant_z = args.constant_z, randomize_labels = args.randomize_labels)
            
            with torch.no_grad():
                # _, _, output = model(xs.to(device), o_vars = o_vars, final_hidden_state = True)
                _, _, outputs, _ = model(xs.to(device), o_vars = o_vars, output_hidden_states = True, final_hidden_state = True)
            output = outputs[-1]
            output = output[:, -2, :]
            assert len(output.shape) == 2, "number of dimensions of final hidden state must match probe"
            if args.probe_type == "us": probe = data_sampler.us_probe
            elif args.probe_type == "theta": 
                if args.o_dims == 1: probe = data_sampler.theta_b[:, :, 0, 0]
                else: probe = data_sampler.theta_b[:, 0, 0, :]
            elif args.probe_type == "beta": probe = data_sampler.beta_b

            for hidden_idx in range(args.n_layer + 1):
                counterfactual_state = outputs[hidden_idx][:, -2, :]
                residuals[hidden_idx].append(counterfactual_state)
            # assert outputs == 4
            # if itr == 0: print(f"probe shape at itr 0: {probe.shape}")
            probes.append(probe)

    elif args.data == "sde":
        eval_steps = 800
        for itr in range(eval_steps):
            p = random.choice(range(min_examples, n_points + 1))
            xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, args.lamb, args.max_time, n_points, itr = itr, split = 'val', poisson = args.poisson, ode = args.ode, number_events = args.number_events)
            
            with torch.no_grad():
                _, _, outputs, _ = model(xs.to(device), o_vars = o_vars, output_hidden_states = True)
            output = outputs[-1]
            output = output[:, 24, :]   # 0 - 21 are examples, 22 indicator, 23 - 24 initial configuration
            assert len(output.shape) == 2, "number of dimensions of final hidden state must match probe"
            if args.probe_type == "alpha": probe = data_sampler.alpha
            elif args.probe_type == "beta": probe = data_sampler.beta
            elif args.probe_type == "gamma": probe = data_sampler.gamma
            elif args.probe_type == "delta": probe = data_sampler.delta

            probe = probe.unsqueeze(-1)

            for hidden_idx in range(args.n_layer + 1):
                counterfactual_state = outputs[hidden_idx][:, -2, :]
                # print("cf shape", counterfactual_state.shape)
                residuals[hidden_idx].append(counterfactual_state)
            # assert outputs == 4
            # if itr == 0: print(f"probe shape at itr 0: {probe.shape}")
            probes.append(probe)

    probes_full = torch.cat(probes, dim = 0).cpu()
    # print(f"probes shape: {probes_full.shape}")
    datasets = []
    evalsets = []
    datasets_diff = []
    evalsets_diff = []
    full_size = probes_full.shape[0]
    train_split = random.sample(range(full_size), k = int(0.8 * full_size))
    test_split = list(set(range(full_size)) - set(train_split))
    probes_train = probes_full[train_split, :]
    probes_eval = probes_full[test_split, :]
    # print(f"probes_train shape: {probes_train.shape}")
    # print(f"probes_eval shape: {probes_eval.shape}")

    for hidden_idx in range(args.n_layer + 1):
        residuals_full = torch.cat(residuals[hidden_idx], dim = 0).cpu()
        if hidden_idx >= 1:
            residuals_diff = residuals_full - torch.cat(residuals[hidden_idx - 1], dim = 0).cpu()
            residuals_diff_train = residuals_diff[train_split]
            residuals_diff_eval = residuals_diff[test_split]
            dataset_diff = TensorDataset(residuals_diff_train, probes_train)
            evalset_diff = TensorDataset(residuals_diff_eval, probes_eval)
            df_diff = pd.DataFrame(torch.cat([residuals_diff_train, probes_train], dim = 1).detach().numpy())
            eval_df_diff = pd.DataFrame(torch.cat([residuals_diff_eval, probes_eval], dim = 1).detach().numpy())
            df_diff.columns = eval_df_diff.columns = [f"res_{i}" for i in range(residuals_diff.shape[1])] + [f"probe_{i}" for i in range(data_sampler.o_dims)]
        residuals_train = residuals_full[train_split]
        residuals_eval = residuals_full[test_split]
        dataset = TensorDataset(residuals_train, probes_train)
        evalset = TensorDataset(residuals_eval, probes_eval)
        df = pd.DataFrame(torch.cat([residuals_train, probes_train], dim = 1).detach().numpy())
        eval_df = pd.DataFrame(torch.cat([residuals_eval, probes_eval], dim = 1).detach().numpy())
        df.columns = eval_df.columns = [f"res_{i}" for i in range(output.shape[1])] + [f"probe_{i}" for i in range(data_sampler.o_dims)]
        parts = ["residual_probe_dataset", f"layer_{hidden_idx}"]
        if args.data == "gaussian":
            parts.append(args.probe_type)
            if args.o_dims != 1:
                parts.append(str(args.o_dims) + "dim")
            if args.constant_z != -1:
                parts.append("constz" + str(args.constant_z))
            if args.min_examples != 2:
                parts.append("minex" + str(args.min_examples))
            if args.n_embd != 256:
                parts.append(f"{args.n_embd}embd")
            if args.n_layer != 8:
                parts.append(f"{args.n_layer}l")
            if args.train_steps != 50000:
                parts.append(f"{args.train_steps}steps")

        elif args.data == "sde":
            parts.append(str(args.train_steps))
            parts.append(args.family)
            parts.append(args.probe_type)
            if args.lamb != 5:
                parts.append(f"{int(args.lamb)}lamb")
            if args.ode:
                parts.append("ode")
            if args.o_dims != 5:
                parts.append(f"{args.o_dims}dim")
            parts.append(args.family)
            if args.poisson != 1:
                parts.append(f"poisson{args.poisson}")
            if args.n_layer != 8:
                parts.append(f"{args.n_layer}l")
            if args.n_head != 1:
                parts.append(f"{args.n_head}h")
            if args.number_events != None:
                parts.append(f"{args.number_events}events")

        filename = "_".join(parts) + ".csv"
        df.to_csv(probing_dir + filename, index=False)
        parts.append("eval")
        filename = "_".join(parts) + ".csv"
        eval_df.to_csv(probing_dir + filename, index=False)
        datasets.append(dataset)
        evalsets.append(evalset)
        if hidden_idx >= 1:
            parts[0] = "residual_diff_probe_dataset"
            filename = "_".join(parts[:-1]) + ".csv"
            df_diff.to_csv(probing_dir + filename, index=False)
            filename = "_".join(parts) + ".csv"
            eval_df_diff.to_csv(probing_dir + filename, index=False)
            datasets_diff.append(dataset_diff)
            evalsets_diff.append(evalset_diff)
    return datasets, evalsets, datasets_diff, evalsets_diff


def train_probe(dataset, evalset, args, hidden_idx = None, probe_diff = False):
    assert args.conference == "iclr"
    if args.data == "gaussian": assert args.probe_type in ["us", "theta", "beta"]
    elif args.data == "sde": assert args.probe_type in ["alpha", "beta", "gamma", "delta"]
    r_shape, probe_shape = len(dataset[0][0]), len(dataset[0][1])

    # print(f"r_shape: {r_shape}, probe_shape: {probe_shape}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Linear(r_shape, probe_shape)

    assert probe_shape == args.o_dims

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = mean_absolute_error

    if args.data == "gaussian": eval_steps = 100
    elif args.data == "sde": eval_steps = 800
    for _ in range(eval_steps):
        for r_batch, probe_batch in dataloader:
            pred = model(r_batch)
            loss = loss_fn(pred, probe_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("\n")
    if not probe_diff: print(f"train loss after layer {hidden_idx}: {loss.item()}")
    else: print(f"train loss difference between layers {hidden_idx} and {hidden_idx - 1}: {loss.item()}")

    model.eval()

    evalloader = DataLoader(evalset, batch_size=32, shuffle=False)
    eval_losses = []

    p_r2 = len(evalset[0][0])
    n_r2 = len(evalset)

    assert p_r2 == args.n_embd
    assert n_r2 == 1280
    # elif args.data == "sde": assert n_r2 == 320

    if probe_shape > 1:
        rss = torch.zeros(probe_shape)
        tss = torch.zeros(probe_shape)
    else:
        rsss = []
        tsss = []

    for r_batch, probe_batch in evalloader:
        pred = model(r_batch)
        loss = loss_fn(pred, probe_batch)
        if probe_shape == 1:
            rss = ((probe_batch - pred) ** 2).sum()
            tss = ((probe_batch - probe_batch.mean()) ** 2).sum()
            rsss.append(rss.item())
            tsss.append(tss.item())
        else:
            rs = ((probe_batch - pred) ** 2)
            ts = ((probe_batch - probe_batch.mean(dim = 0, keepdim = True)) ** 2)
            assert rs.shape[1] == probe_shape
            assert ts.shape[1] == probe_shape
            rss += rs.sum(dim = 0)
            tss += ts.sum(dim = 0)

        eval_losses.append(loss.item())

    eval_loss = torch.tensor(eval_losses).mean()



        
    if probe_shape == 1:
        rsquared = 1 - (sum(rsss) / sum(tsss))
        adj_rsquared = 1 - ((1 - rsquared) * (n_r2 - 1) / (n_r2 - p_r2 - 1))
    else:
        elmnt_rsquared = 1 - (rss / tss)
        if hidden_idx == 0: print(f"final rss {rss}\nfinal tss {tss}")
        elmnt_adj_rsquared = 1 - ((1 - elmnt_rsquared) * (n_r2 - 1) / (n_r2 - p_r2 - 1))
        rsquared = elmnt_rsquared.mean().item()
        adj_rsquared = elmnt_adj_rsquared.mean().item()
    if not probe_diff:
        print(f"eval loss after layer {hidden_idx}: {eval_loss.item()}")
        print(f"eval R squared after layer {hidden_idx}: {rsquared}")
        print(f"eval adjusted R squared after layer {hidden_idx}: {adj_rsquared}")
        if probe_shape > 1:
            print(f"elementwise R squared after layer {hidden_idx}: {elmnt_rsquared}")
            print(f"elementwise adjusted R squared after layer {hidden_idx}: {elmnt_adj_rsquared}")
    else:
        print(f"eval loss difference between layers {hidden_idx} and {hidden_idx - 1}: {eval_loss.item()}")
        print(f"eval R squared difference between layers {hidden_idx} and {hidden_idx - 1}: {rsquared}")
        print(f"eval adjusted R squared difference between layers {hidden_idx} and {hidden_idx - 1}: {adj_rsquared}")
        if probe_shape > 1:
            print(f"elementwise R squared difference between layers {hidden_idx} and {hidden_idx - 1}: {elmnt_rsquared}")
            print(f"elementwise adjusted R squared difference between layers {hidden_idx} and {hidden_idx - 1}: {elmnt_adj_rsquared}")

    statsdir = "eval/iclr/probing/"
    parts = ["statistics"] if not probe_diff else ["statistics_diff"]
    if args.data == "gaussian":
        parts.append(args.probe_type)
        if args.o_dims != 1: 
            parts.append(str(args.o_dims) + "dim")
        if args.constant_z != -1:
            parts.append("constz" + str(args.constant_z))
        if args.min_examples != 2:
            parts.append("minex" + str(args.min_examples))
        if args.n_embd != 256:
            parts.append(f"{args.n_embd}embd")
        if args.n_layer != 8:
            parts.append(f"{args.n_layer}l")
        if args.train_steps != 50000:
            parts.append(f"{args.train_steps}steps")

    elif args.data == "sde":
        parts.append(str(args.train_steps))
        parts.append(args.family)
        parts.append(args.probe_type)
        if args.lamb != 5:
            parts.append(f"{int(args.lamb)}lamb")
        if args.ode:
            parts.append("ode")
        if args.o_dims != 5:
            parts.append(f"{args.o_dims}dim")
        if args.poisson != 1:
            parts.append(f"poisson{args.poisson}")
        if args.n_layer != 8:
            parts.append(f"{args.n_layer}l")
        if args.n_head != 1:
            parts.append(f"{args.n_head}h")
        if args.number_events != None:
            parts.append(f"{args.number_events}events")

    filename = "_".join(parts) + ".csv"
    stats_file = statsdir + filename

    if not os.path.exists(stats_file):
        with open(stats_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hidden_idx", "eval_loss", "r_squared", "adj_r_squared"])

    with open(stats_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([hidden_idx, eval_loss.item(), rsquared, adj_rsquared])

    weight = pd.DataFrame(model.weight.cpu().detach().numpy())
    bias = pd.DataFrame(model.bias.cpu().detach().numpy())
    probing_dir = "/cluster/scratch/millerm/datasets_cf/iclr/probing/"

    if not probe_diff: parts = ["weight"]
    else: parts = ["weight_diff"]
    if hidden_idx is not None:
        parts.append(f"{hidden_idx}")

    if args.data == "gaussian":
        parts.append(args.probe_type)
        if args.o_dims != 1:
            parts.append(str(args.o_dims) + "dim")
        if args.constant_z != -1:
            parts.append("constz" + str(args.constant_z))
        if args.min_examples != 2:
            parts.append("minex" + str(args.min_examples))
        if args.n_embd != 256:
            parts.append(f"{args.n_embd}embd")
        if args.n_layer != 8:
            parts.append(f"{args.n_layer}l")
        if args.train_steps != 50000:
            parts.append(f"{args.train_steps}steps")

    elif args.data == "sde":
        parts.append(str(args.train_steps))
        parts.append(args.family)
        parts.append(args.probe_type)
        if args.lamb != 5:
            parts.append(f"{int(args.lamb)}lamb")
        if args.ode:
            parts.append("ode")
        if args.o_dims != 5:
            parts.append(f"{args.o_dims}dim")
        if args.poisson != 1:
            parts.append(f"poisson{args.poisson}")
        if args.n_layer != 8:
            parts.append(f"{args.n_layer}l")
        if args.n_head != 1:
            parts.append(f"{args.n_head}h")
        if args.number_events != None:
            parts.append(f"{args.number_events}events")

    filename = "_".join(parts) + ".csv"
    weight.to_csv(probing_dir + filename, index = False)
    
    if not probe_diff: parts[0] = "bias"
    else: parts[0] = "bias_diff"
    filename = "_".join(parts) + ".csv"
    bias.to_csv(probing_dir + filename, index = False)
    return eval_loss.item(), rsquared, adj_rsquared


def subtract_probes(data_sampler, args):
    assert args.conference == "iclr"
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    continuation = args.continuation
    min_examples = args.min_examples

    probing_idx = args.probing_idx

    model = retrieve_model(args, continuation = continuation)

    model, device = model_to_device(model)
    model.eval()

    n_points = o_points

    # assert args.n_layer == 8, "adapt residuals vector, see for loop below"
    assert args.ao == 1, "adapt residuals vector, see for loop below"
    assert args.n_head == 1, "adapt residuals vector, see for loop below"

    eval_steps = 1
    for itr in range(eval_steps):
        p = random.choice(range(min_examples, n_points + 1))
        if args.data == "gaussian":
            xs = data_sampler.complete_dataset(n_thetas, p, o_vars, itr = 9597, split = 'val', continuation = continuation, transformation = args.transformation, block_setup = args.block_setup,
                                               constant_z = args.constant_z, randomize_labels = args.randomize_labels)
            with torch.no_grad(): pred, gt, _, _ = model(xs.to(device), o_vars = o_vars, output_hidden_states = True, final_hidden_state = True)
        
        elif args.data == "sde":
            xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, args.lamb, args.max_time, n_points, itr = itr, split = 'val', poisson = args.poisson, ode = args.ode, number_events = args.number_events)
            with torch.no_grad(): pred, gt, _, _ = model(xs.to(device), o_vars = o_vars, output_hidden_states = True)
            
    print((pred - gt).square().mean().item())
    print(pred.squeeze().std().item())


def variance_extensions(data_sampler, args):
    eval_steps = 100
    n_thetas = args.n_thetas
    o_vars = args.o_vars
    assert args.n_thetas == 64
    assert args.o_vars == 2
    o_points = 1
    transformation = args.transformation

    finals = []
    for step in tqdm(range(eval_steps)):
        xs = data_sampler.complete_dataset(n_thetas, o_points, o_vars, transformation = transformation, itr = step, split = 'val')
        final = xs[:,-1,:].tolist()[0]
        finals.append(final)
    
    return torch.var(torch.tensor(finals))


def write_attentions(data_sampler, args):
    family = args.family

    n_layer, n_head = args.n_layer, args.n_head
    
    assert args.predict_x == 0 and args.predict_beta == 0
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars

    ao = args.ao
    if ao: assert "ao" in family
    position = args.position
    itr = args.itr
    predict_y = args.predict_y
    continuation = args.continuation

    model = retrieve_model(args, continuation = continuation)
    
    model, device = model_to_device(model)

    if args.data == "gaussian":
        xs = data_sampler.complete_dataset(n_thetas, o_points, o_vars, itr = itr, split = 'val', continuation=continuation, block_setup = args.block_setup, 
                                           transformation = args.transformation, constant_z = args.constant_z, randomize_labels = args.randomize_labels)

    elif args.data == "sde": 
        if continuation: raise NotImplementedError
        lamb, max_time, n_points = args.lamb, args.max_time, args.n_points
        xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, lamb, max_time, n_points, itr = itr, split = 'val', poisson = args.poisson, ode = args.ode, number_events = args.number_events)
    with torch.no_grad():
        if args.data == "gaussian":
            _, _, attentions = model(xs.to(device), o_vars = o_vars, output_attentions = True)
            
            if args.train_steps == 1000000:
                z_index_file = f"eval/{args.conference}/attentions/{args.data}/z_index_file.csv"
                if not os.path.exists(z_index_file):
                    with open(z_index_file, mode="w", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=["itr", "z", "Y_z"])
                        writer.writeheader()
                row = {"itr": itr, "z": data_sampler.z_index, "Y_z": (data_sampler.z_index * 2 + 1)}
                with open(z_index_file, mode="a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["itr", "z", "Y_z"])
                    writer.writerow(row)
        else: _, _, _, attentions = model(xs.to(device), o_vars = o_vars, output_attentions = True)
    
    if args.conference == "neurips":
        for l in range(0, n_layer):
            print("layer", l)
            for h in range(0, n_head):
                if position == -1: 
                    att = torch.mean(attentions[l][:,h,:,:], dim = 0)
                    att = pd.DataFrame(att.cpu().detach().numpy())
                else: att = pd.DataFrame(attentions[l][position,h,:,:].cpu().detach().numpy())
                if position == -1:
                    attention_path = f"eval/attentions/attentions_{family}_{l}_layer_{h}_head{'_ao' if ao else ''}_{itr}_m{'_cont' if continuation else ''}{'_y' if predict_y else ''}{'_sde' if args.data == 'sde' else ''}.csv"
                else:
                    attention_path = f"eval/attentions/attentions_{family}_{l}_layer_{h}_head{'_ao' if ao else ''}_itr{itr}_pos{position}{'_cont' if continuation else ''}{'_y' if predict_y else ''}{'_sde' if args.data == 'sde' else ''}.csv"            
                att.to_csv(attention_path, index = False)
    
    elif args.conference == "iclr":
        dir_path = f"eval/iclr/attentions/{args.data}/{args.n_layer}layers{args.n_head}heads/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # for l in range(0, n_layer):
        #     for h in range(0, n_head):
        for l in range(6, 7):
            for h in range(5, 6):
                parts = [dir_path + "attn"]
                if position == -1:
                    att = torch.mean(attentions[l][:,h,:,:], dim = 0)
                    att = pd.DataFrame(att.cpu().detach().numpy())
                else: att = pd.DataFrame(attentions[l][position,h,:,:].cpu().detach().numpy())
                if args.n_layer != 8:
                    parts.append(f"{args.n_layer}layers")
                if args.n_head != 1:
                    parts.append(f"{args.n_head}heads")
                if args.ao:
                    parts.append("ao")
                if position != -1:
                    parts.append(f"pos{position}")

                if args.data == "sde":
                    if args.lamb != 5:
                        parts.append(f"{int(args.lamb)}lamb")
                    if args.ode:
                        parts.append("ode")
                    if args.o_dims != 5:
                        parts.append(f"{args.o_dims}dim")
                    if args.family != "gpt2_sde":
                        parts.append(args.family)
                    if args.poisson != 1:
                        parts.append(f"poisson{args.poisson}")
                    if args.number_events != None:
                        parts.append(f"{args.number_events}events")
                else:
                    if args.family != "gpt2":
                        parts.append(args.family)
                    if args.n_embd != 256:
                        parts.append(f"{args.n_embd}embd")
                    if continuation:
                        parts.append("cont")
                    if args.transformation != "addlin":
                        parts.append(args.transformation)
                    if o_dims != 5:
                        parts.append(str(o_dims))
                    if args.theta_dist == "norm":
                        parts.append("normtrain")
                    if args.diversity <= 1000:
                        parts.append(f"dvrsty{args.diversity}")
                    if not args.block_setup:
                        parts.append("noblock")
                    if args.n_bins != 1:
                        parts.append(f"{args.n_bins}bins")
                    if args.learning_rate != 0.0001:
                        lr = abs(Decimal(str(args.learning_rate)).as_tuple().exponent)
                        parts.append(f"lr{lr}")
                    if args.training_loss != "mse":
                        parts.append(f"tloss{args.training_loss}")
                    if args.constant_z != -1:
                        parts.append("constz" + str(args.constant_z))
                    if args.min_examples != 2:
                        parts.append("minex" + str(args.min_examples))
                    if args.train_steps != 50000:
                        parts.append(f"{args.train_steps}steps")
                    if args.randomize_labels:
                        parts.append("randlabels")
                parts.append(f"l{l}")
                parts.append(f"h{h}")
                parts.append(f"itr{itr}")
                attention_path = "_".join(parts) + ".csv"
                att.to_csv(attention_path, index = False)
    print(f"attentions successfully written to .csv for {family}")
    print(attention_path)


def sde_data_for_plot(data_sampler, args):
    n_thetas = args.n_thetas
    o_vars = args.o_vars
    family = args.family
    ao = args.ao
    itr = args.itr
    dimension_idx = args.dim_index
    lamb, max_time = args.lamb, args.max_time
    position = args.position
    if args.position == -1: 
        position = n_thetas - 1
        print("Last element selected")
    n_points = args.n_points
    print("assigned arguments")

    model = retrieve_model(args, continuation = False)
    print("model retrieved")
    model, device = model_to_device(model)
    model.eval()
    print("model in eval mode")

    obs_x_row = {"quantity": "obs_x"}
    obs_y_row = {"quantity": "obs_y"}
    pred_x_row = {"quantity": "pred_x"}
    pred_y_row = {"quantity": "pred_y"}
    gt_x_row = {"quantity": "gt_x"}
    gt_y_row = {"quantity": "gt_y"}
    evnt_t_row = {"quantity": "event_times"}

    xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, lamb, max_time, n_points, itr = itr, split = 'val', poisson = args.poisson, ode = args.ode, number_events = args.number_events)
    print(xs.shape)
    with torch.no_grad():
        pred, gt, mask = model(xs.to(device), o_vars = o_vars)
    # include x_init_cf, y_init_cf

    event_times = data_sampler.event_times
    o_points = data_sampler.o_points

    print(pred[:40])

    print(len(pred))

    assert pred.shape[0] == n_thetas
    assert pred.shape[2] == args.o_dims
    assert pred.shape == gt.shape

    cf_init = xs[:, (o_points * o_vars + 1):(o_points * o_vars + o_vars + 1), :].to(device)
    pred = torch.cat([cf_init, pred], dim=1)
    gt = torch.cat([cf_init, gt], dim = 1)

    xs_obs = xs[:, :(o_points * o_vars), :].to(device)
    obs_x = xs_obs[position, ::2, dimension_idx]
    obs_y = xs_obs[position, 1::2, dimension_idx]

    pred_x = pred[position, ::2, dimension_idx]
    pred_y = pred[position, 1::2, dimension_idx]
    gt_x = gt[position, ::2, dimension_idx]
    gt_y = gt[position, 1::2, dimension_idx]

    for p in range(o_points):
        obs_x_row[str(p)] = obs_x[p].cpu().detach().numpy()
        obs_y_row[str(p)] = obs_y[p].cpu().detach().numpy()
        pred_x_row[str(p)] = pred_x[p].cpu().detach().numpy()
        pred_y_row[str(p)] = pred_y[p].cpu().detach().numpy()
        gt_x_row[str(p)] = gt_x[p].cpu().detach().numpy()
        gt_y_row[str(p)] = gt_y[p].cpu().detach().numpy()
        evnt_t_row[str(p)] = event_times[p].cpu().detach().numpy()


    csv_field_names = [str(p) for p in range(o_points)]
    csv_field_names = ["quantity"] + csv_field_names
    if args.conference == "neurips": filename = f"eval/mse/prediction_chart_sde_{family}{'_ao' if ao else ''}.csv"
    elif args.conference == "iclr":
        parts = ["eval/iclr/sde/prediction/chart", str(args.train_steps)]
        if args.lamb != 5:
            parts.append(f"{int(args.lamb)}lamb")
        if args.ode:
            parts.append("ode")
        if args.o_dims != 5:
            parts.append(f"{args.o_dims}dim")
        if family != "gpt2_sde":
            parts.append(args.family)
        if args.poisson != 1:
            parts.append(f"poisson{args.poisson}")
        if args.n_layer != 8:
            parts.append(f"{args.n_layer}l")
        if args.n_head != 1:
            parts.append(f"{args.n_head}h")
        if args.diffusion != 20:
            parts.append(f"diffusion{args.diffusion}")
        if args.number_events != None:
            parts.append(f"{args.number_events}events")

        filename = "_".join(parts) + ".csv"
            
    prediction_rows = [obs_x_row, obs_y_row, pred_x_row, pred_y_row, gt_x_row, gt_y_row, evnt_t_row]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(prediction_rows)
    print(f"Successfully written .csv file for {family}!")
    print(filename)


def nonnaive_retrieve_model(args):
    rundir = "/cluster/scratch/millerm/models_cf/iclr"
    df = read_run_dir(rundir)

    args_dict = vars(args)
    args_dict.pop("position")

    args_dict.pop("with_pe")

    if args_dict["data"] == "gaussian":
        assert args_dict["dag_type"] == "only_parent"
        assert args_dict["n_thetas"] == args_dict["batch_size"]
        
        if args_dict["ao"]: assert "ao" in args_dict["family"]
        else: assert "ao" not in args_dict["family"]

        args_dict.pop("lamb")
        args_dict.pop("max_time")
        args_dict.pop("n_points")
        args_dict.pop("ode")
        args_dict.pop("poisson")
        args_dict.pop("number_events")

    for key in list(args_dict.keys()):
        if "o_" in key:
            new_key = "n_" + key[2:]
            args_dict[new_key] = args_dict[key]

    mask = pd.Series(True, index=df.index)

    for col in df.columns:
        if col in args_dict:
            mask &= df[col] == args_dict[col]

    matches = df[mask]

    if len(matches) == 0:
        raise ValueError("No matching run found for given args")
    if len(matches) > 1:
        raise ValueError(f"Multiple runs matched ({len(matches)}). "
                         f"Matched run_ids: {matches['run_id'].tolist()}")
    
    return matches.iloc[0]["run_id"]


def loss_with_estimated_beta(data_sampler, args):
    print("p", "mae", "mse")
    allll = []

    assert args.data == "gaussian"
    assert args.transformation == "addlin"
    assert args.n_thetas == 64
    assert args.o_points == 50

    def bootstrap(lst: list, device: str, eval_steps: int = 100):
        tensor = torch.tensor(lst, device=device)

        quantile_indices = random.choices(range(eval_steps), k = eval_steps)
        quantiles = torch.tensor(np.quantile(tensor[quantile_indices].numpy(), q=[0.025, 0.975]))

        estimate = torch.mean(tensor).item()
        lower_q = max(0, 2 * estimate - torch.mean(quantiles[1]).item())
        upper_q = max(0, 2 * estimate - torch.mean(quantiles[0]).item())

        return estimate, lower_q, upper_q

    eval_steps = 100

    mean_row = {"statistic": "mean"}
    q025_row = {"statistic": "q025"}
    q975_row = {"statistic": "q975"}

    for p in range(args.min_examples, args.o_points + 1):
        mses = []
        maes = []

        for itr in range(eval_steps):
            xs = data_sampler.complete_dataset(args.n_thetas, p, args.o_vars, itr = itr, transformation=args.transformation, constant_z = args.constant_z)

            XS = xs[:,:int(p * 2):2,:]
            YS = xs[:,1:int(p * 2):2,:]

            x_bar = torch.mean(XS, dim = 1).unsqueeze(1)
            y_bar = torch.mean(YS, dim = 1).unsqueeze(1)

            assert len(x_bar.shape) == 3

            XS_centered = XS - x_bar
            YS_centered = YS - y_bar

            num = (XS_centered * YS_centered).sum(dim = 1)
            den = (XS_centered * XS_centered).sum(dim = 1)

            w_hat = (num / den).unsqueeze(1)
            b = data_sampler.beta_b.unsqueeze(1)

            assert w_hat.shape == b.shape

            mae = (w_hat - b).abs().mean()

            x_cf = xs[:, -2, :].unsqueeze(1)
            y_cf = xs[:, -1, :].unsqueeze(1)

            z_index = data_sampler.z_index

            x_z = xs[:, int(2 * z_index), :].unsqueeze(1)
            y_z = xs[:, int(2 * z_index + 1), :].unsqueeze(1)

            y_pred = (x_cf - x_z) * w_hat + y_z
            y_gt = (x_cf - x_z) * b + y_z

            assert torch.allclose(y_cf, y_gt, atol = 1e-04)

            mse = (y_pred - y_gt).abs().mean()

            mses.append(mse.item())
            maes.append(mae.item())

        mean_mse, lower_q, upper_q = bootstrap(mses, xs.device, eval_steps)
        
        mean_row[str(p)] = mean_mse
        q025_row[str(p)] = lower_q
        q975_row[str(p)] = upper_q

    mse_rows = [mean_row, q025_row, q975_row]
    csv_field_names = ["statistic"] + [str(p) for p in range(args.min_examples, args.o_points + 1)]

    filedir = "eval/iclr/mse/"
    filename = filedir + "estimated_loss.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(mse_rows)
    print(f"Successfully written .csv!\n{filename}")
        

# the models are not provided in the supplementary material
# provided code suffices to train models from scratch
# training on one NVIDIA GeForce RTX 3090 GPU takes between 10 minutes and 3 hours
# most models can be trained in 30 to 60 minutes
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for dataset arguments')

    parser.add_argument('--o_dims',         type=int, default=5, help="o_dims for dataset setup")   # 5
    parser.add_argument('--o_vars',         type=int, default=2, help="o_vars for dataset setup")
    parser.add_argument('--o_points',       type=int, default=50, help="o_points for dataset setup")
    parser.add_argument('--n_points',       type=int, default=60, help="n_points for sde dataset setup")
    parser.add_argument('--n_thetas',       type=int, default=64, help="n_thetas for dataset setup")
    parser.add_argument('--n_bins',         type=int, default=1, help="n_bins for classification prediction")
    parser.add_argument('--lamb',           type=int, default=40, help="lambda parameter for poisson process")
    parser.add_argument('--max_time',       type=float, default=.5, help="maximum time for brownian motion: in expectation, lamb * max_time = o_points")
    parser.add_argument('--poisson',        type=int, default=1, help="poisson process (1) or uniform time steps (0)")
    parser.add_argument('--ode',            type=int, default=0, help="ode (1) or sde (0) for diffusion of sdeint under g()")
    parser.add_argument('--diffusion',      type=int, default=20, help="diffusion parameter, i.e., parameter for exponential distribution governing sigma")
    parser.add_argument('--number_events',  type=int, default=None, help="fixed number of event for poisson 1")
    parser.add_argument('--dag_type',       type=str, default="only_parent", help="dag_type for dataset setup")

    parser.add_argument('--n_layer',        type=int, default=12, help="number of layers of transformer model")
    parser.add_argument('--n_head',         type=int, default=8, help="number of heads of transformer model")
    parser.add_argument('--n_embd',         type=int, default=256, help="d_model of transformer model")

    parser.add_argument('--data',           type=str, default="gaussian", help="data type for get_data_sampler")
    parser.add_argument('--train_steps',    type=int, default=50000, help="Number of steps the model is trained on:    required for eval_seeds_dict")
    parser.add_argument('--eval_steps',     type=int, default=10000, help="Number of evaluation steps:                   required for eval_seeds_dict")
    parser.add_argument('--learning_rate',  type=float, default=0.0001, help="learning rate")
    parser.add_argument('--diversity',      type=int, default=16000000, help="Diversity")           # 16000000 for o_dims = 5
    parser.add_argument('--constant_z',     type=int, default=-1, help="Train model on constant z_index")
    parser.add_argument('--block_setup',    type=int, default=1, help="Train model on block setup")
    parser.add_argument('--with_pe',        type=int, default=1, help="Train model with positional encodings")
    parser.add_argument('--min_examples',   type=int, default=2, help="Minimum number of in-context examples during training")
    parser.add_argument('--theta_dist',     type=str, default="uniform", help="Distribution of theta")
    parser.add_argument('--eval_theta_dist',    type=str, default="uniform", help="Distribution of theta at evaluation time, different from the one extracted from training")
    parser.add_argument('--family',         type=str, default="gpt2", help="Family of model: one of [gpt2, gpt2_ao, gpt2_mlp, lstm, gru, rnn]")
    parser.add_argument('--transformation', type=str, default="addlin", help="Transformation of complete dataset")
    parser.add_argument('--batch_size',     type=int, default=64, help="similar to n_thetas")
    # parser.add_argument('--model_size',     type=str, default="standard", help="model_size to evaluate, one of [tiny, small, standard, fourlayer, eightlayer]")
    parser.add_argument('--training_loss',  type=str, default="mse", help="training loss of model: all are evaluated here on MSE; this serves as comparison")
    parser.add_argument('--eval_loss',      type=str, default="mse", help="evaluation loss of model: all are evaluated here on MSE; this serves as comparison")
    parser.add_argument('--ao',             type=int, default=0, help="attention_only argument for model extraction")
    parser.add_argument('--predict_y',      type=int, default=0, help="predict_y argument for prediction of observational y")
    parser.add_argument('--predict_x',      type=int, default=0, help="predict_x argument for prediction of x_CF - x")
    parser.add_argument('--predict_beta',   type=int, default=0, help="predict_beta argument for prediction of beta")
    parser.add_argument('--randomize_labels',   type=int, default=0, help="randomize_labels argument for model")
    parser.add_argument('--probe_type',     type=str, default="theta", help="probe_type argument for model")
    parser.add_argument('--probing_idx',    type=int, default=500, help="probing_idx for layer at which we suppose the relevant information is stored, choose high number as default")
    
    parser.add_argument('--disabled_layers', type=int, default=-1, help="disabled layers for analysis of relevant model")
    parser.add_argument('--continuation',   type=int, default=0, help="continuation argument for prediction of y_{n+1}")
    # parser.add_argument('--disentangled',   type=int, default=0, help="work on the disentangled transformer")
    # parser.add_argument('--weights',        type=int, default=0, help="retrieve weight matrices")
    parser.add_argument('--position',       type=int, default=-1, help="position to compute attentions for, default: -1, mean")
    parser.add_argument('--itr',            type=int, default=0, help="iteration to compute attentions for, default: 0, first seed sampling iteration")
    parser.add_argument('--dim_index',      type=int, default=0, help="dimension to compute SDE predictions for, default: 0, first dimension")
    parser.add_argument('--conference',     type=str, default="iclr", help="conference name for the experiment")
    args = parser.parse_args()


    o_dims = args.o_dims
    kwargs = {"dag_type": args.dag_type}

    if args.data == "gaussian":
        reset_theta_dist = False
        if args.theta_dist != args.eval_theta_dist:
            theta_dist = args.theta_dist
            args.theta_dist = args.eval_theta_dist
            reset_theta_dist = True
        data_sampler = get_data_sampler(args, o_dims, **kwargs)
        if reset_theta_dist:
            args.theta_dist = theta_dist
        write_attentions(data_sampler, args)
        # loss_with_estimated_beta(data_sampler, args)
        # perfectly_correlated_prediction(data_sampler, args)
                
        # if args.n_bins != 1: predict_bins(data_sampler, args)
        # else: write_mse(data_sampler, args)

        # write_ess_file()
                
        # datasets, evalsets, datasets_diff, evalsets_diff = extract_residual_stream(data_sampler, args)
        # for hidden_idx, dataset in tqdm(enumerate(datasets)):
        #     evalset = evalsets[hidden_idx]
        #     train_probe(dataset, evalset, args, hidden_idx=hidden_idx)
        #     if hidden_idx > 0:
        #         dataset_diff = datasets_diff[hidden_idx - 1]
        #         evalset_diff = evalsets_diff[hidden_idx - 1]
        #         train_probe(dataset_diff, evalset_diff, args, hidden_idx=hidden_idx, probe_diff=True)
        
    if args.data == "sde":
        print("starting data sampler")
        data_sampler = get_sde_data_sampler(args, o_dims, **kwargs)
        print("finalizing data sampler")
        # write_attentions(data_sampler, args)
        # sde_data_for_plot(data_sampler, args)
        write_mse_sde(data_sampler, args)
        
        # datasets, evalsets, datasets_diff, evalsets_diff = extract_residual_stream(data_sampler, args)
        # for hidden_idx, dataset in tqdm(enumerate(datasets)):
        #     evalset = evalsets[hidden_idx]
        #     train_probe(dataset, evalset, args, hidden_idx=hidden_idx)
        #     if hidden_idx > 0:
        #         dataset_diff = datasets_diff[hidden_idx - 1]
        #         evalset_diff = evalsets_diff[hidden_idx - 1]
        #         train_probe(dataset_diff, evalset_diff, args, hidden_idx=hidden_idx, probe_diff=True)