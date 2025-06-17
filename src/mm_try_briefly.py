import argparse

from datetime import datetime
from dataset import get_data_sampler
from sde import get_sde_data_sampler, plot, variance
import torch
from tqdm import tqdm


print("functions imported")
    
parser = argparse.ArgumentParser(description='parser for dataset arguments')
### -----------
parser.add_argument('--o_dims',         type=int, default=5, help="o_dims for dataset setup")
parser.add_argument('--o_vars',         type=int, default=2, help="o_dims for dataset setup")
parser.add_argument('--o_points',       type=int, default=3, help="o_dims for dataset setup")
parser.add_argument('--n_points',       type=int, default=60, help="n_points, max_events for sde dataset setup")
parser.add_argument('--n_thetas',       type=int, default=64, help="n_thetas for dataset setup")
parser.add_argument('--batch_size',     type=int, default=64, help="similar to n_thetas")
parser.add_argument('--lamb',           type=int, default=200, help="lambda parameter for poisson process")
parser.add_argument('--max_time',       type=int, default=.1, help="maximum time for brownian motion: in expectation, lamb * max_time = o_points")
parser.add_argument('--dag_type',       type=str, default="only_parent", help="dag_type for dataset setup")
### -----------
parser.add_argument('--data',           type=str, default="gaussian", help="data type for get_data_sampler")
parser.add_argument('--train_steps',    type=int, default=10000, help="Number of steps the model is trained on:    required for eval_seeds_dict")
parser.add_argument('--eval_steps',     type=int, default=1000, help="Number of evaluation steps:                      required for eval_seeds_dict")
parser.add_argument('--transformation', type=str, default="addlin", help="Transformation of complete dataset")
parser.add_argument('--diversity',      type=int, default=12800000, help="Number of evaluation steps:                      required for eval_seeds_dict")
parser.add_argument('--theta_dist',     type=str, default="uniform", help="distribution of theta: either norm or uniform")
args = parser.parse_args()

data_sampler = get_data_sampler(args, args.o_dims)

xs_3 = data_sampler.complete_dataset(args.n_thetas, args.o_points, args.o_vars, itr = 3, transformation=args.transformation)
us_3 = data_sampler.us_b
# print(xs_3[0])
# t3 = data_sampler.theta_b
# w3 = data_sampler.w_b
# xs_4 = data_sampler.complete_dataset(args.n_thetas, args.o_points, args.o_vars, itr = 4, transformation=args.transformation)
# # t4 = data_sampler.theta_b
# xs_5 = data_sampler.complete_dataset(args.n_thetas, args.o_points, args.o_vars, itr = 5, transformation=args.transformation)
# t5 = data_sampler.theta_b
# xs_3 = data_sampler.complete_dataset(args.n_thetas, args.o_points, args.o_vars, itr = 3, split = "val")
# t3v = data_sampler.theta_b
# w3v = data_sampler.w_b

# print(t3v.unique())
# print(xs_3)

# data_sampler = get_sde_data_sampler(args, args.o_dims)

s = datetime.now()
# xs_3, ys, ts = data_sampler.complete_sde_dataset(args.n_thetas, args.o_vars, args.lamb, args.max_time, args.n_points, itr = 3, split = "train")
# xs_3 = data_sampler.complete_sde_dataset(args.n_thetas, args.o_vars, args.lamb, args.max_time, args.n_points, itr = 3, split = "train")
a, b, c = xs_3.shape
print(xs_3.shape)
# assert torch.sum(torch.isfinite(xs_3)) == a * b * c
# n_thetas, o_positions, _ = xs_3.shape

o_positions = 13
o_vars = 2
assert ((o_positions - 1)/ (2 * o_vars)).is_integer()
o_points = int((o_positions - 1) / (2 * o_vars))
gt_inds = torch.arange(o_points * o_vars + 1, o_positions)
gt_inds = gt_inds.reshape(o_points, -1)[:,1:].reshape(-1)
pred_inds = gt_inds - 1# # print(xs_3.shape)

print(gt_inds, pred_inds)
# # print(xs_3[0,gt_inds,:])
print(us_3[0,:,0,:])
print(us_3[0,:,0,:][-2])

print(torch.max(xs_3), torch.min(xs_3))
# e = datetime.now()
# elapsed = (e - s).total_seconds()
# print("Elapsed:", elapsed, "s")
# print(xs_3.shape)

# print(torch.max(xs_3), torch.min(xs_3))

# ys = torch.transpose(ys, 0, 1)



# plot(ts, ys, xlabel="t", ylabel = "y")

