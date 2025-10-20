import argparse

from datetime import datetime
from collections import Counter
from dataset import get_data_sampler
from sde import get_sde_data_sampler, plot, variance
import torch
from statistics import mean
from tqdm import tqdm

print("functions imported")
    
parser = argparse.ArgumentParser(description='parser for dataset arguments')
### -----------
parser.add_argument('--o_dims',         type=int, default=5, help="o_dims for dataset setup")
parser.add_argument('--o_vars',         type=int, default=2, help="o_vars for dataset setup")
parser.add_argument('--o_points',       type=int, default=2, help="o_points for dataset setup")
parser.add_argument('--n_points',       type=int, default=60, help="n_points, max_events for sde dataset setup")
parser.add_argument('--n_thetas',       type=int, default=2, help="n_thetas for dataset setup")
parser.add_argument('--batch_size',     type=int, default=8, help="similar to n_thetas")
parser.add_argument('--lamb',           type=float, default=5, help="lambda parameter for poisson process")
parser.add_argument('--max_time',       type=float, default=4, help="maximum time for brownian motion: in expectation, lamb * max_time = o_points")
parser.add_argument('--poisson',        type=int, default=1, help="poisson process (1) or uniform time steps (0)")
parser.add_argument('--ode',            type=int, default=0, help="ode (1) or sde (0) for diffusion of sdeint under g()")
parser.add_argument('--dag_type',       type=str, default="only_parent", help="dag_type for dataset setup")
parser.add_argument('--predict_y',      type=int, default=0, help="predict y instead of the counterfactual")
parser.add_argument('--predict_x',      type=int, default=0, help="predict x_CF - x instead of the counterfactual")
parser.add_argument('--predict_beta',   type=int, default=0, help="predict beta instead of the counterfactual")
### -----------
parser.add_argument('--data',           type=str, default="gaussian", help="data type for get_data_sampler")
parser.add_argument('--train_steps',    type=int, default=10000, help="Number of steps the model is trained on:    required for eval_seeds_dict")
parser.add_argument('--eval_steps',     type=int, default=1000, help="Number of evaluation steps:                      required for eval_seeds_dict")
parser.add_argument('--transformation', type=str, default="addlin", help="Transformation of complete dataset")
parser.add_argument('--diversity',      type=int, default=50, help="Number of evaluation steps:                      required for eval_seeds_dict") # 12800000
parser.add_argument('--theta_dist',     type=str, default="uniform", help="distribution of theta: either norm or uniform")
args = parser.parse_args()

kwargs = {"dag_type": "only_parent"}

print("p", "mae", "mse")
data_sampler = get_data_sampler(args, args.o_dims, **kwargs)

allll = []

# xs = data_sampler.complete_dataset(args.n_thetas, args.o_points, args.o_vars, transformation=args.transformation, itr = 0, block_setup=False)

# print(xs)

for p in range(30, 51): # [2, 5, 10, 30, 50, 100, 500]: #, 1000, 5000]: #, 10000, 50000, 100000]:
    mses = []
    maes = []
    # print(f"{p} points")
    for itr in range(100):
        xs = data_sampler.complete_dataset(args.n_thetas, p, args.o_vars, itr = itr, transformation=args.transformation)

        # assert args.n_thetas == 1
        # xs = xs.view(-1, args.o_dims)

        XS = xs[:,:int(p * 2):2,:]
        YS = xs[:,1:int(p * 2):2,:]
        # x_bar = torch.mean(XS, dim = 0).unsqueeze(0)
        # y_bar = torch.mean(YS, dim = 0).unsqueeze(0)

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

        # print(y_cf, y_gt)

        assert torch.allclose(y_cf, y_gt, atol = 1e-04)

        mse = (y_pred - y_gt).abs().mean()

        mses.append(mse.item())
        maes.append(mae.item())
    
    print(p, mean(maes), mean(mses))
    allll.append(mean(mses))
print(mean(allll))

# us = data_sampler.us_b

# theta = data_sampler.theta_b
# w = data_sampler.w_b

# print(xs)

# print(w)

# print(w)

# print(data_sampler.z_index)

# print(xs)
# print(torch.min(xs))



# ddd = torch.load(f"sde/data_{args.train_steps}_{0}_lambda_80.pt")
# dd2 = torch.load(f"sde/data_{args.train_steps}_{1}_lambda_80.pt")
# dd3 = torch.load(f"sde/data_{args.train_steps}_{2}_lambda_80.pt")
# dd4 = torch.load(f"sde/data_{args.train_steps}_{3}_lambda_80.pt")
# print(ddd.shape)
# print(ddd[:,42:,:].count_nonzero() / 4300000)
# print(torch.max(ddd))
# print(dd2[:,42:,:].count_nonzero() / 4300000)
# print(torch.max(dd2))
# print(dd3[:,42:,:].count_nonzero() / 4300000)
# print(torch.max(dd3))
# print(dd4[:,42:,:].count_nonzero() / 4300000)
# print(torch.max(dd4))

# print(ddd.shape)

# print(ddd[0,:,:])

# maxx = 0
# data_sampler = get_sde_data_sampler(args, args.o_dims, **kwargs)

# # for itr in range(8):
# # for i, mt in enumerate([1, 2]): #, 4, 8]):
# mt = 4
# lamb = 20 / mt
# xs_3 = data_sampler.complete_sde_dataset(args.n_thetas, args.o_vars, lamb, mt, args.n_points, itr = 3, split = "train", poisson = args.poisson, ode = args.ode)
# print(torch.max(xs_3))
# print(torch.min(xs_3))
# # maxx = max(maxx, torch.max(xs_3))

# # print(maxx)

# # o_positions = xs_3.shape[1]

# # gt_inds = torch.arange((o_positions // 2) + 1 + 2, o_positions)

# # data = xs_3

# # is_zero = (data == 0)  # shape: (B, P, D)

# # gt_idx_tensor = torch.zeros_like(is_zero)
# # gt_idx_tensor[:, gt_inds, :] = 1

# # ### this does not extract what we think it should extract

# # trailing_zeros = is_zero.flip(dims=[1]).cummin(dim=1).values.flip(dims=[1])

# # # print(xs_3[0,43:,3])

# # valid_trailing_zero_pos = is_zero & trailing_zeros
# # valid_positions = torch.logical_not(valid_trailing_zero_pos) & gt_idx_tensor
# # gt_indices = valid_positions.nonzero(as_tuple=False)
# # diff_tensor = torch.zeros_like(gt_indices)
# # diff_tensor[:,1] = 1
# # pred_indices = gt_indices - diff_tensor

# ts = torch.linspace(0, mt, steps = int(lamb * mt) + 1)
# xs = data_sampler.xs
# xs_cf = data_sampler.xs_cf

# # # # # print(xs)
# # # # # print(xs_cf)
# # # # # print(data_sampler.xs_init)
# # # # # print(data_sampler.theta_b)
# # # # print(xs[:,::2,:,:].reshape(-1, args.o_dims))

# # xs = ddd[0,:42:2,:].unsqueeze(0).unsqueeze(-1)
# # ys = ddd[0,1:42:2,:].unsqueeze(0).unsqueeze(-1)
# # xs = torch.stack([xs, ys], dim=2)
# # xs_cf = ddd[0,43::2,:].unsqueeze(0).unsqueeze(-1)
# # ys_cf = ddd[0,44::2,:].unsqueeze(0).unsqueeze(-1)
# # xs_cf = torch.stack([xs_cf, ys_cf], dim=2)

# for i in range(5):
#     ys = xs[:,:,:,i]
#     ys_cf = xs_cf[:,:,:,i]

#     ys = torch.transpose(ys, 0, 1).squeeze(-1)
#     ys_cf = torch.transpose(ys_cf, 0, 1).squeeze(-1)

#     # full_ys = torch.stack([ys, ys_cf], dim=1).squeeze(2)
#     print(ys.shape)
#     # print(full_ys.shape)

#     # plot(ts, full_ys, xlabel="t", ylabel = "y")
#     assert args.poisson == 0
#     plot(ts, ys, xlabel="t", ylabel = "y")
#     plot(ts, ys_cf, xlabel="t", ylabel = "y")

