import argparse

from datetime import datetime
from sde import get_sde_data_sampler

print("functions imported")
    
parser = argparse.ArgumentParser(description='parser for dataset arguments')
### -----------
parser.add_argument('--o_dims',         type=int, default=5, help="o_dims for dataset setup")
parser.add_argument('--o_vars',         type=int, default=2, help="o_vars for dataset setup")
parser.add_argument('--o_points',       type=int, default=50, help="o_points for dataset setup")
parser.add_argument('--n_points',       type=int, default=60, help="n_points, max_events for sde dataset setup")
parser.add_argument('--n_thetas',       type=int, default=8, help="n_thetas for dataset setup")
parser.add_argument('--batch_size',     type=int, default=8, help="similar to n_thetas")
parser.add_argument('--lamb',           type=float, default=5, help="lambda parameter for poisson process")
parser.add_argument('--max_time',       type=float, default=4, help="maximum time for brownian motion: in expectation, lamb * max_time = o_points")
parser.add_argument('--poisson',        type=int, default=1, help="poisson process (1) or uniform time steps (0)")
parser.add_argument('--ode',            type=int, default=0, help="ode (1) or sde (0) for diffusion of sdeint under g()")
parser.add_argument('--diffusion',      type=int, default=20, help="diffusion parameter, i.e., parameter for exponential distribution governing sigma")
parser.add_argument('--number_events',  type=int, default=None, help="fixed number of event for poisson 1")
parser.add_argument('--dag_type',       type=str, default="only_parent", help="dag_type for dataset setup")
parser.add_argument('--data',           type=str, default="sde", help="data type for get_data_sampler")
parser.add_argument('--train_steps',    type=int, default=50000, help="Number of steps the model is trained on:    required for eval_seeds_dict")
parser.add_argument('--eval_steps',     type=int, default=1000, help="Number of evaluation steps:                      required for eval_seeds_dict")
parser.add_argument('--transformation', type=str, default="addlin", help="Transformation of complete dataset")
parser.add_argument('--diversity',      type=int, default=2000000, help="Number of evaluation steps:                      required for eval_seeds_dict") # 12800000
parser.add_argument('--theta_dist',     type=str, default="uniform", help="distribution of theta: either norm or uniform")
args = parser.parse_args()

kwargs = {"dag_type": "only_parent"}


data_sampler = get_sde_data_sampler(args, args.o_dims)

start = datetime.now()
data = data_sampler.write_sde_dataset(args, split = "train")
end = datetime.now()
print(f"Time to write dataset: {end - start}")
