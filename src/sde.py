import torch
import torchsde
import matplotlib.pyplot as plt
from scipy import stats
from torch.distributions.exponential import Exponential
from torch.distributions.beta import Beta
import random
from tqdm import tqdm

import datetime

from dataset import LinearAssignments, direct_thetas


def get_sde_data_sampler(args, o_dims, sampler_seed = 12032025, **kwargs):
    names_to_classes = {
        "sde": SDESampler,
    }
    data_name = args.data

    random.seed(sampler_seed)

    theta_diversity = args.diversity
    if theta_diversity >= 12801: assert theta_diversity == args.train_steps * args.batch_size * o_dims
    theta_quadruple = direct_thetas(args, lwr=1, upr=2)

    seeds = random.sample(range(args.train_steps * 6), args.train_steps * 6)
    eval_seeds = random.sample(range(args.train_steps * 6, args.train_steps * 6 + args.eval_steps * 6), args.eval_steps * 6)
    seeds_dict = {idx: {"theta": seeds[idx*6], "u": seeds[idx*6+1], 
                        "brownian": seeds[idx*6+2], "do": seeds[idx*6+3], 
                        "greek": seeds[idx*6+4], "poisson": seeds[idx*6+5]} for idx in range(0, args.train_steps)}
    eval_seeds_dict = {idx: {"theta": eval_seeds[idx*6], "u": eval_seeds[idx*6+1],
                            "brownian": eval_seeds[idx*6+2], "do": eval_seeds[idx*6+3],
                            "greek": eval_seeds[idx*6+4], "poisson": eval_seeds[idx*6+5]} for idx in range(0, args.eval_steps)}

    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(o_dims, seeds = seeds_dict, eval_seeds = eval_seeds_dict, theta_quadruple = theta_quadruple, **kwargs)
    else:
        raise NotImplementedError("Unknown sampler: need 'sde'")
    

class SDESampler(LinearAssignments):
    def __init__(self, o_dims, seeds=None, eval_seeds=None, theta_quadruple=None, dag_type='only_parent'):
        super().__init__(o_dims, seeds=seeds, eval_seeds=eval_seeds, theta_quadruple=theta_quadruple)
        
        assert dag_type == "only_parent"
        self.seeds = seeds
        self.eval_seeds = eval_seeds
        self.theta_quadruple = theta_quadruple
        self.ess, _, _, self.dist = self.theta_quadruple
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        if seeds == None: print("No random seeds provided.")


    def _sample_greeks(self, n_thetas, itr = None, split = 'train'):
        theta_b = self.theta_b
        theta_b = theta_b.view(n_thetas, 1, self.o_dims)

        sigma_b = torch.ones_like(theta_b) * self.diffusion_parameter

        if itr is None:
            sigma = torch.cat([Exponential(sigma_b).sample() for _ in range(2)], dim = 1)
            print("No greek seed provided.")
            raise NotImplementedError
        else:
            if split == "train": self.greek_seed = self.seeds[itr]["greek"]
            elif split == "val": self.greek_seed = self.eval_seeds[itr]["greek"]
            generator = torch.Generator()
            generator.manual_seed(self.greek_seed)
            sigma = torch.cat([Exponential(sigma_b).sample() for _ in range(2)], dim = 1)

        xs_min = torch.ones(n_thetas * self.o_dims) * self.lower_bound
        ys_min = torch.ones(n_thetas * self.o_dims) * self.lower_bound
        self.mins = torch.cat([xs_min.unsqueeze(1), ys_min.unsqueeze(1)], dim = 1)
        assert torch.equal(xs_min, ys_min)

        xs_max = torch.ones_like(xs_min) * self.upper_bound
        ys_max = torch.ones_like(ys_min) * self.upper_bound
        self.maxs = torch.cat([xs_max.unsqueeze(1), ys_max.unsqueeze(1)], dim = 1)
        assert torch.equal(xs_max, ys_max)
        
        x_init, y_init = self.xs_init.permute(0,3,2,1)[:,:,0,0].reshape(n_thetas * self.o_dims), self.xs_init.permute(0,3,2,1)[:,:,1,0].reshape(n_thetas * self.o_dims)

        beta_lower = torch.log(xs_max / xs_min) / (self.max_time_bound * (ys_max - ys_min))
        assert beta_lower.shape[0] == n_thetas * self.o_dims
        assert len(beta_lower.shape) == 1
        assert torch.min(beta_lower) > 0

        delta_lower = torch.log(ys_max / ys_min) / (self.max_time_bound * (xs_max - xs_min))
        
        assert delta_lower.shape[0] == n_thetas * self.o_dims
        assert len(delta_lower.shape) == 1
        assert torch.min(delta_lower) > 0

        b = beta_lower.view(n_thetas, self.o_dims) + Exponential(theta_b).sample().squeeze(1)
        d = delta_lower.view(n_thetas, self.o_dims) + Exponential(theta_b).sample().squeeze(1)

        b = b.view(n_thetas * self.o_dims)
        d = d.view(n_thetas * self.o_dims)

        alpha_upper = (torch.log(xs_min / x_init) / self.max_time_bound) + b * ys_max
        alpha_lower = (torch.log(xs_max / x_init) / self.max_time_bound) + b * ys_min

        gamma_upper = d * xs_max - torch.log(ys_max / y_init) / self.max_time_bound
        gamma_lower = d * xs_min - torch.log(ys_min / y_init) / self.max_time_bound

        a = torch.rand_like(b) * (alpha_upper - alpha_lower) + alpha_lower
        c = torch.rand_like(d) * (gamma_upper - gamma_lower) + gamma_lower

        s = sigma.view(n_thetas * self.o_dims, 2)
    
        self.alpha = a.clone().detach()
        self.beta  = b.clone().detach()
        self.gamma = c.clone().detach()
        self.delta = d.clone().detach()

        self.sigma = s.clone().detach()
    

    def f(self, t, y):
        dx = self.alpha * y[:,0] - self.beta * y[:,0] * y[:,1]
        dy = self.delta * y[:,0] * y[:,1] - self.gamma * y[:,1]

        dx, dy = dx.unsqueeze(1), dy.unsqueeze(1)
        
        dd = torch.cat([dx, dy], dim = 1)
        return dd


    def g(self, t, y):
        eyes = 10 * self.sigma
        if self.ode: eyes = torch.zeros_like(eyes) # sanity check for diffusion impact
        # eyes = torch.ones_like(eyes) # sanity check for diffusion impact
        assert y.shape == eyes.shape
        return eyes


    def _poisson_process(self, l, t, max_events, itr = None, split = "train", poisson = True, number_events = None):
        if poisson:
            event_times, o_events = torch.tensor([0]), 2
            while len(torch.unique(event_times)) != o_events:
                if itr is None:
                    o_events = stats.poisson.rvs(l*t)
                    event_times = torch.sort(torch.rand(o_events)).values 
                    event_times = event_times * t
                    print("No poisson seed provided.")
                else:
                    if split == "train": self.poisson_seed = self.seeds[itr]["poisson"]
                    elif split == "val": self.poisson_seed = self.eval_seeds[itr]["poisson"]
                    if number_events == None: o_events = stats.poisson.rvs(l*t, random_state = self.poisson_seed)
                    else: o_events = number_events
                    generator = torch.Generator()
                    generator.manual_seed(self.poisson_seed)
                    event_times = torch.sort(torch.rand(o_events)).values 
                    event_times = event_times * t
                    # event_times = torch.concat((torch.tensor([0]), event_times))
        else:
            event_times = torch.linspace(0, t, steps = int(l * t) + 1)
            o_events = len(event_times)
        o_events = min(o_events, max_events)
        return o_events, event_times
    

    def _sample_brownian_motions(self, n_thetas, o_vars, max_time, itr = None, split = "train"):
        bm_list = []
        if itr is None:

            bm = torchsde.BrownianInterval(t0=0, t1=max_time, size=(n_thetas * self.o_dims, o_vars))  # (batch_size, state_size)
            print("No brownian seed provided.")
        else:
            if split == "train": self.brownian_seed = self.seeds[itr]["brownian"]
            elif split == "val": self.brownian_seed = self.eval_seeds[itr]["brownian"]
            generator = torch.Generator()
            generator.manual_seed(self.brownian_seed)

            bm = torchsde.BrownianInterval(t0=0, t1=max_time, size=(n_thetas * self.o_dims, o_vars))
        bm_list = [bm]
        return bm_list

    def _sample_init(self, n_thetas, o_vars, itr = None, split = 'train', continuation = False):
        theta_b = self.theta_b
        
        a_beta_dist = 1 / (2 - theta_b)
        b_beta_dist = torch.ones_like(a_beta_dist) * 2
        
        if itr is None:
            xs_init = torch.cat([Beta(a_beta_dist, b_beta_dist).sample() for _ in range(o_vars)], dim = 2) + self.theta_lower_bound
            print("No U seed provided.")
        elif continuation:
            xs_init = torch.cat([Beta(a_beta_dist, b_beta_dist).sample() for _ in range(o_vars)], dim = 2) + self.theta_lower_bound
        else:
            if split == "train": self.u_seed = self.seeds[itr]["u"]
            elif split == "val": self.u_seed = self.eval_seeds[itr]["u"]
            xs_init = torch.zeros(n_thetas, 1, o_vars, self.o_dims)
            generator = torch.Generator()
            generator.manual_seed(self.u_seed)
            xs_init = torch.cat([Beta(a_beta_dist, b_beta_dist).sample() for _ in range(o_vars)], dim = 2) + self.theta_lower_bound
            
        if self.scale is not None:
            xs_init = xs_init @ self.scale
        if self.bias is not None:
            xs_init += self.bias
        assert torch.max(xs_init) <= self.theta_upper_bound
        ###
        self.xs_init = xs_init
        return xs_init
    
    def _compose_sde(self, n_thetas, o_vars, event_times, o_points, brownians,
                     counterfactual = False, itr = None,
                     split = 'train'):
      
        xs_b = torch.zeros(n_thetas, o_points, o_vars, self.o_dims)

        if not counterfactual:
            xs_init = self._sample_init(n_thetas, o_vars, itr = itr, split = split)

            assert xs_init.shape[1] == 1
            assert torch.min(xs_init) >= 0
            assert len(xs_init.shape) == 4

        else:
            if itr is None:
                xs_init_do = torch.rand_like(self.maxs)
                assert xs_init_do.shape[1] == 1
                xs_init_do = xs_init_do.mul(self.upper_bound - self.lower_bound)
                xs_init_do = xs_init_do.add(self.lower_bound)
                print(f"No do seed provided.")
            else:
                if split == 'train': self.do_seed = self.seeds[itr]["do"]
                elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
                
                generator = torch.Generator()
                generator.manual_seed(self.do_seed)
                xs_init_do = torch.rand_like(self.maxs)
                xs_init_do = xs_init_do.mul(self.upper_bound - self.lower_bound)
                xs_init_do = xs_init_do.add(self.lower_bound)


        if not counterfactual: 
            self._sample_greeks(n_thetas, itr = itr, split=split)              # instantiate greeks
            y0_now = xs_init.permute(0, 3, 2, 1).reshape(n_thetas * self.o_dims, o_vars)
        else: 
            y0_now = xs_init_do

        brownians = brownians[0]
        with torch.no_grad():
            ys_now = torchsde.sdeint(self, y0_now, event_times, bm=brownians, method='euler')

        ys_now = torch.transpose(ys_now, 0, 1)

        xs_b_now = ys_now.view(n_thetas, self.o_dims, o_points, o_vars)
        xs_b_now = xs_b_now.permute(0, 2, 3, 1)
        
        self.xs_b = xs_b_now
        self.o_points = o_points
        self.event_times = event_times
        return xs_b_now
    

    def _sample_delimiters(self, n_thetas):
        return torch.ones((n_thetas, 1, self.o_dims))
    

    def _mask_nans(self, xs):
        o_thetas, o_points, o_vars, o_dims = xs.shape

        xs = torch.nan_to_num(xs)
        xs = torch.maximum(xs, torch.zeros_like(xs))

        zero_mask = xs == 0

        any_zero = zero_mask.any(dim=2)

        indices = torch.arange(o_points, device=xs.device).view(1, -1, 1).expand_as(any_zero)
        masked_indices = torch.where(any_zero, indices, o_points)

        first_zero_idx = masked_indices.min(dim=1).values

        point_indices = torch.arange(o_points, device=xs.device).view(1, -1, 1)
        cutoff_mask = point_indices >= first_zero_idx.unsqueeze(1)

        cutoff_mask = cutoff_mask.unsqueeze(2).expand(-1, -1, o_vars, -1)

        return cutoff_mask
    

    def complete_sde_dataset(self, n_thetas, o_vars, lamb, max_time, n_points, itr = None, split = 'train', poisson = True, ode = False, diffusion_parameter = 20, number_events = None):
        self.theta_lower_bound = 1
        self.theta_upper_bound = 2
        theta_b = self._sample_theta(n_thetas, o_vars, itr = itr, split = split, lwr = self.theta_lower_bound, upr = self.theta_upper_bound)

        self.ode = ode
        self.lower_bound = 0.5
        self.upper_bound = 2
        self.diffusion_parameter = diffusion_parameter
        brownian_list = self._sample_brownian_motions(n_thetas, o_vars, max_time, itr=itr, split=split)
        o_points, event_times = self._poisson_process(lamb, max_time, max_events = n_points, itr = itr, split = "train", poisson = poisson, number_events = number_events)

        self.max_time_bound = 1
        if o_vars != 2: raise NotImplementedError
        self.theta_b = theta_b
        xs = self._compose_sde(n_thetas, o_vars, event_times = event_times, o_points=o_points, brownians=brownian_list, itr = itr, split = split)
        xs_mask = self._mask_nans(xs)
        xs_cf = self._compose_sde(n_thetas, o_vars, event_times = event_times, o_points=o_points, itr =itr, brownians=brownian_list, split = split, counterfactual=True)
        xs_cf_mask = self._mask_nans(xs_cf)
        mask = torch.maximum(xs_mask, xs_cf_mask)
        xs = xs.masked_fill(mask, 0)
        xs_cf = xs_cf.masked_fill(mask, 0)
        self.xs = xs
        self.xs_cf = xs_cf
        
        delimiter = self._sample_delimiters(n_thetas)
        xs_view = xs.view(n_thetas, o_points * o_vars, self.o_dims)
        xs_cf_view = xs_cf.view(n_thetas, o_points * o_vars, self.o_dims)
        data_view = torch.concat([xs_view, delimiter, xs_cf_view], dim = 1)
        
        return data_view
    

    def write_sde_dataset(self, args, split = 'train'):
        parts = [f"data_{args.train_steps}"]
        if args.lamb != 5:
            parts.append(f"lambda_{int(args.lamb)}")
        if args.ode:
            parts.append("ode")
        if args.o_dims != 5:
            parts.append(f"{args.o_dims}dim")
        if args.diffusion != 20:
            parts.append(f"diffusion{args.diffusion}")
        if args.number_events != None:
            parts.append(f"{args.number_events}events")

        if args.train_steps > 2500:
            if args.train_steps % 2500 != 0: raise NotImplementedError
            for i in tqdm(range(args.train_steps // 2500)):
                data = self.complete_sde_dataset(int(args.n_thetas * 2500), args.o_vars, args.lamb, args.max_time, args.n_points, itr = i, split = split, poisson = args.poisson, ode = args.ode, diffusion_parameter = args.diffusion, number_events = args.number_events)
                filename = f"sde/{'_'.join(parts)}_{i}.pt"
                torch.save(data, filename)
        else:
            filename = f"sde/{'_'.join(parts)}.pt"
            data = self.complete_sde_dataset(int(args.n_thetas * args.train_steps), args.o_vars, args.lamb, args.max_time, args.n_points, itr = 0, split = split, poisson = args.poisson, ode = args.ode, diffusion_parameter = args.diffusion, number_events = args.number_events)
            torch.save(data, filename)
        return data


def plot(ts, samples, xlabel, ylabel, title=''):
    _, batch_size, _ = samples.shape
    ts = ts.cpu()

    plt.figure()
    print(samples.shape)
    for b in range(batch_size):
        samples_b = samples[:,b,:]
        samples_plot = samples_b.squeeze().t().cpu()

        for i, sample in enumerate(samples_plot):
            plt.plot(ts, sample, marker='x', label=f'sample {i}')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
    plt.show()


def variance(args):
    var_list = []
    data_sampler = get_sde_data_sampler(args, args.o_dims)
    for i in range(args.eval_steps):
        data = data_sampler.complete_sde_dataset(args.n_thetas, args.o_vars, args.lamb, args.max_time, args.n_points, itr = i, split = "val")
        _, o_positions, _ = data.shape
        data = data[:, int((o_positions + 1) / 2):, :]
        var_list.append(torch.var(data))
    return var_list


def analyze_datasets(lamb, ode = False, training_steps = 50000):
    parts = [f"data_{training_steps}"]
    if lamb != 5:
        parts.append(f"lambda_{int(lamb)}")
    if ode:
        parts.append("ode")

    if training_steps > 2500:
        final_min, final_max = float('inf'), float('-inf')
        if training_steps % 2500 == 0:    
            for i in range(training_steps // 2500):
                data = torch.load(f"sde/{'_'.join(parts)}_{i}.pt")
                print(f"Part {i}: mean {torch.mean(data)}, var {torch.var(data)}")

                if i == 6:
                    print(data[25, :, 4])
                final_min = min(final_min, torch.min(data))
                final_max = max(final_max, torch.max(data))
        else:
            data = torch.load(f"sde/{'_'.join(parts)}.pt")
    print(final_min, final_max)
    return data

if __name__ == "__main__":
    analyze_datasets(40, ode = False, training_steps = 50000)