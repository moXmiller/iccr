import torch
import torchsde
import matplotlib.pyplot as plt
from scipy import stats
from torch.distributions.exponential import Exponential
import random
import tqdm as tqdm

from dataset import LinearAssignments, direct_thetas


def get_sde_data_sampler(args, o_dims, sampler_seed = 12032025, **kwargs):
    names_to_classes = {
        "sde": SDESampler,
    }
    data_name = args.data

    random.seed(sampler_seed)

    theta_diversity = args.diversity
    if theta_diversity >= 12800001: assert theta_diversity == args.train_steps * args.batch_size * o_dims * 2   # 2: n_vars for theta sampling
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
        print("Unknown sampler: need 'sde'")
        raise NotImplementedError
    

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


    def _sample_greeks(self, n_thetas, o_vars, itr = None, split = 'train'):
        theta_b = self.theta_b
        theta_b = theta_b
        theta_b = theta_b.view(n_thetas, self.o_dims, o_vars)    # 18.07. # need to replace o_vars with 1 due to correct setup for theta

        if itr is None:
            greeks = [Exponential(theta_b).sample() for _ in range(5)]
            print("No greek seed provided.")
        else:
            if split == "train": self.greek_seed = self.seeds[itr]["greek"]
            elif split == "val": self.greek_seed = self.eval_seeds[itr]["greek"]
            generator = torch.Generator()
            generator.manual_seed(self.greek_seed)
            greeks = [Exponential(theta_b).sample() for _ in range(3)]
        
        # greeks = torch.maximum(torch.tensor(0.5), greeks) # 18.07.
        # greeks_tensor = torch.stack(greeks)  # Shape: (3,)
        # greeks = torch.clamp_min(greeks_tensor, 0.5)
        # if itr == 3: print(greeks)
        x_greeks, y_greeks, s = greeks

        a, b = x_greeks[:,:,0].view(n_thetas, self.o_dims), x_greeks[:,:,1].view(n_thetas, self.o_dims) # here, o_vars has a different role than in the regression setup
        c, d = y_greeks[:,:,0].view(n_thetas, self.o_dims), y_greeks[:,:,1].view(n_thetas, self.o_dims)

        self.alpha = a.clone().detach()
        self.beta = b.clone().detach()
        self.gamma = c.clone().detach()
        self.delta = d.clone().detach()
        self.sigma = s.clone().detach()
        return greeks
    

    def f(self, t, y):
        dx = self.alpha[:, self.embd_idx] * y[:,0] - self.beta[:, self.embd_idx] * y[:,0] * y[:,1]
        dy = self.delta[:, self.embd_idx] * y[:,0] * y[:,1] - self.gamma[:, self.embd_idx] * y[:,1]
        dx, dy = dx.unsqueeze(1), dy.unsqueeze(1)
        dd = torch.cat([dx, dy], dim = 1)
        return dd


    def g(self, t, y):
        eyes = self.sigma[:, self.embd_idx, :]
        assert y.shape == eyes.shape
        return eyes


    def _poisson_process(self, l, t, max_events, itr = None, split = "train"):
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
                o_events = stats.poisson.rvs(l*t, random_state = self.poisson_seed)
                generator = torch.Generator()
                generator.manual_seed(self.poisson_seed)
                event_times = torch.sort(torch.rand(o_events)).values 
                event_times = event_times * t
        
        o_events = min(o_events, max_events)
        return o_events, event_times
    

    def _sample_brownian_motions(self, n_thetas, o_vars, max_time, itr = None, split = "train"):
        bm_list = []
        if itr is None:
            for embd_idx in range(self.o_dims):
                bm = torchsde.BrownianInterval(t0=0, t1=max_time, size=(n_thetas, o_vars))  # (batch_size, state_size)
                bm_list.append(bm)
            print("No brownian seed provided.")
        else:
            if split == "train": self.brownian_seed = self.seeds[itr]["brownian"]
            elif split == "val": self.brownian_seed = self.eval_seeds[itr]["brownian"]
            generator = torch.Generator()
            generator.manual_seed(self.brownian_seed)
            for embd_idx in range(self.o_dims):
                bm = torchsde.BrownianInterval(t0=0, t1=max_time, size=(n_thetas, o_vars))  # (batch_size, state_size)
                bm_list.append(bm)
        return bm_list


    def _compose_sde(self, n_thetas, o_vars, event_times, o_points, brownians,
                     counterfactual = False, itr = None, lwr_do = 1, upr_do = 2, 
                     split = 'train'):
        assert 0 <= lwr_do < upr_do
        
        us_b = self._sample_us(n_thetas, o_points, o_vars, itr = itr, split = split)
        assert (n_thetas, o_points, o_vars, self.o_dims) == us_b.shape

        xs_init = torch.zeros_like(us_b[:,:1,:,:])
        assert len(xs_init.shape) == 4
        xs_b = torch.zeros_like(us_b)

        greeks = self._sample_greeks(n_thetas, o_vars, itr = itr, split=split)              # instantiate greeks

        if not counterfactual:
            xs_init = us_b[:,0,:,:].unsqueeze(1)
            assert xs_init.shape[1] == 1

        else:
            if itr is None:
                xs_init = torch.rand(n_thetas, o_vars, self.o_dims).unsqueeze(1)
                assert len(xs_init.shape) == 4
                assert xs_init.shape[1] == 1
                xs_init = torch.mul(xs_init, (lwr_do - upr_do))
                xs_init = xs_init.add(upr_do)
                print(f"No do seed provided.")
            else:
                if split == 'train': self.do_seed = self.seeds[itr]["do"]
                elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
                
                generator = torch.Generator()
                generator.manual_seed(self.do_seed)
                xs_init = torch.rand(n_thetas, o_vars, self.o_dims, generator=generator).unsqueeze(1)
                xs_init = torch.mul(xs_init, (lwr_do - upr_do))
                xs_init = xs_init.add(upr_do)
                assert len(xs_init.shape) == 4
                assert xs_init.shape[1] == 1

        assert len(xs_init.shape) == 4
        for embd_idx in range(self.o_dims):
            self.embd_idx = embd_idx
            bm = brownians[embd_idx]
            y0 = xs_init[:,0,:, embd_idx].view(n_thetas, o_vars)                            # combine X0 and Y0
            with torch.no_grad():
                ys = torchsde.sdeint(self, y0, event_times, bm=bm, method='euler')
            ys = torch.transpose(ys, 0, 1)
            xs_b[:,:,:, embd_idx] = ys

        self.embd_idx = None
        self.xs_b = xs_b
        self.o_points = o_points
        self.event_times = event_times
        self.ys = ys
        return xs_b
    

    def _sample_delimiters(self, n_thetas):
        return torch.ones((n_thetas, 1, self.o_dims))
    

    def complete_sde_dataset(self, n_thetas, o_vars, lamb, max_time, n_points, itr = None, split = 'train'):
        theta_b = self._sample_theta(n_thetas, o_vars, itr = itr, split = split, lwr = 1, upr = 2)
        brownian_list = self._sample_brownian_motions(n_thetas, o_vars, max_time, itr=itr, split=split)
        o_points, event_times = self._poisson_process(lamb, max_time, max_events = n_points, itr = itr, split = "train")

        if o_vars != 2: raise NotImplementedError
        self.theta_b = theta_b
        xs = self._compose_sde(n_thetas, o_vars, event_times = event_times, o_points=o_points, brownians=brownian_list, itr = itr, split = split)
        xs_cf = self._compose_sde(n_thetas, o_vars, event_times = event_times, o_points=o_points, itr =itr, brownians=brownian_list, split = split, counterfactual=True)
        delimiter = self._sample_delimiters(n_thetas)
        xs_view = xs.view(n_thetas, o_points * o_vars, self.o_dims)
        xs_cf_view = xs_cf.view(n_thetas, o_points * o_vars, self.o_dims)
        data_view = torch.concat([xs_view, delimiter, xs_cf_view], dim = 1)
        return data_view


def plot(ts, samples, xlabel, ylabel, title=''):
    _, batch_size, _ = samples.shape
    ts = ts.cpu()

    for b in range(batch_size):
        plt.figure()
        samples_b = samples[:,b,:]
        samples_plot = samples_b.squeeze().t().cpu()
        for i, sample in enumerate(samples_plot):
            plt.plot(ts, sample, marker='x', label=f'sample {i}')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
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