import random
import numpy as np
from scipy.stats import uniform, norm

import torch

class DataSampler:
    def __init__(self, o_dims):
        self.o_dims = o_dims

    def sample_xs(self):
        raise NotImplementedError


def effective_support_size(n_values, dist = "norm", seed = None, lwr = -6, upr = 6):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if dist == "norm":
        mean, std = 0, np.sqrt(12)
        values = torch.randn(n_values, generator=generator) * std + mean
        # pdfs = norm.pdf(values, loc = mean, scale = std)
        # weights = torch.tensor(pdfs / sum(pdfs))
        weights = torch.full_like(values, 1 / n_values)                    # among all those sampled according to the Gaussian density, we choose all with equal probability
        shannon = - torch.sum(weights * torch.log(weights))
    elif dist == "uniform":
        values = torch.rand(n_values, generator=generator)
        values = torch.mul(values, (lwr - upr))
        values = values.add(upr)
        pdfs = uniform.pdf(values, loc = lwr, scale = upr - lwr)
        weights = torch.tensor(pdfs / sum(pdfs))
        shannon = - torch.sum(weights * torch.log(weights))
    else: raise NotImplementedError
    ess = torch.exp(shannon)
    return ess, values, weights


def direct_thetas(args, lwr = -6, upr = 6, sampler_seed = 6052025):
    n_values = args.diversity
    dist = args.theta_dist
    ess, val, wei = effective_support_size(n_values, dist, seed = sampler_seed, lwr = lwr, upr = upr)
    return ess, val, wei, dist


def get_data_sampler(args, o_dims, sampler_seed = 12032025, **kwargs): # args here is args.training in base.yaml
    names_to_classes = {
        "gaussian": LinearAssignments,
    }
    data_name = args.data

    random.seed(sampler_seed)

    theta_diversity = args.diversity
    if theta_diversity >= 12800001: assert theta_diversity == args.train_steps * args.batch_size * o_dims # * 2   # 2: n_vars for theta sampling
    # 10.07. no * 2 n_vars, as we do not have different thetas for different noise terms
    theta_quadruple = direct_thetas(args)

    seeds = random.sample(range(args.train_steps * 4), args.train_steps * 4)
    eval_seeds = random.sample(range(args.train_steps * 4, args.train_steps * 4 + args.eval_steps * 4), args.eval_steps * 4)
    seeds_dict = {idx: {"theta": seeds[idx*4], "u": seeds[idx*4+1],
                        "w": seeds[idx*4+2], "do": seeds[idx*4+3]} for idx in range(0, args.train_steps)}
    eval_seeds_dict = {idx: {"theta": eval_seeds[idx*4], "u": eval_seeds[idx*4+1],
                            "w": eval_seeds[idx*4+2], "do": eval_seeds[idx*4+3]} for idx in range(0, args.eval_steps)}

    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(o_dims, seeds = seeds_dict, eval_seeds = eval_seeds_dict, theta_quadruple = theta_quadruple, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError
    

class GaussianSamplerCF(DataSampler):
    def __init__(self, o_dims, bias=None, scale=None, seeds=None, eval_seeds=None, theta_quadruple=None):
        """
        n_thetas:   number of distinct latents per batch in pre-training corpus
        o_points:   number of observation points per theta: collection of variables     o_points <= n_points
        o_vars:     number of (multivariate) variables in DAG at current iteration:     o_vars <= n_vars, for n_vars maximum number of variables
        o_dims:     number of dimensions of each variable
        
        Note that while n_thetas is a global parameter, we can increase o_points and o_dims along the dataset
        """
        super().__init__(o_dims)
        self.bias = bias
        self.scale = scale
        self.seeds = seeds
        self.eval_seeds = eval_seeds
        self.theta_quadruple = theta_quadruple
        self.ess, _, _, self.dist = self.theta_quadruple

        if seeds == None: print("No random seeds provided.")


    def _sample_theta(self, n_thetas, o_vars, itr=None, split = 'train', lwr=-6, upr=6, continuation = False):
        assert lwr < upr
        if lwr == -6: assert lwr == -upr
        if itr is None: raise NotImplementedError
        else:
            if split == "train":
                self.theta_seed = self.seeds[itr]["theta"]
                _, val, wei, _ = self.theta_quadruple
                generator = torch.Generator()
                generator.manual_seed(self.theta_seed)
                if len(val) < 12800001:
                    # indices = list(torch.utils.data.WeightedRandomSampler(wei, n_thetas * o_vars * self.o_dims, generator=generator))
                    indices = list(torch.utils.data.WeightedRandomSampler(wei, n_thetas * self.o_dims, generator=generator))
                    theta_b = val[indices]
                else: 
                    # theta_len = n_thetas * o_vars * self.o_dims
                    theta_len = n_thetas * self.o_dims
                    theta_b = val[range(itr * theta_len, (itr + 1) * theta_len)]
                # theta_b = theta_b.view(n_thetas, 1, o_vars, self.o_dims)
                theta_b = theta_b.view(n_thetas, 1, 1, self.o_dims)

            elif split == "val": 
                self.theta_seed = self.eval_seeds[itr]["theta"]
                # theta_b = torch.zeros(n_thetas, 1, o_vars, self.o_dims)
                theta_b = torch.zeros(n_thetas, 1, 1, self.o_dims)
                generator = torch.Generator()
                generator.manual_seed(self.theta_seed) 
                if self.dist == "uniform":
                    # raw_theta = torch.rand(n_thetas, 1, o_vars, self.o_dims, generator = generator)
                    raw_theta = torch.rand(n_thetas, 1, 1, self.o_dims, generator = generator)
                    raw_theta = torch.mul(raw_theta, lwr-upr)
                    theta_b = raw_theta.add(upr)
                elif self.dist == "norm":
                    # raw_theta = torch.randn(n_thetas, 1, o_vars, self.o_dims, generator = generator)
                    raw_theta = torch.randn(n_thetas, 1, 1, self.o_dims, generator = generator)
                    theta_b = torch.mul(raw_theta, np.sqrt(12))
                else: raise NotImplementedError

        self.theta_b = theta_b
        return theta_b
    

    def _sample_us(self, n_thetas, o_points, o_vars, itr = None, split = 'train', continuation = False):
        theta_b = self.theta_b
        if itr is None:
            us_b = torch.randn(n_thetas, o_points, o_vars, self.o_dims) + theta_b
            print("No U seed provided.")
        elif continuation:
            us_b = torch.randn(n_thetas, o_points, o_vars, self.o_dims) + theta_b
        else:
            if split == "train": self.u_seed = self.seeds[itr]["u"]
            elif split == "val": self.u_seed = self.eval_seeds[itr]["u"]
            us_b = torch.zeros(n_thetas, o_points, o_vars, self.o_dims)
            generator = torch.Generator()
            generator.manual_seed(self.u_seed)
            us_b = torch.randn(n_thetas, o_points, o_vars, self.o_dims, generator=generator) + theta_b
        if self.scale is not None:
            us_b = us_b @ self.scale
        if self.bias is not None:
            us_b += self.bias
        self.us_b = us_b
        return us_b
    

class LinearAssignments(GaussianSamplerCF):
    def __init__(self, o_dims, bias=None, scale=None, seeds=None, eval_seeds=None, theta_quadruple = None, dag_type='only_parent'):
        super().__init__(o_dims, bias, scale, seeds, eval_seeds, theta_quadruple=theta_quadruple)
        self.dag_type = dag_type
        

    def _sample_weights(self, n_thetas, o_vars, itr = None, split = 'train'):       # sample depending on theta
        theta_b = self.theta_b
        if itr is None:
            if self.dag_type != "any": w_b = torch.randn(n_thetas, 1, o_vars, self.o_dims) + theta_b
            else: 
                ###
                # t1 = theta_b.unsqueeze(3)
                # t2 = theta_b.unsqueeze(2)
                # theta_pairwise = t1 + 0 * t2
                # theta_long = theta_pairwise.view(n_thetas, 1, o_vars ** 2, o_dims)
                w_b = torch.randn(n_thetas, 1, o_vars ** 2, self.o_dims) + theta_b
            print("No w seed provided.")
        
        else:
            if split == "train": self.w_seed = self.seeds[itr]["w"]
            elif split == "val": self.w_seed = self.eval_seeds[itr]["w"]
            generator = torch.Generator()
            generator.manual_seed(self.w_seed)
            if self.dag_type != "any": 
                ###
                w_b = torch.zeros(n_thetas, 1, o_vars, self.o_dims)
                w_b = torch.randn(n_thetas, 1, o_vars, self.o_dims, generator=generator) + theta_b
            else: 
                w_b = torch.zeros(n_thetas, 1, o_vars ** 2, self.o_dims)
                w_b = torch.randn(n_thetas, 1, o_vars ** 2, self.o_dims, generator=generator) + theta_b
        self.w_b = w_b
        return w_b
        

    # function to implement more complex structural assignments: not relevant for this project
    def _parent_assignment(self, o_vars, itr = None, split = 'train'):
        if itr is not None:
            if split == 'train': self.do_seed = self.seeds[itr]["do"]
            elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
            random.seed(self.do_seed)
        else: print(f"No do seed provided.")
                    
        if self.dag_type == "one_parent":
            pa = {}
            for j in range(1, o_vars):
                parent_idx = random.choice(range(-1, j))
                pa[j] = parent_idx
            return pa
        
        if self.dag_type == "any":
            pa = {j: [] for j in range(1, o_vars)}
            num_max_edges = int(o_vars * (o_vars - 1) / 2)
            np.random.seed(self.do_seed)
            num_edges = int(np.random.binomial(num_max_edges, 0.5))
            edge_indices = random.sample(range(num_max_edges), k = num_edges)
            
            start = 0
            end = 1 # (o_vars - 1)
            for j in range(1, o_vars):
                parent_edges = range(start, end)
                relevant_edges = list(set(edge_indices) & set(parent_edges))
                if relevant_edges != []:
                    edges = [r - start for r in relevant_edges]
                    pa[j].extend(edges)
                if j < o_vars - 1:
                    start = end
                    end += j + 1
            assert end == num_max_edges

            self.pa = pa
            edge_indices.sort()
            concat_indices = ''.join([str(index + 1) for index in edge_indices]) if edge_indices != [] else 0
            self.concat_indices = int(concat_indices)
            return pa
        else: raise NotImplementedError


    def _compose_dag(self, n_thetas, o_points, o_vars, 
                     counterfactual = False, itr = None, lwr_do = -6, upr_do = 6, 
                     split = 'train', continuation = False, predict_y = False, transformation = "addlin"):
        assert lwr_do != upr_do
        if not continuation: us_b = self._sample_us(n_thetas, o_points, o_vars, itr = itr, split = split)
        # continuation: continuation of the in-context chain instead of starting counterfactual
        # resample fresh U
        else: us_b = self._sample_us(n_thetas, o_points, o_vars, itr=itr, split=split, continuation=continuation)
        assert (n_thetas, o_points, o_vars, self.o_dims) == us_b.shape
        
        xs_b = torch.zeros_like(us_b)
        w_b = self._sample_weights(n_thetas, o_vars, itr = itr, split = split)    
        ### 09.07.
        # if not counterfactual: w_b = self._sample_weights(n_thetas, o_vars, itr = itr, split = split)
        # else: w_b = self._sample_weights(n_thetas, o_vars, split = split)
        ### 09.07.
        if self.dag_type == "only_parent":
            transformations = {
                'addlin': lambda x, w, u: torch.mul(x, w) + u,
                'mullin': lambda x, w, u: (3410.4**(-0.5)) * torch.mul(x, w) * u,
                'tanh': lambda x, w, u: torch.tanh((1/13) * (torch.mul(x, w) + u)),
                'sigmoid': lambda x, w, u: 1 / (1 + torch.exp((-1/13)*(torch.mul(x, w) + u))),
            }
            if transformation not in transformations: raise ValueError(f"Unknown transformation: {transformation}. Valid options are: {list(transformations.keys())}")
            if not counterfactual:
                xs_b[:,:,0,:] = us_b[:,:,0,:]
                for j in range(1, o_vars):
                    xs_b[:,:,j,:] = transformations[transformation](us_b[:,:,0,:], w_b[:,:,j,:], us_b[:,:,j,:])
                if predict_y:
                    if split == 'train': self.do_seed = self.seeds[itr]["do"]
                    elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
            else:
                if itr is None:
                    xs_b[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims)[:,:,0]
                    xs_b = torch.mul(xs_b, (lwr_do - upr_do))
                    xs_b = xs_b.add(upr_do)
                    print(f"No do seed provided.")
                else:
                    if split == 'train': self.do_seed = self.seeds[itr]["do"]
                    elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
                    xs_b[:,:,0,:] = torch.zeros(n_thetas, o_points, 1, self.o_dims)[:,:,0]
                    generator = torch.Generator()
                    generator.manual_seed(self.do_seed)
                    xs_b[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims, generator=generator)[:,:,0]
                    xs_b = torch.mul(xs_b, (lwr_do - upr_do))
                    xs_b = xs_b.add(upr_do)
                for j in range(1, o_vars):
                    xs_b[:,:,j,:] = transformations[transformation](xs_b[:,:,0,:], w_b[:,:,j,:], us_b[:,:,j,:])

        # required to implement more complex structural assignments: not relevant for this project
        elif self.dag_type == "one_parent":
            pa = self._parent_assignment(o_vars, itr = itr, split = split)
            if not counterfactual:
                xs_b[:,:,0,:] = us_b[:,:,0,:]
                for j in range(1, o_vars):
                    if pa[j] != -1:
                        xs_b[:,:,j,:] = torch.mul(xs_b[:,:,pa[j],:], w_b[:,:,j,:]) + us_b[:,:,j,:]
                    else:
                        xs_b[:,:,j,:] = us_b[:,:,j,:]
            else:
                if itr is None:
                    xs_b[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims)[:,:,0]
                    xs_b = torch.mul(xs_b, (lwr_do - upr_do))
                    xs_b = xs_b.add(upr_do)
                    print(f"No do seed provided.")
                else:
                    if split == 'train': self.do_seed = self.seeds[itr]["do"]
                    elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
                    xs_b[:,:,0,:] = torch.zeros(n_thetas, o_points, 1, self.o_dims)[:,:,0]
                    generator = torch.Generator()
                    generator.manual_seed(self.do_seed)
                    xs_b[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims, generator=generator)[:,:,0]
                    xs_b = torch.mul(xs_b, (lwr_do - upr_do))
                    xs_b = xs_b.add(upr_do)
                for j in range(1, o_vars):
                    if pa[j] != -1:
                        xs_b[:,:,j,:] = torch.mul(xs_b[:,:,pa[j],:], w_b[:,:,j,:]) + us_b[:,:,j,:]
                    else:
                        xs_b[:,:,j,:] = us_b[:,:,j,:]
        
        # required to implement more complex structural assignments: not relevant for this project
        elif self.dag_type == "any":
            transformations = {
                'addlin': lambda x, w, u: torch.sum(torch.mul(x, w), dim = 2) + u,
                # 'mullin': lambda x, w, u: (3410.4**(-0.5)) * torch.mul(x, w) * u,
                # 'tanh': lambda x, w, u: torch.tanh((1/13) * (torch.mul(x, w) + u)),
                # 'sigmoid': lambda x, w, u: 1 / (1 + torch.exp((-1/13)*(torch.mul(x, w) + u))),
            }
            if transformation not in transformations: raise ValueError(f"Unknown transformation: {transformation}. Valid options are: {list(transformations.keys())}")
            pa = self._parent_assignment(o_vars, itr = itr, split = split)
            if not counterfactual:
                xs_b[:,:,0,:] = us_b[:,:,0,:]
                for j in range(1, o_vars):
                    if pa[j] != []:
                        ### transformations
                        # xs_b[:,:,j,:] = torch.sum(torch.mul(xs_b[:,:,pa[j],:], w_b[:,:,j,:].unsqueeze(1)), dim = 2) + us_b[:,:,j,:]  
                        # xs_b[:,:,j,:] = transformations[transformation](xs_b[:,:,pa[j],:], w_b[:,:,j,:].unsqueeze(1), us_b[:,:,j,:])
                        weight_indices = [p * o_vars + j for p in pa[j]]
                        xs_b[:,:,j,:] = transformations[transformation](xs_b[:,:,pa[j],:], w_b[:, :, weight_indices, :], us_b[:,:,j,:])
                        # we use the weights associated with the child as we have P_{f_j}(theta_j), i.e., the function determining f_j is completely determined by theta_j, in this simple case: beta_j
                    else:
                        xs_b[:,:,j,:] = us_b[:,:,j,:]
            else:
                if itr is None:
                    xs_b[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims)[:,:,0]
                    xs_b = torch.mul(xs_b, (lwr_do - upr_do))
                    xs_b = xs_b.add(upr_do)
                    print(f"No do seed provided.")
                else:
                    # if split == 'train': self.do_seed = self.seeds[itr]["do"]
                    # elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
                    # we already set this during self._parent_assignment()
                    xs_b[:,:,0,:] = torch.zeros(n_thetas, o_points, 1, self.o_dims)[:,:,0]
                    generator = torch.Generator()
                    generator.manual_seed(self.do_seed)
                    xs_b[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims, generator=generator)[:,:,0]
                    xs_b = torch.mul(xs_b, (lwr_do - upr_do))
                    xs_b = xs_b.add(upr_do)
                for j in range(1, o_vars):
                    if pa[j] != []:
                        # xs_b[:,:,j,:] = torch.sum(torch.mul(xs_b[:,:,pa[j],:], w_b[:,:,j,:].unsqueeze(1)), dim = 2) + us_b[:,:,j,:]  
                        weight_indices = [p * o_vars + j for p in pa[j]]
                        xs_b[:,:,j,:] = transformations[transformation](xs_b[:,:,pa[j],:], w_b[:, :, weight_indices, :], us_b[:,:,j,:])
                        # we use the weights associated with the child as we have P_{f_j}(theta_j), i.e., the function determining f_j is completely determined by theta_j, in this simple case: beta_j
                    else:
                        xs_b[:,:,j,:] = us_b[:,:,j,:]

        else: raise NotImplementedError
        self.xs_b = xs_b
        return xs_b


    def _sample_delimiters(self, n_thetas, o_points):
        random.seed(self.do_seed)
        z_index = random.choice(range(o_points))

        delimiters = torch.full((n_thetas, 1, self.o_dims), z_index)

        # filled = torch.full((n_thetas, 1, self.o_dims), z_index)
        # delimiters = 10 * torch.sin(filled) + 1000

        # binary_str = format(z_index, f'0{(self.o_dims)}b')
        # assert len(binary_str) <= self.o_dims
        # binary_tensor = torch.tensor([int(b) * (-200) + 100 for b in binary_str], dtype=torch.float32)
        # delimiters = binary_tensor.view(1, 1, self.o_dims).repeat(n_thetas, 1, 1)

        return delimiters, z_index


    def _us_dataset(self, n_thetas, o_points, o_vars, itr = None, split = 'train', continuation = False, lwr = -1, upr = 1):
        us = self._sample_us(n_thetas, o_points, o_vars, itr, split, continuation)
        if itr is None: raise NotImplementedError
        if split == 'train': self.do_seed = self.seeds[itr]["do"]
        elif split == 'val': self.do_seed = self.eval_seeds[itr]["do"]
        us_cf = torch.clone(us)
        generator = torch.Generator()
        generator.manual_seed(self.do_seed)
        us_cf[:,:,0,:] = torch.rand(n_thetas, o_points, 1, self.o_dims, generator=generator)[:,:,0]
        us = us.view(n_thetas, o_vars * o_points, self.o_dims)
        us_cf = us_cf.view(n_thetas, o_vars * o_points, self.o_dims)
        us_view = torch.cat([us,us_cf], dim = 1)
        return us_view


    def complete_dataset(self, n_thetas, o_points, o_vars, itr = None, split = 'train', continuation = False, block_setup = True, with_delimiter = True, predict_y = False, transformation = "addlin"):
        if continuation: with_delimiter = False
        _ = self._sample_theta(n_thetas, o_vars, itr = itr, split = split, continuation = continuation)                     # initiate self.theta_b

        # alternate observational and counterfactual data points
        if not block_setup:
            with_delimiter = False # to be deleted # 10.07.
            if continuation: raise NotImplementedError
            xs = self._compose_dag(n_thetas, o_points, o_vars, itr = itr, split = split, transformation = transformation, predict_y = predict_y)   # xs is four-dimensional
            xs_cf = self._compose_dag(n_thetas, o_points, o_vars, counterfactual = True, itr = itr, split = split, predict_y = predict_y, transformation = transformation)
            xs_concat = torch.concat([xs, xs_cf], dim = 2)                                                                  # concat along o_vars dimension
            data_view = xs_concat.view(n_thetas, 2 * o_vars * o_points, self.o_dims)

        # first learn beta in-context and then start counterfactual generation: block setup
        else:
            assert self.dag_type == "only_parent"
            xs = self._compose_dag(n_thetas, o_points, o_vars, itr = itr, split = split, transformation = transformation)
            # 10.07. default set to False
            if predict_y: xs_cf = self._compose_dag(n_thetas, o_points, o_vars, itr = itr, split = split, transformation = transformation, predict_y = predict_y)
            # 10.07.
            else: xs_cf = self._compose_dag(n_thetas, o_points, o_vars, itr =itr, split = split, counterfactual=True, transformation = transformation, predict_y = predict_y)
            delimiter, z_index = self._sample_delimiters(n_thetas, o_points)
            self.z_index = z_index
            xs_cf_sub = xs_cf[:, z_index, :, :]    # extract relevant data
            xs_view = xs.view(n_thetas, o_points * o_vars, self.o_dims)
            if with_delimiter: data_view = torch.concat([xs_view, delimiter, xs_cf_sub], dim = 1) # 09.07.
            else: data_view = torch.concat([xs_view, xs_cf_sub], dim = 1)

        return data_view

if __name__ == "__main__":
    seeds = {3: {"theta": 4, 
             "u": 1, 
             "w": 2, 
             "do": 10}}
    n_thetas = 6
    o_points = 4
    o_vars = 3
    o_dims = 5
    LA = LinearAssignments(o_dims, seeds=seeds, dag_type = "only_parent")
    xs = LA.complete_dataset(n_thetas,o_points,o_vars, itr = 3, continuation=False)
    print("theta", LA.theta_b)
    print(xs.shape)