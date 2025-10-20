import torch

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def masked_squared_error(ys_pred, ys, mask):
    mse = (ys - ys_pred).square() * mask
    return mse.sum()

def masked_mean_squared_error(ys_pred, ys, mask):
    mse = (ys - ys_pred).square() * mask
    return mse.sum() / mask.sum()


def absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs()

def mean_absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs().mean()


def brier(ys_large, logits, temperature = 1):
    return (ys_large - torch.nn.functional.softmax(logits / temperature, dim = 1)).square().mean()


def rps(ys_large, logits, temperature = 1):
    cumsums = torch.cumsum(torch.nn.functional.softmax(logits / temperature, dim = 1) - ys_large, dim = 1)
    return cumsums.square().sum(dim = 1).mean()


def spherical(ys_large, logits, temperature = 1):
    return (1 - ((ys_large * torch.nn.functional.softmax(logits / temperature, dim = 1)) / torch.nn.functional.softmax(logits / temperature, dim = 1).square().sum(dim = 1).sqrt()).sum(dim = 1)).mean()


def cross_entropy(ys_target, logits):
    return torch.nn.functional.cross_entropy(logits, ys_target) #, reduction="none")

def hhi(logits, temperature = 1):
    return (torch.nn.functional.softmax(logits / temperature, dim = 1)**2).sum(dim = 1).mean()

def r2(ys_pred, ys):
    assert len(ys_pred.shape) == len(ys.shape) == 2
    assert ys_pred.shape == ys.shape
    assert ys.shape[1] == 1
    y_mean = ys.mean()
    ss_total = ((ys - y_mean) ** 2).sum()
    ss_residual = ((ys - ys_pred) ** 2).sum()
    r2_scores = 1 - (ss_residual / ss_total)
    return r2_scores.mean()


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "counterfactual_regression": CounterfactualRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError
    

class CounterfactualRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        super(CounterfactualRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)

    @staticmethod
    def get_metric(training_loss="mse"):
        if training_loss == "mse": return squared_error
        if training_loss == "mae": return absolute_error
        if training_loss == "masked_mse": return masked_squared_error
        else: raise NotImplementedError

    @staticmethod
    def get_training_metric(training_loss="mse"):
        if training_loss == "mse": return mean_squared_error
        if training_loss == "mae": return mean_absolute_error
        if training_loss == "masked_mse": return masked_mean_squared_error
        if training_loss == "rps": return rps
        if training_loss == "brier": return brier
        if training_loss == "spherical": return spherical
        if training_loss == "cross_entropy": return cross_entropy
        else: raise NotImplementedError
