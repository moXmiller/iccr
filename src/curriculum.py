import math
import random


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_vars_truncated = args.vars.start
        self.n_points = args.points.start
        # self.phase = args.phase.start
        # self.loss_weights = [1] + [0] * (args.phase.end)
        self.n_dims_schedule = args.dims
        self.n_vars_schedule = args.vars
        self.n_points_schedule = args.points
        # self.phase_schedule = args.phase
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_vars_truncated = self.update_var(
            self.n_vars_truncated, self.n_vars_schedule
        )
        self.n_points = self.update_var(
            self.n_points, self.n_points_schedule
        )

    def update_var(self, var, schedule):
        
        # this enforces exchangeability
        trunc = random.choice(range(schedule.start, schedule.end + 1))
        return trunc
        
    def update_loss_weights(self, loss_weights, schedule, phase):
        if self.step_count % schedule.interval == 0:

            if min(loss_weights) > 0 or phase == schedule.end + 1:
                loss_weights = [1] + [0] * schedule.end
            elif phase <= schedule.end:
                for idx, weight in enumerate(loss_weights):
                    if weight > 0 and idx > 0:
                        loss_weights[idx] = loss_weights[idx] / 2
                    elif idx == 0:
                        loss_weights[idx] -= 0.2 / 2**(phase)
                    elif phase == idx - 1:
                        loss_weights[idx] = 0.2
            assert sum(loss_weights) == 1, "Loss weights must sum to 1"
            phase += schedule.inc
        return loss_weights, min(phase, schedule.end + 1)
    
# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
