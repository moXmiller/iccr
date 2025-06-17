import math
import random


class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_vars_truncated = args.vars.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_vars_schedule = args.vars
        self.n_points_schedule = args.points
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

    # updates the curriculum every iteration: we sample randomly to enforce exchangeability
    def update_var(self, var, schedule):
        trunc = random.choice(range(schedule.start, schedule.end + 1))
        return trunc

# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
