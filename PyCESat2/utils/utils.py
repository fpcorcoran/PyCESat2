import numpy as np
from scipy.optimize import differential_evolution

class OptimizationError(Exception):
	def __init__(self, a_bounds, c_bounds, d_bounds):
		self.message = "Unable to find optimal parameters within bounds: \
		a ∈ {0}, c ∈ {1}, d ∈ {2}".format(a_bounds, c_bounds, d_bounds)
		super().__init__(self.message)

def exp_curve(x, a, c):
	return a*np.exp(-c*x)


def optimize_params(metric, a_bounds=[-2000, 2000], c_bounds=[-10,10]):
	#return differential_evolution(metric, [a_bounds, c_bounds], seed=3).x
	opt_params = differential_evolution(metric, [a_bounds, c_bounds], seed=3)

	if opt_params.success:
		return (opt_params.x[0], opt_params.x[1])

	else:
		raise OptimizationError(a_bounds,c_bounds, d_bounds)
