import numpy as np
from scipy.optimize import differential_evolution
import warnings

class OptimizationError(Exception):
	def __init__(self, a_bounds, c_bounds):
		self.message = "Unable to find optimal parameters within bounds: a ∈ {0}, c ∈ {1}".format(a_bounds, c_bounds)
		super().__init__(self.message)

class MetricError(Exception):
	def __init__(self, bad_metric_arg):
		self.message = "Metric {0} not recognized".format(bad_metric_arg)
		super().__init__(self.message)


def exp_curve(x, a, c):
	return a*np.exp(-c*x)

def error_metric(height, count, metric):
	if metric == "SSE":
		def SumSquaredError(params):
			warnings.filterwarnings("ignore")
			return np.sum((count - exp_curve(height, *params)) ** 2.0)

		return SumSquaredError

	elif metric == "RMSE":
		def RootMeanSquaredError(params):
			warnings.filterwarnings("ignore")
			return np.sqrt(np.mean((count - exp_curve(height, *params)) ** 2.0))

		return RootMeanSquaredError

	elif metric == "MAE":
		def MeanAbsoluteError(params):
			return np.mean(np.abs(count - exp_curve(height, *params)))

		return MeanAbsoluteError

	else:
		raise MetricError(metric)



def optimize_params(metric, a_bounds=[-2000, 2000], c_bounds=[-10,0]):
	#return differential_evolution(metric, [a_bounds, c_bounds], seed=3).x
	opt_params = differential_evolution(metric, [a_bounds, c_bounds], seed=3)

	if opt_params.success:
		return (opt_params.x[0], opt_params.x[1])

	else:
		raise OptimizationError(a_bounds,c_bounds)
