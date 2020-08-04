import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

class waveForm:
	def __init__(self, h, n):
		self.height = h
		self.count = n
		self.curve = False


	def fit_curve(self, metric="RMSE"):
		"""
		Fit a continuous curve to the discrete, binned LiDAR waveform.

		Initial parameters for the curve fitting are selected using the
		differenction evolution genetic algorithm.

		NOTE: This method is in development - your mileage may vary.
		Currently only supports exponential decay curves.

		Returns:
		-------
		self (WaveForm)
		"""

		metric = error_metric(self.height, self.count, metric=metric)

		optimal_params = optimize_params(metric)

		height_fit = np.linspace(np.max(self.height),
									np.min(self.height),
									len(self.height))

		count_fit = exp_curve(height_fit, *optimal_params)

		self.curve = True

		setattr(self, "height_fit", height_fit)
		setattr(self, "count_fit", count_fit)
		setattr(self, "curve_params", {"a":optimal_params[0],
										"c":optimal_params[1]})

		setattr(self, "metric", metric(optimal_params))
		setattr(self, "r2", r2_score(self.count, self.count_fit))

		return self

	def plot(self):
		plt.figure(figsize=(6,6))
		plt.barh(self.height, self.count)

		if self.curve:
			plt.plot(self.count_fit, self.height, c="r")
			#plt.plot(self.height_fit, self.count_fit, c="r")

			plt.show()

		return self


""" TESTING """

def main():
	a = 100.0
	c = -0.01
	d = 0.0

	p0=(a, c, d)

	h = np.linspace(2500,3000,200)
	n = a*np.exp(c*h) + d

	wf = waveForm(h, n).fit_curve(p0).plot()
	wf2 = waveForm(h, n).plot()
	print("Done!")




if __name__ == "__main__":
	main()
