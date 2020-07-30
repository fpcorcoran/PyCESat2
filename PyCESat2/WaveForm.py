import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class waveForm:
    def __init__(self, h, n):
        self.height = h
        self.count = n
        self.curve = False
        self.height_fit = []
        self.count_fit = []

    def fit_curve(self, p0):
        """
        Fit a continuous curve to the discrete, binned LiDAR waveform.

        NOTE: This method is in development - your mileage may vary.
              Currently only supports exponential decay curves.

        Parameters:
        ----------
        p0 (tuple) - initial optimizition parameters for curve of the form
                     (A, C, D), where wave = A * e^(-C * elevation) + D.

        Returns:
        -------
        self (WaveForm)
        """

        def exp_decay(x, a, c, d):
            return a*np.exp(-c*x)+d

        popt, pcov = curve_fit(exp_decay,
                               self.count,
                               self.height,
                               p0,
                               maxfev=1000000)

        height_fit = np.linspace(np.max(self.height),
                                 np.min(self.height),
                                 len(self.height))

        count_fit = exp_decay(height_fit, *popt)

        self.curve = True
		setattr(self, "height_fit", height_fit)
		setattr(self, "count_fit", count_fit)
		setattr(self, "curve_params", {"a":popt[0],
									   "c":popt[1],
									   "d":popt[2]})

        return self

    def plot(self):
        plt.figure(figsize=(6,6))
        plt.barh(self.height, self.count)

        if self.curve:
            plt.plot(self.count_fit, self.height_fit, c="r")

        plt.show()

        return self


""" TESTING """

def main():
    a = 100.0
    c = -0.01
    d = 0.0

    p0=(a, c, d)

    h = np.linspace(2500,3000,200)
    n = a*np.exp(-c*h) + d

    wf = waveForm(h, n).fit_curve(p0).plot()
    wf2 = waveForm(h, n).plot()
    print("Done!")




if __name__ == "__main__":
    main()
