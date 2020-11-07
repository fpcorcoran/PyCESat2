import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

class waveForm:
    def __init__(self, h, n, **kwargs):
        self.height = h
        self.count = n
        self.curve = False
        self.dist = False
        
        for key, value in kwargs.items():
            setattr(self,key,value)


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
    
    def stats(self):
        stats = {'mean':[mean(self)],
                 'mode':[mode(self)],
                 'median':[median(self)],
                 'stdev':[stdev(self)],
                 'variance':[variance(self)],
                 'skewness':[skewness(self)],
                 'kurtosis':[kurtosis(self)],
                 'pearson1':[pearson_skewness(self, 1)],
                 'pearson2':[pearson_skewness(self, 2)],
                 'AUC':[AUC(self)]}
        
        return stats
    
    def fit_dist(self, dist='norm'):
        
        #store distribution name
        name = dist
        
        #Get table of distributions
        dist = distributions[dist]
        
        #Fit distribution to get parameters
        params = dist.fit(self.count)
        
        #Split up parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        
        # Get histogram of original data
        bins= len(self.height) * 10
        y, x = np.histogram(self.count, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        
        #get start/end of middle 98% of distribution - regardless of args
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        #define domain of distribution
        dist_x = np.linspace(start, end, bins)
        
        #compute f(x) where f is the probability density function
        pdf = dist.pdf(dist_x, loc=loc, scale=scale, *arg)
        
        #scale results to match waveform
        dist_y = pdf*(np.max(self.count)/np.max(pdf))
        
        self.dist = True
        setattr(self, "dist_h", -dist_x)
        setattr(self, 'dist_fit', dist_y)
        setattr(self, 'dist_params',params)
        setattr(self, 'dist_name', name)
        

        return self
        

    def plot(self, fname=None):
        plt.figure(figsize=(6,6))
        plt.barh(self.height, self.count,height=0.1,label="waveform")

        if self.curve:
            plt.plot(self.count_fit, self.height, c="r", label="fitted curve")
            
        if self.dist:
            plt.plot(self.dist_fit, self.dist_h, c='g', label="distribtion")
            
        plt.legend()
        plt.ylim(np.min(self.height), np.max(self.height))

        if fname == None:
            plt.show()
        
        else:
            plt.savefig(fname)
            

        return self
