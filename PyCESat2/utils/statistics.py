import numpy as np
def n_bar(wf):
	n = np.arange(1,len(wf.count)+1)
	sum_yn = np.sum(n*wf.count)
	return sum_yn / np.sum(wf.count)

def moment(wf, m):
	n_minus_nbar = wf.count - n_bar(wf)    
	return np.sum(wf.count * n_minus_nbar**m) / np.sum(wf.count)

def skewness(wf):
	return moment(wf,3) / moment(wf,2)**1.5

def kurtosis(wf):
	return moment(wf,4) / moment(wf,2)**2

def mode(wf):
	peak = np.where(wf.count == wf.count.max())[0][-1]
	return wf.height[peak]

def mean(wf):
	n = np.arange(1,len(wf.count)+1)
	return np.sum(n*wf.count) / np.sum(wf.count)

def sum_squares(wf):
	n = np.arange(1,len(wf.count)+1)
	return np.sum(n**2 * wf.count)

def variance(wf):
	n = np.arange(1,len(wf.count)+1) 
	mean_square = sum_squares(wf) / np.sum(wf.count)
	return mean_square - mean(wf)**2

def stdev(wf):
	return np.sqrt(variance(wf))

def median(wf):
	n_halves = np.sum(wf.count) / 2
	sub_sums = [np.sum(wf.count[:i]) for i in range(len(wf.count))]
	i = np.where(sub_sums > n_halves)[0][0]    
	return i + (n_halves - sub_sums[i-1]) / wf.count[i]

def pearson_skewness(wf, n):
	if n == 1:
		return (mean(wf) - mode(wf)) / stdev(wf)
	elif n == 2:
		return 3 * (mean(wf) - median(wf)) / stdev(wf)
	else:
		raise ValueError("Pearson Skew Coefficient must be 1 or 2 (i.e. n=1 or n=2)")

def AUC(wf):
	return np.trapz(wf.count)
