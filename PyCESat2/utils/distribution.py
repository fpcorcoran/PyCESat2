from scipy.stats import *
distributions = {
	'alpha': alpha,
	'anglit':anglit,
	'arcsine':arcsine,
	'beta': beta,
	'betaprime': betaprime,
	'bradford': bradford,
	'burr': burr,
	'cauchy': cauchy,
	'chi': chi,
	'chi2': chi2,
	'cosine': cosine,
	'dgamma': dgamma,
	'dweibull':dweibull,
	'erlang': erlang,
	'expon': expon,
	'exponnorm': exponnorm,
	'exponpow': exponpow,
	'exponweib': exponweib,
	'f': f,
	'fatiguelife': fatiguelife,
	'fisk': fisk,
	'foldcauchy':foldcauchy,
	'foldnorm':foldnorm,
	'frechet_l': frechet_l,
	'frechet_r': frechet_r,
	'gamma': gamma,
	'gausshyper':gausshyper,
	'genexpon': genexpon,
	'genextreme': genextreme,
	'gengamma': gengamma,
	'genhalflogistic': genhalflogistic,
	'genlogisitic': genlogistic,
	'gennorm': gennorm,
	'genpareto': genpareto,
	'gilbrat': gilbrat,
	'gompertz': gompertz,
	'gumbel_l': gumbel_l,
	'gumbel_r': gumbel_r,
	'halfcauchy': halfcauchy,
	'halfgennorm': halfgennorm,
	'halflogistic': halflogistic,
	'halfnorm': halfnorm,
	'hypsecant': hypsecant,
	'invgamma': invgamma,
	'invgauss': invgauss,
	'invweibull': invweibull,
	'johnsonsb': johnsonsb,
	'johnsonsu': johnsonsu,
	'ksone': ksone,
	'kstwobign': kstwobign,
	'laplace': laplace,
	'levy': levy,
	'levy_l': levy_l,
	'loggamma': loggamma,
	'logistic': logistic,
	'loglaplace': loglaplace,
	'lognorm': lognorm,
	'lomax': lomax,
	'maxwell': maxwell,
	'mielke': mielke,
	'nakagami': nakagami,
	'ncf': ncf,
	'nct': nct,
	'ncx2': ncx2,
	'norm': norm,
	'pareto': pareto,
	'pearson3': pearson3,
	'powerlaw': powerlaw,
	'powerlognorm': powerlognorm,
	'powernorm': powernorm,
	'rayleigh': rayleigh,
	'rdist': rdist,
	'recipinvgauss': recipinvgauss,
	'reciprocal': reciprocal,
	'rice': rice,
	'semicircular': semicircular,
	't': t,
	'triang': triang,
	'truncexpon': truncexpon,
	'truncnorm': truncnorm,
	'tukeylambda': tukeylambda,
	'uniform': uniform,
	'vonmises': vonmises,
	'vonmises': vonmises_line,
	'wald': wald,
	'weibull_max': weibull_max,
	'weibull_min': weibull_min,
	'wrapcauchy': wrapcauchy
	}