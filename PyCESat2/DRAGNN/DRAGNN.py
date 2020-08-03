import numpy as np

def eucledian_distances(heights, distances, P=20):
	photons = list(zip(heights,distances))

	n_neighbors = []
	for photon in photons:
		photon_dists = []
		for p in photons:
			if p != photon:
				euc_dist = abs(np.linalg.norm(photon - p))
				photon_dists.append(euc_dist)
		n_neighbors.append(len(np.where(np.asarray(photon_dists) > P)))


	return n_neighnbors

def get_neighbors():
	pass

def compute_histogram():
	pass

def find_peaks():
	pass

def fit_gaussian():
	pass
