def above(distance, height, ransac):

	photons = list(zip(height, distance))
	above = np.asarray([photon for photon in photons if photon[0] > ransac.predict(photon[1].reshape(1, -1))])

	return beamObject(above[:,0],above[:,1],[np.nan],[np.nan])

def below(distance, height, ransac):

	photons = list(zip(height, distance))
	below = np.asarray([photon for photon in photons if photon[0] < ransac.predict(photon[1].reshape(1, -1))])

	return beamObject(below[:,0],below[:,1],[np.nan],[np.nan])
