import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .WaveForm import waveForm

class beamObject:
	def __init__(self, h, d):
		self.height = h
		self.distance = d

	def filter(self, win_x=3.0, win_h=3.0, threshold=15):
		"""

		"""

		#get along track range
		track_min = np.min(self.distance)
		track_max = np.max(self.distance)

		#get height range
		h_min = np.min(self.height)
		h_max = np.max(self.height)

		#combine height and track -> 2D array
		xy = pd.DataFrame.from_dict({"height":self.height,"distance":self.distance})

		#init window
		x_start = track_min
		y_start = h_max

		x_stop = track_min + win_x

		filteredPhotons = []

		#starting with along track window
		while x_stop < track_max:
			#define right boundary of X window
			if x_start + win_x < track_max:
				x_stop = x_start + win_x
			else:
				x_stop = track_max

			#proceed through depth for each X window
			while y_start >= h_min:
				#define bottom of depth window
				y_stop = y_start - win_h

				#get the photons inside the window
				photons = xy.loc[(xy["height"]<=y_start) & (xy["height"]>=y_stop) & (xy["distance"]>=x_start) & (xy["distance"]<=x_stop)]

				#count number of photons in window
				n_photons = len(photons["height"].values)


				#check if n_photons meets density threshold criteria
				if n_photons > threshold:
					#convert to 2D array by stacking depthwise
					win_photons = np.dstack((photons["height"].values, photons["distance"].values))

					#add photon x/y to the list of filtered photons
					for ph in win_photons:
						filteredPhotons.append(ph)



				#update top of depth window to bottom of last window
				y_start = y_stop

			#reset height window to top of transect
			y_start = h_max

			#update left of X window to right of last window
			x_start = x_stop

		filteredPhotons = np.vstack(filteredPhotons)

		return beamObject(filteredPhotons[:,0],filteredPhotons[:,1])

	def bin(self, win_x=3.0, win_h=3.0):
		#get along track range
		track_min = np.min(self.distance)
		track_max = np.max(self.distance)

		#get height range
		h_min = np.min(self.height)
		h_max = np.max(self.height)

		#combine height and track -> 2D array
		xy = pd.DataFrame.from_dict({"height":self.height,"distance":self.distance})

		#init window
		x_start = track_min
		y_start = h_max

		x_stop = track_min + win_x

		waveforms = []

		while x_stop < track_max:
			#define right boundary of X window
			if x_start + win_x < track_max:
				x_stop = x_start + win_x
			else:
				x_stop = track_max

			window = []

			#proceed through depth for each X window
			while y_start >= h_min:
				#define bottom of depth window
				y_stop = y_start - win_h

				#get number of photons inside current window
				bin_n = len(xy.loc[
								  (xy["height"] >= y_stop) \
								& (xy["height"] <= y_start) \
								& (xy["distance"] >= x_start) \
								& (xy["distance"] <= x_stop)
								].index)

				window.append([y_start, bin_n])

				#update top of depth window to bottom of last window
				y_start = y_stop

			window = np.asarray(window)
			waveforms.append([x_start, waveForm(window[:,0], window[:,1])])

			#reset height window to top of transect
			y_start = h_max

			#update left of X window to right of last window
			x_start = x_stop

		return waveforms

	def plot(self):
		plt.figure(figsize=(6,6))
		plt.scatter(self.distance, self.height, marker="X",s=0.1)
		plt.show()
