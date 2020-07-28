import numpy as np
import pandas as pd
import os
from rasterstats import point_query
from shapely.geometry import Point
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from .WaveForm import waveForm

class beamObject:
	def __init__(self, h, d, lat, lon):
		self.height = h
		self.distance = d
		self.lat = lat
		self.lon = lon

	def filter(self, win_x=3.0, win_y=3.0, threshold=15):
		"""
		Remove low density photon regions.

		Description:
		------------
    	Applies a moving window function of dimension (win_x,win_y) to the beam
		photons. Windows containing less than the threshold number of photons
		are removed. The window starts from the top left track, moving down. On
		reaching the bottom, the window is moved to the right by win_x and
		repositioned at the top.

		NOTE: Edge effects are present on the right side of the track for
			  situations where track_width % win_x != 0. Suggestions for
			  minimizing edge effects are welcome. Please raise as issue before
			  submitting pull request.

    	Parameters:
		-----------
    		win_x (float) 	- X dimension of the moving window filter.

			win_y (float) 	- Y dimension of the moving window filter.

			threshold (int) - Minimum number of photons needed within filter to
			 				  avoid filtering


    	Returns:
		--------
    		beamObject
		"""

		#get along track range
		track_min = np.min(self.distance)
		track_max = np.max(self.distance)

		#get height range
		h_min = np.min(self.height)
		h_max = np.max(self.height)

		#combine height and track -> 2D array
		xy = pd.DataFrame.from_dict({"height":self.height,
									 "distance":self.distance,
									 "lat":self.lat,
									 "lon":self.lon})

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
				y_stop = y_start - win_y

				#get the photons inside the window
				photons = xy.loc[(xy["height"]<=y_start) \
				 & (xy["height"]>=y_stop) & (xy["distance"]>=x_start) \
				 & (xy["distance"]<=x_stop)]

				#count number of photons in window
				n_photons = len(photons["height"].values)


				#check if n_photons meets density threshold criteria
				if n_photons > threshold:
					#convert to 2D array by stacking depthwise
					win_photons = np.dstack((photons["height"].values,
											photons["distance"].values,
											photons["lat"].values,
											photons["lon"].values))

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

		return beamObject(filteredPhotons[:,0],filteredPhotons[:,1],
						  filteredPhotons[:,2],filteredPhotons[:,3])

	def to_egm08(self, interpolate_geoid=True, sample_n=10):
		"""
		Convert ellipsoid referenced elevations to geoid referenced elevations.

		Description:
		------------
    	ATLAS measures photon elevations referenced from the ellipsoid, however
		it is often more convenient to analyze elevations referenced from the
		geoid.

		This function currently supports the EGM08 geoid model. EGM20 will be
		added as an update upon its public release.


    	Parameters:
		-----------
    		interpolate_geoid (bool) - To improve processing speed, the geoid
									   can be interpolated by sampling lat/lon
									   along the ground track. If TRUE, a cubic
									   spline is fit to the sample lat/lons and
									   photon heights are calculated from the s
									   pline. Default=TRUE.

			sample_n (int) 			- The number of sample lat/lons to use for the
							  		  interpolated geoid spline. Default=10.


    	Returns:
		--------
    		beamObject
		"""

		#path to geoid model file
		geoid = os.path.join(os.path.dirname(__file__),"data","EGM08","EGM2008_mosaic.tif")

		latlon = list(zip(self.lat,self.lon))

		if interpolate_geoid:
			lat_min, lat_max = np.min(self.lat), np.max(self.lat)
			lon_min, lon_max = np.min(self.lon), np.max(self.lon)


			latlon_samples = list(zip(
									np.linspace(lat_min, lat_max, sample_n),
									np.linspace(lon_min, lon_max, sample_n)
									))

			undulations=[]
			distances=[]
			for ll in latlon_samples:
				pt = Point(ll)

				undulations.append(point_query(pt, geoid)[0])

				diff = np.array(lat_max, lon_max) - np.asarray(ll)
				distances.append(np.linalg.norm(diff))


			fit = CubicSpline(distances[::-1], undulations[::-1])


			geoid_heights=[]
			for i in range(len(latlon)):
				diff = np.array(lat_max, lon_max) - np.asarray(latlon[i])
				dist = np.linalg.norm(diff)

				geoid_heights.append(self.height[i] + fit(dist))

			return beamObject(np.asarray(geoid_heights), self.distance,
							  self.lat, self.lon)

		else:
			pts=[]

			for lat,lon in latlon:
				pts.append(Point(lat,lon))

			return beamObject(self.height + np.asarray(point_query(pts,geoid)),
							  self.distance, self.lat, self.lon)

	def bin(self, win_x=3.0, win_y=3.0):
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
				y_stop = y_start - win_y

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
		plt.ylabel("Elevation (m)")
		plt.xlabel("Along Track Distance")
		plt.show()
