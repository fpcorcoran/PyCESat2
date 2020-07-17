import h5py
import numpy as np
from .Beam import beamObject


class ATL03:
	def __init__(self, filename):
		self.filename = filename
		self.h5 = h5py.File(filename, mode="r")

		self.meta = self.h5.get("METADATA")

		self.strong_beams = [b for b in list(self.h5.keys()) if b in ["gt1r","gt2r","gt3r"]]
		self.weak_beams = [b for b in list(self.h5.keys()) if b in ["gt1l","gt2l","gt3l"]]

		for beam in self.strong_beams + self.weak_beams:
			if "heights" in list(self.h5.get(beam).keys()):
				h = np.asarray(self.h5.get(beam).get("heights").get("h_ph"))
				d = np.asarray(self.h5.get(beam).get("heights").get("dist_ph_along"))

			setattr(self, beam, beamObject(h,d))

