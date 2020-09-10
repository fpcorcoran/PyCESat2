import h5py
import numpy as np
import warnings
from .Beam import beamObject


class ATL03:
    def __init__(self, filename, ignore_warnings=False):
        self.filename = filename
        self.h5 = h5py.File(filename, mode="r")

        self.meta = self.h5.get("METADATA")

        self.strong_beams = [b for b in list(self.h5.keys()) if b in ["gt1r","gt2r","gt3r"]]
        self.weak_beams = [b for b in list(self.h5.keys()) if b in ["gt1l","gt2l","gt3l"]]

        for beam in self.strong_beams + self.weak_beams:
            if "heights" in list(self.h5.get(beam).keys()):
                h = np.asarray(self.h5.get(beam).get("heights").get("h_ph"))
                d = np.asarray(self.h5.get(beam).get("heights").get("dist_ph_along"))
                lat = np.asarray(self.h5.get(beam).get("heights").get("lat_ph"))
                lon = np.asarray(self.h5.get(beam).get("heights").get("lon_ph"))
                
                ph_conf = np.asarray(self.h5.get(beam).get('heights').get('signal_conf_ph'))
                
                setattr(self, beam, beamObject(h,d,lat,lon,ph_conf=ph_conf))

            else:
                if ignore_warnings:
                    setattr(self, beam, beamObject([np.nan],[np.nan],[np.nan],[np.nan]))
                else:
                    warnings.warn("Beam {0} has no photon data".format(beam))
                    setattr(self, beam, beamObject([np.nan],[np.nan],[np.nan],[np.nan]))
                    
    def __getitem__(self, key):
        return getattr(self, key)
