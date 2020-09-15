import h5py
import numpy as np
import warnings
from .Beam import beamObject


class ATL03:
    def __init__(self, filename, ignore_warnings=False, keep_empty_beams=False):
        #Store basic features of file, including name and raw h5 data
        self.filename = filename
        self.h5 = h5py.File(filename, mode="r")

        #Store metadata for easy access - unpacking needed here
        self.meta = self.h5.get("METADATA")

        #get list of beam names
        self.strong_beams = [b for b in list(self.h5.keys()) if b in ["gt1r","gt2r","gt3r"]]
        self.weak_beams = [b for b in list(self.h5.keys()) if b in ["gt1l","gt2l","gt3l"]]

        #fill each beam object with corresponding photon data
        for beam in self.strong_beams + self.weak_beams:
            if "heights" in list(self.h5.get(beam).keys()):
                h = np.asarray(self.h5.get(beam).get("heights").get("h_ph"))
                d = np.asarray(self.h5.get(beam).get("heights").get("dist_ph_along"))
                lat = np.asarray(self.h5.get(beam).get("heights").get("lat_ph"))
                lon = np.asarray(self.h5.get(beam).get("heights").get("lon_ph"))
                
                #photon confidence mask used for basic landcover classification
                ph_conf = np.asarray(self.h5.get(beam).get('heights').get('signal_conf_ph'))
                
                setattr(self, beam, beamObject(h,d,lat,lon,ph_conf=ph_conf,beam=beam))

            else:
                #if empty beams are to be kept, fill with NaN and None where appropriate
                if keep_empty_beams:
                    #raise warning by default
                    if ignore_warnings:
                        setattr(self, beam, beamObject([np.nan],[np.nan],[np.nan],[np.nan],
                                                       ph_conf=None, beam=beam))
                    else:
                        warnings.warn("Beam {0} has no photon data".format(beam))
                        setattr(self, beam, beamObject([np.nan],[np.nan],[np.nan],[np.nan],
                                                       ph_conf=None, beam=beam))
                        
                #by default, empty beams are removed from ATL03 object
                else:
                    #raise warning by default
                    if ignore_warnings:
                        if beam in self.strong_beams:
                            self.strong_beams.remove(beam)
                        else:
                            self.weak_beams.remove(beam)
                    else:
                        if beam in self.strong_beams:
                            self.strong_beams.remove(beam)
                            warnings.warn("Beam {0} has no photon data".format(beam))
                        else:
                            self.weak_beams.remove(beam)
                            warnings.warn("Beam {0} has no photon data".format(beam))
                    
                    
    def __getitem__(self, key):
        return getattr(self, key)
