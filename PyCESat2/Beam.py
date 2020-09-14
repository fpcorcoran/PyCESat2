import numpy as np
import pandas as pd
import os
from rasterstats import point_query
from shapely.geometry import Point
from scipy.interpolate import CubicSpline
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from .WaveForm import waveForm
from .utils import *

class beamObject:
    def __init__(self, h, d, lat, lon, ph_conf=None, beam=None):
        super(beamObject, self).__init__()
        self.height = h
        self.distance = d
        self.lat = lat
        self.lon = lon
        self.ph_conf = ph_conf
        self.beam = beam
        
        #if a photon confidence mask is passed to the initializer
        if ph_conf is not None:
            #seperate each landcover mask from the 4D array and take the highest confidence photons (i.e. 4)
            mask = {
                'land' : np.where(ph_conf[:,0] == 4)[0],
                'ocean' : np.where(ph_conf[:,1] == 4)[0],
                'ice' : np.where(ph_conf[:,2] == 4)[0],
                'land ice' : np.where(ph_conf == 4)[0],
                'inland water' : np.where(ph_conf == 4)[0]
                }
            
            #apply each mask to the beam and set the resulting beam objects as attributes
            for _class in mask.keys():
                _mask = mask[_class]
                
                class_beam = beamObject(self.height[_mask], self.distance[_mask],
                                        self.lat[_mask], self.lon[_mask])
                
                setattr(self, _class, class_beam)
                
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
            while y_start > h_min:
                #define bottom of depth window
                y_stop = y_start - win_h

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

    def to_egm08(self, interpolate_geoid=True):
        #path to geoid model file
        geoid = os.path.join(os.path.dirname(__file__),"data","EGM08","EGM2008_mosaic.tif")

        #list of coordinates
        latlon = list(zip(self.lat,self.lon))

        if interpolate_geoid:
            lat_min, lat_max = np.min(self.lat), np.max(self.lat)
            lon_min, lon_max = np.min(self.lon), np.max(self.lon)


            latlon_samples = list(zip(
                                    np.linspace(lat_min, lat_max, 500),
                                    np.linspace(lon_min, lon_max, 500)
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

    def RANSAC(self, name, residual_threshold=None):
        n = len(self.distance)

        if residual_threshold == None:
            ransac = RANSACRegressor(random_state=3).fit(
                                     np.asarray(self.distance).reshape(n,1),
                                     np.asarray(self.height).reshape(n,1))
        else:
            ransac = RANSACRegressor(random_state=3,
                                     residual_threshold=residual_threshold).fit(
                                     np.asarray(self.distance).reshape(n,1),
                                     np.asarray(self.height).reshape(n,1))

        return surfaceBeamObject(self, name=name, model=ransac)


    def bin(self, win_x=3.0, win_h=3.0):
        #get along track range
        track_min = np.min(self.distance)
        track_max = np.max(self.distance)

        #get height range
        h_min = np.min(self.height)
        h_max = np.max(self.height)

        #combine height and track -> 2D array
        xy = pd.DataFrame.from_dict({"height":self.height,"distance":self.distance})
        
        #combine lat/lon -> 2D array
        latlon = pd.DataFrame.from_dict({'lat':self.lat,'lon':self.lon,'distance':self.distance})

        #init window
        x_start = track_min
        y_start = h_max

        x_stop = track_min + win_x

        waveforms = []

        while x_stop < track_max:
            #set up dict to hold kwargs for initializing waveform
            waveform_kwargs = {}

            #define right boundary of X window
            if x_start + win_x < track_max:
                x_stop = x_start + win_x
            else:
                x_stop = track_max
                
            ll = latlon.loc[(latlon['distance'] >= x_start) & (latlon['distance'] <= x_stop)]
            
            waveform_kwargs['start_coords'] = (np.min(ll['lat'].values), np.min(ll['lon'].values))
            waveform_kwargs['end_coords'] = (np.max(ll['lat'].values), np.max(ll['lon'].values))
            
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
            waveform_kwargs['start_distance'] = x_start
            waveform_kwargs['end_distance'] = x_stop
            waveform_kwargs['beam'] = self.beam
            
            waveforms.append(waveForm(window[:,0], window[:,1], **waveform_kwargs))

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
         
 
class surfaces:    
    def __init__(self):
        super(surfaces, self).__init__()
        self.n_surfaces = 0
        self.surfaces = []
        
    def __getitem__(self, name):
        return getattr(self, name)
        
    def add_modeled_surface(self, name, model, distance):

        distance_space = np.linspace(np.min(distance),np.max(distance),1000)
        
        #check if model is a scipy model
        if hasattr(model, 'predict'):
            height = model.predict(distance_space.reshape(1000,1))
            
        #else create numpy vectorized function
        else:
            vmodel = np.vectorize(model)
            height = vmodel(distance_space)
            
        surface = dict(height=height, distance=distance_space, model=model)
        
        setattr(self, name, surface)
        self.n_surfaces += 1
        self.surfaces.append(name)
        
        return self        
 
         
class surfaceBeamObject(beamObject, surfaces):    
    def __init__(self, beamObject, name, model=None):
        super(surfaceBeamObject, self).__init__(beamObject.height, 
                                                beamObject.distance,
                                                beamObject.lat, 
                                                beamObject.lon,
                                                ph_conf = beamObject.ph_conf,
                                                beam = beamObject.beam)
        
        self.add_modeled_surface(name, model, beamObject.distance)
        print("beam:",self.beam)
        
    
    def inliers(self, surface):
        model = self[surface]['model']
        mask = model.inlier_mask_
        
        new_beam = beamObject(self.height[mask] ,self.distance[mask],
                              self.lat[mask], self.lon[mask],
                              ph_conf=None, beam=self.beam)
        
        return surfaceBeamObject(new_beam, surface, model=model)
    
    def outliers(self, surface):
        model = self[surface]['model']
        mask = np.invert(model.inlier_mask_)
        
        new_beam = beamObject(self.height[mask] ,self.distance[mask],
                              self.lat[mask], self.lon[mask],
                              ph_conf=None, beam=self.beam)
        
        return surfaceBeamObject(new_beam, surface, model=model)
    
    def below(self, surface):
        model = self[surface]['model']
        photons = list(zip(self.height, self.distance, self.lat, self.lon))
        
        if hasattr(model, "predict"):
            below = np.asarray([photon for photon in photons if photon[0] < model.predict(photon[1].reshape(1,-1))])
    
        else:
            below = np.asarray([photon for photon in photons if photon[0] < model(photon[1])])
        
        below_beam = beamObject(below[:,0],below[:,1],below[:,2],below[:,3],
                                ph_conf=None, beam=self.beam)
        
        return surfaceBeamObject(below_beam, surface, model=model)
    
    def above(self, surface):
        model = getattr(self,surface)['model']
        photons = list(zip(self.height, self.distance, self.lat, self.lon))
        
        if hasattr(model, "predict"):
            above = np.asarray([photon for photon in photons if photon[0] > model.predict(photon[1].reshape(1,-1))])
    
        else:
            above = np.asarray([photon for photon in photons if photon[0] > model(photon[1])])        
        
        above_beam = beamObject(above[:,0],above[:,1],above[:,2],above[:,3],
                                ph_conf=None, beam=self.beam)

        return surfaceBeamObject(above_beam, surface, model=model)
    
    def between(self, surface1, surface2):
        model1 = getattr(self,surface1)['model']
        model2 = getattr(self,surface2)['model']
        
        model1, model2 = order_surfaces(self.distance, model1, model2)
        
        photons = list(zip(self.height, self.distance, self.lat, self.lon))
        
        if hasattr(model1, 'predict') and hasattr(model2, 'predict'):
            between = [photon for photon in photons if (photon[0] < model1.predict(photon[1].reshape(1,-1))) 
                                                        and
                                                       (photon[0] > model2.predict(photon[1].reshape(1,-1)))]
        
        elif hasattr(model1, 'predict') and (hasattr(model2, 'predict') == False):
            between = [photon for photon in photons if (photon[0] < model1.predict(photon[1].reshape(1,-1))) 
                                                        and
                                                       (photon[0] > model2(photon[1]))]
        
        elif (hasattr(model1, 'predict') == False) and hasattr(model2, 'predict'):
            between = [photon for photon in photons if (photon[0] < model1(photon[1])) 
                                                        and
                                                       (photon[0] > model2.predict(photon[1].reshape(1,-1)))]
        
        else:
            between = [photon for photon in photons if (photon[0] < model1(photon[1])) 
                                                        and
                                                       (photon[0] > model2(photon[1]))]
        
        between = np.asarray(between)
        between_beam = beamObject(between[:,0], between[:,1],between[:,2],between[:,3],
                                  ph_conf=None, beam=self.beam)
        
        between_beam = surfaceBeamObject(between_beam, surface1, model1)
        between_beam.add_modeled_surface(surface2, model2, self.distance)
        
        return between_beam
    
    def outside(self, surface1, surface2):
        model1 = getattr(self,surface1)['model']
        model2 = getattr(self,surface2)['model']
        
        model1, model2 = order_surfaces(self.distance, model1, model2)
        
        photons = list(zip(self.height, self.distance, self.lat, self.lon))
        
        if hasattr(model1, 'predict') and hasattr(model2, 'predict'):
            outside = [photon for photon in photons if (photon[0] > model1.predict(photon[1].reshape(1,-1))) 
                                                        or
                                                       (photon[0] < model2.predict(photon[1].reshape(1,-1)))]
        
        elif hasattr(model1, 'predict') and (hasattr(model2, 'predict') == False):
            outside = [photon for photon in photons if (photon[0] >model1.predict(photon[1].reshape(1,-1))) 
                                                        or
                                                       (photon[0] < model2(photon[1]))]
        
        elif (hasattr(model1, 'predict') == False) and hasattr(model2, 'predict'):
            outside = [photon for photon in photons if (photon[0] > model1(photon[1])) 
                                                        or
                                                       (photon[0] < model2.predict(photon[1].reshape(1,-1)))]
        
        else:
            outside = [photon for photon in photons if (photon[0] > model1(photon[1])) 
                                                        or
                                                       (photon[0] < model2(photon[1]))]
        
        outside = np.asarray(outside)
        outside_beam = beamObject(outside[:,0], outside[:,1], outside[:,2], outside[:,3],
                                  ph_conf=None, beam=self.beam)
        
        outside_beam = surfaceBeamObject(outside_beam, surface1, model1)
        outside_beam.add_modeled_surface(surface2, model2, self.distance)
        
        return outside_beam
  
    def plot(self, surfaces=None, fname=None):
        plt.figure(figsize=(6,6))
        plt.scatter(self.distance, self.height, marker="X",s=0.1, label="photon")
        
        if surfaces == None:
            surfaces = self.surfaces
            
        for surface in surfaces:
            if surface in self.surfaces:
                c = random_hex()
                plt.plot(self[surface]['distance'], self[surface]['height'], 
                          c=c,label=surface)
            else:
                print("Error: {} not a known surface.".format(surface))
            
        plt.ylabel("Elevation (m)")
        plt.xlabel("Along Track Distance")
        plt.legend()
        
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)
        
        return self
