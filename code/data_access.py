# MCM 2015 Submission
#
# Authors:
# Nicholas Sharp (nsharp3@vt.edu)
# Brendan Avent
# Saurav Sharma

### Various functions for accessing external datasets

# Local imports
from searching import *
from planes import *
from crashing import*

# Python utilities
from collections import namedtuple
from math import sqrt
from math import isnan 

# External libraries
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

import scipy.stats
from scipy import interpolate

from shapely.geometry import LineString
from shapely.geometry import Point

import netCDF4
import os.path



def get_airports(dim):

    #TODO dummy data for now
    airports = []
   
    
    airports.append(Point_l(14.7,100.6))
    airports.append(Point_l(-3.6,105.3))
    airports.append(Point_l(4.3, 97.5))
    airports.append(Point_l(2.8, 115.2))
    airports.append(Point_l(13.8, 93.2))
    
    '''
    # Other dummy data
    airports.append(Point_l(25.0,145.0))
    airports.append(Point_l(35.0,155.0))
    '''

    return airports

def get_currents(dim_m, dim_l, m):

    print("\n=== Getting current data")

    mydate='20150208';
    server_prefix = 'http://nomads.ncep.noaa.gov:9090/dods/'
    dataset = 'rtofs/rtofs_global'+mydate
    datavarU = 'rtofs_glo_3dz_forecast_daily_uvel'
    datavarV = 'rtofs_glo_3dz_forecast_daily_vvel'

    # Load the U data from cache
    url= server_prefix + dataset + '/' + datavarU
    cacheFilename = 'cache/dataset' + '-' + datavarU + '-%s.dump'

    print("URL is: " + url)
    print("Reading data from cache")
    latU = np.load(cacheFilename%('lat'))
    lonU = np.load(cacheFilename%('lon'))
    dataU = np.load(cacheFilename%('data'))

    # Load the V data from cache
    url= server_prefix + dataset + '/' + datavarV
    cacheFilename = 'cache/dataset' + '-' + datavarV + '-%s.dump'

    print("URL is: " + url)
    print("Reading data from cache")
    latV = np.load(cacheFilename%('lat'))
    lonV = np.load(cacheFilename%('lon'))
    dataV = np.load(cacheFilename%('data'))

    print("Done loading U and V data")
    
    # Interpolators for the data
    print("Creating interpolators")
    '''
    lonPts = np.zeros((dataU.size,1))
    latPts = np.zeros((dataU.size,1))
    UPts = np.zeros((dataU.size,1))
    VPts = np.zeros((dataU.size,1))
    ind = 0
    for i in range(len(lonU)):
        
        if(lonU[i] < dim_l.lon_min or lonU[i] > dim_l.lon_max):
            continue

        for j in range(len(latU)):
        
            if(latU[j] < dim_l.lat_min or latU[j] > dim_l.lat_max):
                continue

            if(not (isnan(dataU[j,i]) or isnan(dataV[j,i]))):
           
                lonPts[ind] = lonU[i]
                latPts[ind] = latU[j]
                UPts[ind] = dataU[j,i]
                VPts[ind] = dataV[j,i]

                ind += 1

    lonPts = lonPts[:ind]
    latPts = latPts[:ind]
    UPts = UPts[:ind]
    VPts = VPts[:ind]

    interpU = interpolate.interp2d(lonPts, latPts, UPts)
    interpV = interpolate.interp2d(lonPts, latPts, UPts)
    '''
    interpU = interpolate.interp2d(lonU, latU, dataU.filled(fill_value=0), kind='quintic')
    interpV = interpolate.interp2d(lonV, latV, dataV.filled(fill_value=0), kind='quintic')

    # Output meshes
    print("Generating output meshes")
    Udata = np.zeros((dim_m.x_res,dim_m.y_res))
    Vdata = np.zeros((dim_m.x_res,dim_m.y_res))

    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res, )
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   
    # Convert meteres/sec to meters/day
    conv = 60*60*24

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            if(not m.is_land(p_y[j], p_x[i])):

                lat, lon = m(p_y[j], p_x[i], inverse=True)
                
                Udata[i,j] = interpU(lat, lon) * conv
                Vdata[i,j] = interpV(lat, lon) * conv


    # TODO FIXME FIXME
    #Udata = Udata * 0
    #Udata = np.ones(Udata.shape) * Vdata.mean() * 40
    #Vdata = np.ones(Vdata.shape) * Vdata.mean() * 40
    #Vdata = Vdata * 0
    
    # Account for the corrdinate change
    return Vdata, Udata
