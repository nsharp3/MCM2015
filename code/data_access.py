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

    return airports

def get_currents(dim, m):

    print("\n=== Getting current data")

    mydate='20150208';
    server_prefix = 'http://nomads.ncep.noaa.gov:9090/dods/'
    dataset = 'rtofs/rtofs_global'+mydate
    datavarU = 'rtofs_glo_3dz_forecast_daily_uvel'
    datavarV = 'rtofs_glo_3dz_forecast_daily_uvel'

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

    print("Done loading U and V data\n")

    # Interpolators for the data
    interpU = interpolate.interp2d(lonU, latU, dataU)
    interpV = interpolate.interp2d(lonV, latV, dataV)

    # Output meshes
    #Udata = 


    return None, None
