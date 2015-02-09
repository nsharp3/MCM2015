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
import urllib2
import StringIO
import csv
import pickle as pickle



def get_airports(dim):

    #TODO dummy data for now
    airports = []
   
    '''
    airports.append(Point_l(14.7,100.6))
    airports.append(Point_l(-3.6,105.3))
    airports.append(Point_l(4.3, 97.5))
    airports.append(Point_l(2.8, 115.2))
    airports.append(Point_l(13.8, 93.2))
    '''

    '''
    # Other dummy data
    airports.append(Point_l(25.0,145.0))
    airports.append(Point_l(35.0,155.0))
    '''

    # Case study data
    airports.append(Point_l(3.1387103, 101.682099528432)) # Takeoff, kuala lumpur
    #airports.append(Point_l(39.9059093, 116.3913489))  # Landing
    #airports.append(Point_l(3.5896654, 98.6738261))
    #airports.append(Point_l(5.8927453, 95.3225751))
    airports.append(Point_l(4.2166144, 100.6952825))
    airports.append(Point_l(7.715984, 98.322126372155))
    airports.append(Point_l(7.065145, 125.561038))
    airports.append(Point_l( 5.886214, 116.046878 ))
    airports.append(Point_l( -8.746963, 115.189944 ))
    airports.append(Point_l( -5.083052, 119.606448 )) # Our start
    airports.append(Point_l( 13.610979, 100.753909 ) ) # Our end
    #airports.append(Point_l(  ))
    #airports.append(Point_l(  ))

    return airports

# See http://polar.ncep.noaa.gov/global/examples/usingpython.shtml
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
    interpU = interpolate.interp2d(lonU, latU, dataU.filled(fill_value=0), kind='cubic')
    interpV = interpolate.interp2d(lonV, latV, dataV.filled(fill_value=0), kind='cubic')

    # Output meshes
    print("Generating output meshes")
    Udata = np.zeros((dim_m.x_res,dim_m.y_res))
    Vdata = np.zeros((dim_m.x_res,dim_m.y_res))

    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   
    # Convert meteres/sec to meters/day
    conv = 60*60*24

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            #if(not m.is_land(p_y[j], p_x[i])):

                lon, lat = m(p_x[i], p_y[j], inverse=True)
                
                Udata[j,i] = interpU(lon, lat) * conv
                Vdata[j,i] = interpV(lon, lat) * conv


    # Account for the corrdinate change
    return Vdata, Udata

# Returns depths in meters
# Adapted from http://oceanpython.org/2013/03/21/bathymetry-topography-srtm30/
def get_depths(dim_l, dim_m, m):

    print("\n=== Getting depth data")

    # Definine the domain of interest
    minlat = dim_l.lat_min
    maxlat = dim_l.lat_max
    minlon = dim_l.lon_min
    maxlon = dim_l.lon_max

    cacheFilename = 'cache/depth_%0.2f_%0.2f_%0.2f_%0.2f.p'%(minlat,maxlat,minlon,maxlon)



    if(not os.path.isfile(cacheFilename)):
    #if(True):

        # Read data from: http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.html
        print("Data cache not found, downloading...")
        skipRes = 10
        url = 'http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.csv?topo[(' \
                                    +str(maxlat)+'):'+str(skipRes)+':('+str(minlat)+')][('+str(minlon)+'):'+str(skipRes)+':('+str(maxlon)+')]'
        print("url: " + url)
        response = urllib2.urlopen(url)
                                  
        print("Done downloading.")

        data = StringIO.StringIO(response.read())
        print(data)
         
        r = csv.DictReader(data,dialect=csv.Sniffer().sniff(data.read(1000)))
        data.seek(0)

        # Initialize variables
        lat, lon, topo = [], [], []
         
        # Loop to parse 'data' into our variables
        # Note that the second row has the units (i.e. not numbers). Thus we implement a
        # try/except instance to prevent the loop for breaking in the second row (ugly fix)
        for row in r:
            try:
                lat.append(float(row['latitude']))
                lon.append(float(row['longitude']))
                topo.append(float(row['topo']))
            except:
                print 'Row '+str(row)+' is a bad...'
         
        # Convert 'lists' into 'numpy arrays'
        lat  = np.array(lat,  dtype='float')
        lon  = np.array(lon,  dtype='float')
        topo = np.array(topo, dtype='float')
        
        latPts = set(lat)
        lonPts = set(lon)
        latPts = sorted([x for x in latPts])
        lonPts = sorted([x for x in lonPts])

        latIndDict = {}
        lonIndDict = {}
        for ind,val in enumerate(latPts):
            latIndDict[val] = ind
        for ind,val in enumerate(lonPts):
            lonIndDict[val] = ind

        latPts = np.array(latPts)
        lonPts = np.array(lonPts)

        topoPts = np.zeros((len(latPts), len(lonPts)))
        for i in range(len(lat)):
            lonInd = lonIndDict[lon[i]]
            latInd = latIndDict[lat[i]]
            topoPts[latInd,lonInd] = topo[i]
        
 
        # Make an empty 'dictionary'... place the 3 grids in it.
        TOPO = {}
        TOPO['lats']=latPts
        TOPO['lons']=lonPts
        TOPO['topo']=topoPts
         
        # Save (i.e. pickle) the data for later use
        # This saves the variable TOPO (with all its contents) into the file: topo.p
        pickle.dump(TOPO, open(cacheFilename,'wb'))
        print("Data cached for later use")
     
    TOPO = pickle.load(open(cacheFilename,'r'))
    print("Loading cache file " + cacheFilename)
    latPts = TOPO['lats']
    lonPts = TOPO['lons']
    topoPts = TOPO['topo']

    # Interpolators for the data
    print("Creating interpolators")
    interp = interpolate.interp2d(lonPts, latPts, topoPts, kind='cubic')

    # Output meshes
    print("Generating output mesh")
    data = np.zeros((dim_m.x_res,dim_m.y_res))

    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   
    # Convert meteres/sec to meters/day

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            #if(not m.is_land(p_y[j], p_x[i])):

                lon, lat = m(p_x[i], p_y[j], inverse=True)
                
                #data[i,j] = min(interp(lat, lon),0)
                data[j,i] = min(interp(lon, lat),0)



    return data

