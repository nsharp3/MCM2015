# MCM 2015 Submission
#
# Authors:
# Nicholas Sharp (nsharp3@vt.edu)
# Brendan Avent
# Saurav Sharma

### Main runner function and plotting routines

# Imports

# Local imports
from searching import *
from data_access import *
from crashing import*

# Python utilities
from collections import namedtuple
from math import sqrt

# External libraries
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

import scipy.stats

from shapely.geometry import LineString
from shapely.geometry import Point


Point_l = namedtuple('Point', 'lat lon')
Point_m = namedtuple('Point_m', 'y x')
Dimension_l = namedtuple('Dimension', 'lat_min lat_max lat_res lon_min lon_max lon_res') # in lat/lon
Dimension_m = namedtuple('Dimension_m', 'y_min y_max y_res x_min x_max x_res') # in meters

# Constants that define the problem
# TODO resolve the problem of defining distance over lat/lon
cost_flight = 1.0   # Cost, measured in $ / distance from nearest airport
unguided_crash_dev = 100000.0*(10^3)   # The devitaion of the normal distribution for the unguided crash sites
Pr_intact = 0.13 	# Probability that the type of crash left the plane relatively intact (1-Pr_destructive)
p_debris_float = 0	# Proportion of debris that floats after an "intact" type crash
p_guided_failure = 0.1 # Probability that a crash is guided (as opposed to unguided)
# Search vehicle params
pr_sp_srfc = 0.90   # Probability that a search plane will spot crash debris in its area
LSALT = 400     # "Lowest Safe ALTitude" that a plane can fly (in meters)
c_MAD = 3.435e-8    # constant to fit the MAD's functionality curve
pr_sb_srfc = 0.66   # Probability that a search boat will spot crash debris in its area
pr_sb_snkn = 1.0    # Probability that a search boat will detect sunken crash in its area

def main():

    ## Problem parameters
    
    # The intended flight
    source = Point_l(14.7,100.6)
    target = Point_l(-3.6,105.3)
    
    # The problem domain and resolution
    lat_min = -10
    lat_max = 20
    lon_min = 90
    lon_max = 120
    lat_res = 100 
    lon_res = 100
    dim = Dimension_l(lat_min, lat_max, lat_res, lon_min, lon_max, lon_res)
    
    # create Basemap instance.
    m = Basemap(projection='mill',\
                llcrnrlat=lat_min,urcrnrlat=lat_max,\
                llcrnrlon=lon_min,urcrnrlon=lon_max,\
                resolution='l')

    # Convert the problem domain to meters
    x_min, y_min = m(lon_min, lat_min)
    x_max, y_max = m(lon_max, lat_max)
    dim_m = Dimension_m(y_min, y_max, lat_res, x_min, x_max, lon_res)
    source_m = Point_m(m(source.lon,source.lat)[0], m(source.lon, source.lat)[1])
    target_m = Point_m(m(target.lon,target.lat)[0], m(target.lon, target.lat)[1])

    # Airport locations
    airports = get_airports(dim)
    airports_m = [Point_m(m(apt.lon,apt.lat)[0], m(apt.lon, apt.lat)[1]) for apt in airports]

    # Get current data
    U_m, V_m = get_currents(dim_m, dim, m)

    ## Run calculations to evaluate the problem
    print("\n=== Evaluating problem\n")

    costs = calc_costs(dim, airports)

    init_vals = calc_init_values(dim_m, source_m, target_m, airports_m, m)
   
    
    
    ## Plot
    
    # Set up the figure
    '''
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])    
    
    m.fillcontinents(zorder=1)
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(0.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    '''

    # Draw the line representing the flight
    (xSource, ySource) = m(source.lon, source.lat)
    (xTarget, yTarget) = m(target.lon, target.lat)
    #m.plot([xSource, xTarget], [ySource, yTarget], lw=4, c='black', zorder=4)

    # Draw dots on airports
    aptCoords = [(p.lon, p.lat) for p in airports]
    aptCoords = np.array(aptCoords)
    xA, yA = m(aptCoords[:,0], aptCoords[:,1])
    #m.scatter(xA, yA, 30, marker='s', color='red', zorder=5)


    # Draw a contour map of costs
    '''
    lons, lats = m.makegrid(lat_res, lon_res)
    x, y = m(lons, lats)
    cs = m.contourf(x, y, init_vals)
    '''

    # Draw a contour map of probabilities
    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
    x, y = np.meshgrid(p_x, p_y)
    '''
    cs = m.contourf(x, y, u_m, cmap="blues")
    cbar = m.colorbar(cs,location='bottom',pad="5%")
    '''

    # Allow the probabilities to flow in the current
    current_mapping = generate_current_mapping(dim_m, U_m, V_m, 1.0/4)
    curr_probs = init_vals.copy()

    for iDay in range(100):

        # Plotting stuff
        fig = plt.figure(figsize=(8,8))
        
        ax = fig.add_axes([0.1,0.1,0.8,0.8])    
        m.plot([xSource, xTarget], [ySource, yTarget], lw=4, c='black', zorder=4)
        m.scatter(xA, yA, 30, marker='s', color='red', zorder=5)
        
        m.fillcontinents(zorder=1)
        m.drawcoastlines()
        m.drawcountries()
        parallels = np.arange(0.,90,10.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        
        cs = m.contourf(x, y, curr_probs, 50, cmap="YlOrRd")
        #cs = m.pcolormesh(x, y, curr_probs, cmap="YlOrRd")
        cbar = m.colorbar(cs,location='bottom',pad="5%")
      
        plt.savefig("output/probs_"+str(iDay)+".png")
        plt.close()
        
        # Update the probabilities
        curr_probs = apply_current_mapping(dim_m, curr_probs, current_mapping, m)
    



if __name__ == "__main__":
    main()
