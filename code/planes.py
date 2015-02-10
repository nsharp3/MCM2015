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
unguided_crash_dev = 1000.0*(10**3)   # The devitaion of the normal distribution for the unguided crash sites
p_intact = 0.13 	# Probability that the type of crash left the plane relatively intact (1-Pr_destructive)
#p_intact = 1.0 	# Probability that the type of crash left the plane relatively intact (1-Pr_destructive)
#p_guided_failure = 0.50 # Probability that a crash is guided (as opposed to unguided)
p_guided_failure = .10 # Probability that a crash is guided (as opposed to unguided)
# Search vehicle params
pr_sp_srfc = 0.90   # Probability that a search plane will spot crash debris in its area
sp_alt = 400        # "Lowest Safe ALTitude" that a plane can fly (in meters)
c_MAD = 3.435e-8    # constant to fit the MAD's functionality curve
pr_sv_srfc = 0.66   # Probability that a search vessel will spot crash debris in its area
pr_sv_snkn = 1.0    # Probability that a search vessel will detect sunken crash in its area
sp_range = 6390.0*(10**3) # The operational range of a search plane (in meters)
sp_dist_per_area = 1000    # The distance that needs to be flown to search a 1km square area (in meters)
sp_cost_tank_gas = 33476    # The cost of a tank of gas for a plane (in dollars)
sb_cost_area = 34.34 # The cost for a boat to search a 1 square kilometer area (in dollars)

def main():

    ## Problem parameters
    
    # The intended flight
    #source = Point_l(14.7,100.6)
    #target = Point_l(-3.6,105.3)
    source = Point_l( -5.083052, 119.606448 ) # Our start
    target = Point_l( 13.610979, 100.753909 ) # Our end
    
    # The problem domain and resolution
    '''
    lat_min = -10.0
    lat_max = 20.0
    lon_min = 90.0
    lon_max = 120.0
    lat_res = 150 
    lon_res = 150
    '''

    # Rectangle
    '''
    lat_min = 20
    lat_max = 40
    lon_min = 140
    lon_max = 160
    lat_res = 200 
    lon_res = 200
    '''
    
    # Case study
    lat_min = -10
    lat_max = 20
    lon_min = 90
    lon_max = 130
    lat_res = 300
    lon_res = 300

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

    print("X resolution (km) is: " + str((dim_m.x_max - dim_m.x_min) / (dim_m.x_res - 1) / 1000.0 ))
    print("Y resolution (km) is: " + str((dim_m.y_max - dim_m.y_min) / (dim_m.y_res - 1) / 1000.0 ))

    # Airport locations
    airports = get_airports(dim)
    airports_m = [Point_m(m(apt.lon,apt.lat)[0], m(apt.lon, apt.lat)[1]) for apt in airports]

    # Get current data
    U_m, V_m = get_currents(dim_m, dim, m)

    # Get depth data
    depth_m = get_depths(dim, dim_m, m)

    ## Run calculations to evaluate the problem
    print("\n=== Evaluating problem\n")

    # Calculate costs
    costs_sp = calc_costs_sp(dim_m, airports_m)
    costs_sb = calc_costs_sb(dim_m)

    init_vals = calc_init_values(dim_m, source_m, target_m, airports_m, m)
    sink_crash = p_intact * init_vals
    surface_crash = (1 - p_intact) * init_vals
    
    # Useful things for later plotting
    (xSource, ySource) = m(source.lon, source.lat)
    (xTarget, yTarget) = m(target.lon, target.lat)
    
    aptCoords = [(p.lon, p.lat) for p in airports]
    aptCoords = np.array(aptCoords)
    xA, yA = m(aptCoords[:,0], aptCoords[:,1])

    # Calculate the likelihood that a search plane finds a crash at
    # each location
    sp_find_prob = search_plane_probabilities(dim_m, surface_crash, sink_crash, depth_m)
    sb_find_prob = search_vessel_probabilities(dim_m, surface_crash, sink_crash)
        
    # Calculate the price to search each area
    price_per_prob_sp = costs_sp / sp_find_prob
    price_per_prob_sb = costs_sb / sp_find_prob

    sp_mean = price_per_prob_sp[np.where(np.logical_not(np.isinf(price_per_prob_sp)))].mean()
    sb_mean = price_per_prob_sb[np.where(np.logical_not(np.isinf(price_per_prob_sb)))].mean()
    print(sp_mean)  
    print(sb_mean)  
 
    price_per_prob_sp[np.where(np.isinf(price_per_prob_sp))] = sp_mean
    price_per_prob_sb[np.where(np.isinf(price_per_prob_sb))] = sb_mean
    
    ## Plot
    # Set up the figure
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])    
    
    m.fillcontinents(zorder=1)
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(0.,90,10.)
    #m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

    # Draw the line representing the flight
    m.plot([xSource, xTarget], [ySource, yTarget], lw=4, c='black', zorder=4)

    # Draw dots on airports
    m.scatter(xA, yA, s=80, marker='^', color='purple', zorder=5)


    # Draw a contour map of costs
    lons, lats = m.makegrid(lat_res, lon_res)
    x, y = m(lons, lats)
    #cs = m.contourf(x, y, depth_m, 50, cmap='Blues_r')
    #pc = m.pcolormesh(x, y, depth_m)
    print(price_per_prob_sp.max())
    print(price_per_prob_sp.min())
    pc = m.pcolormesh(x, y, price_per_prob_sb, cmap='YlOrRd_r',vmax=5e6)
    #cbar = m.colorbar(pc,location='bottom',pad="5%", ticks = [0, init_vals.max()])
    cbar = m.colorbar(pc,location='bottom',pad="5%")
    #plt.title("Distribution of undirected crashes")
    plt.show()
    return

    # Draw a contour map of probabilities
    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
    x, y = np.meshgrid(p_x, p_y)
    '''
    cs = m.contourf(x, y, u_m, cmap="blues")
    cbar = m.colorbar(cs,location='bottom',pad="5%")
    '''

    # Allow the probabilities to flow in the current
    current_mapping = generate_current_mapping(dim_m, U_m, V_m, 1.0/10)
    curr_probs = init_vals.copy()
    maxInitProb = init_vals.max() * 1.2

    for iDay in range(100):

        curr_probs = sink_crash + surface_crash

        # Calculate the likelihood that a search plane finds a crash at
        # each location
        sp_find_prob = search_plane_probabilities(dim_m, surface_crash, sink_crash, depth_m)
        sb_find_prob = search_vessel_probabilities(dim_m, surface_crash, sink_crash)

        # Calculate the price to search each area
        price_per_prob_sp = costs_sp / sp_find_prob
        price_per_prob_sb = costs_sb / sp_find_prob

        #price_per_prob_sp = mask_vals(dim_m, price_per_prob_sp, m, fill=10000)
        #print(price_per_prob_sp.min())
        #price_per_prob_sp[np.where(np.isinf(price_per_prob_sp))] = float('NaN')
        #print(price_per_prob_sp.max())

        # Plotting stuff
        if(iDay % 5 == 0):
            fig = plt.figure(figsize=(8,8))
            
            ax = fig.add_axes([0.1,0.1,0.8,0.8])    
            m.plot([xSource, xTarget], [ySource, yTarget], lw=4, c='black', zorder=4)
            m.scatter(xA, yA, s=80, marker='^', color='purple', zorder=5)
            
            m.fillcontinents(zorder=1)
            m.drawcoastlines()
            m.drawcountries()
            parallels = np.arange(0.,90,10.)
            m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
            
            #cs = m.contourf(x, y, curr_probs, 50, cmap="YlOrRd", vmin=0, vmax=maxInitProb)
            #cs = m.pcolormesh(x, y, curr_probs, cmap="YlOrRd", vmin = 0, vmax=maxInitProb)
            cs = m.pcolormesh(x, y, curr_probs, cmap="YlOrRd", vmax = maxInitProb)
            #cs = m.pcolormesh(x, y, depth_m, cmap="bone")
            cbar = m.colorbar(cs,location='bottom',pad="5%")
            #Q = m.quiver(x, y, U_m, V_m)
            plt.show() 
        
        # Update the probabilities
        surface_crash = apply_current_mapping(dim_m, surface_crash, current_mapping, m)
    

def mask_vals(dim_m, vals, m, fill=0):

    print("Masking and normalizing...")
    
    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            if(m.is_land(p_y[j], p_x[i])):
                vals[i,j] = fill


    return vals


if __name__ == "__main__":
    main()
