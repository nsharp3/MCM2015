# MCM 2015 Submission
#
# Authors:
# Nicholas Sharp (nsharp3@vt.edu)
# Brendan Avent
# Saurav Sharma

### Functions relating to the cost of searching a given area,
### as well as the probability of finding the plane there


# Local imports
from data_access import *
from planes import *
import planes
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

# Calculate the cost of searching an area
# Returns a lat x lon array with the cost per square km
# of searching each area
def calc_costs_sp(dim, airports):

    print("=== Calculating costs")

    p_lat = np.linspace(dim.lat_min, dim.lat_max, dim.lat_res)
    p_lon = np.linspace(dim.lon_min, dim.lon_max, dim.lon_res)

    costs = np.ones((dim.lat_res, dim.lon_res)) * float('inf')

    # For now, cost is simply the distance to the closest airport
    for airport in airports:

        for i in range(dim.lat_res):
            for j in range(dim.lon_res):

                dX = p_lat[i] - airport.lat
                dY = p_lon[j] - airport.lon
                dist = sqrt(dX*dX + dY*dY)

                if(dist > planes.sp_range * 0.45):
                    # Too far away to search effectively with a plane
                    continue

                cost = planes.sp_dist_per_area * planes.sp_cost_tank_gas / (planes.sp_range - 2 * dist)
                costs[i,j] = min(costs[i,j], cost)

    print("=== Done calculating costs")

    return costs

# Calculate the cost of searching an area
# Returns a lat x lon array with the cost per square km
# of searching each area
def calc_costs_sp(dim_m, airports):

    print("=== Calculating costs for plane")
    
    costs = np.ones((dim_m.x_res, dim_m.y_res)) * float('inf')
    
    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   
    for airport in airports:

        for i in range(dim_m.x_res):
            for j in range(dim_m.y_res):

                dX = p_x[i] - airport.x
                dY = p_y[j] - airport.y
                dist = sqrt(dX*dX + dY*dY)
        
                if(dist > planes.sp_range * 0.45):
                    # Too far away to search effectively with a plane
                    continue

                
                cost = planes.sp_dist_per_area * planes.sp_cost_tank_gas / (planes.sp_range - 2 * dist)
                costs[i,j] = min(costs[i,j], cost)

    print("=== Done calculating costs")

    return costs

# Calculates the cost of searching an area with a boat
# Given in dollars per square km
def calc_costs_sb(dim_m):
    print("=== Calculating costs for boat")
    return np.ones((dim_m.x_res, dim_m.y_res)) * planes.sb_cost_area

# Calculate probabilities of locating the crash with a search plane
# Returns a lat x lon array with the probabilities of locating the
# crash in those areas
def search_plane_probabilities(dim_m, srfc_probabilities, snkn_probabilities, depth_data):

    print("=== Calculating search vehicle probabilities")

    search_probabilities = np.ones((dim_m.y_res, dim_m.x_res)) * float('inf')

    for i in range(dim_m.y_res):
        for j in range(dim_m.x_res):
            
            # At an (x,y), compute the probability of locating the crash
            Pr_locating_given_srfc = pr_sp_srfc
            Pr_locating_given_snkn = pr_sp_srfc / (1 + c_MAD*pow(sp_alt+depth_data[i,j],3))
            search_probabilities[i,j] = Pr_locating_given_srfc * srfc_probabilities[i,j] + \
                                        Pr_locating_given_snkn * snkn_probabilities[i,j]


    print("=== Done calculating search vehicle probabilities")

    return search_probabilities


# Calculate probabilities of locating the crash with a search vessel
# Returns a lat x lon array with the probabilities of locating the
# crash in those areas
def search_vessel_probabilities(dim_m, srfc_probabilities, snkn_probabilities):

    print("=== Calculating search vehicle probabilities")

    search_probabilities = np.ones((dim_m.y_res, dim_m.x_res)) * float('inf')

    for i in range(dim_m.y_res):
        for j in range(dim_m.x_res):
            
            # At an (x,y), compute the probability of locating the crash
            search_probabilities[i,j] = pr_sv_srfc * srfc_probabilities[i,j] + \
                                        pr_sv_snkn * snkn_probabilities[i,j]


    print("=== Done calculating search vehicle probabilities")

    return search_probabilities
