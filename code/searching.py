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
# Returns a lat x lon array with the cost
# of searching each area
def calc_costs(dim, airports):

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
                costs[i,j] = min(costs[i,j], dist * cost_flight)

    print("=== Done calculating costs")

    return costs



# Calculate probabilities of locating the crash with a search vehicle
# Returns a lat x lon array with the probabilities of locating the
# crash in those areas
def search_probabilities(dim_m, srfc_probabilities, snkn_probabilities, depth_data, p_surface_viz, p_snkn_viz):

    print("=== Calculating search vehicle probabilities")

    search_probabilities = np.ones((dim_m.y_res, dim_m.x_res)) * float('inf')

    for i in range(dim_m.y_res):
        for j in range(dim_m.x_res):
            
            # At an (x,y), compute the probability of locating the crash
            Pr_locating_given_srfc = p_surface_viz
            Pr_locating_given_snkn = p_surface_viz / (1 + p_depth_viz*pow(depth_data[i,j],3))
            search_probabilities[i,j] = Pr_locating_given_srfc * srfc_probabilities[i,j] + \
                                        Pr_locating_given_snkn * snkn_probabilities[i,j]


    print("=== Done calculating search vehicle probabilities")

    return search_probabilities

