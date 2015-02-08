# MCM 2015 Submission
#
# Authors:
# Nicholas Sharp (nsharp3@vt.edu)
# Brendan Avent
# Saurav Sharma

### Functions relating to the locations the plane may crash
### and the distribution of the wreckage over time

# Local imports
from data_access import *
from planes import *
import planes
from searching import*

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

# Calculate the probability that the plane is in each area
# Returns a lat x lon array with the likelihood that the plane
# crashed in that area 
def calc_init_values(dim_m, source, target, m):
    
    print("=== Calculating values")

    prob_unguided = calc_init_values_unguided(dim_m, source, target)
    mask_and_normalize_probs(dim_m, prob_unguided, m)

    return prob_unguided

# Take a probability map over the search area. Set all areas on land to
# 0 and normalize the rest to sum to 1.
# Modifies the input data
def mask_and_normalize_probs(dim_m, probs, m):

    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   
    sumProb = 0

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            if(m.is_land(p_y[j], p_x[i])):
                probs[i,j] = 0

            sumProb += probs[i,j]

    probs = probs * (1 / sumProb)


def calc_init_values_unguided(dim, source, target):

    print("=== Calculating unguided probabilities")

    p_x = np.linspace(dim.x_min, dim.x_max, dim.x_res)
    p_y = np.linspace(dim.y_min, dim.y_max, dim.y_res)

    probs = np.zeros((dim.x_res, dim.y_res))

    flightLine = LineString([(source.x, source.y), (target.x, target.y)])
    
    # Compute area statistics along each element
    dX = p_x[1] - p_x[0]
    dY = p_y[1] - p_y[0]

    # Transform the element areas to be along the line
    th = np.arctan2(abs(target.y - source.y), abs(source.x - target.x))
    dAlongLine = np.cos(th)*dX + np.sin(th)*dY
    dPerpLine = np.sin(th)*dX + np.cos(th)*dY

    distrib = scipy.stats.norm(0, planes.unguided_crash_dev)
    
    for i in range(dim.x_res):
        for j in range(dim.y_res):
            
            # Coordinates of this point
            pX = p_x[i]
            pY = p_x[j]

            dist = Point(pX,pY).distance(flightLine)

            prob = (dAlongLine / flightLine.length) * \
                (distrib.cdf(dist + 0.5 * dPerpLine) - distrib.cdf(dist - 0.5 * dPerpLine))

            probs[i,j] = prob

    return probs


def calc_init_values_guided(dim, source, target, airports):

    print("=== Calculating guided probabilities")

    nLinePts = 1000 # Constant controlling granularity along line for computation

    p_x = np.linspace(dim.x_min, dim.x_max, dim.x_res)
    p_y = np.linspace(dim.y_min, dim.y_max, dim.y_res)

    probs = np.zeros((dim.x_res, dim.y_res))

    flightLine = LineString([(source.x, source.y), (target.x, target.y)])
    
    # Compute area statistics along each element
    dX = p_x[1] - p_x[0]
    dY = p_y[1] - p_y[0]

    # Transform the element areas to be along the line
    th = np.arctan2(abs(target.y - source.y), abs(source.x - target.x))
    dAlongLine = np.cos(th)*dX + np.sin(th)*dY
    dPerpLine = np.sin(th)*dX + np.cos(th)*dY

    distrib = scipy.stats.norm(0, unguided_crash_dev)
 
    lineX = np.linspace(source.x, target.x, nLinePts)
    lineY = np.linspace(source.y, target.y, nLinePts)

    # Helper function to find the closest airport to a point
    def closest_airport(pInd):

        minDist = float('inf')
        minPort = -1

        for ind,apt in enumerate(airports):

            dX = lineX[pInd] - apt.x
            dY = lineY[pInd] - apt.y
            dist = sqrt(dX*dX + dY*dY)
            if(dist < minDist):
                minDist = dist
                minPort =    ind

        return minPort

    # Helper function to process a subset of the line with a single
    # closest airport
    def process_segment(startInd, endInd, airport):

        print("hi mom")
        


    # Walk along the points of the line, stopping to process each
    # segment for which a given airport is closest.
    rangeStartPt = 0
    currPt = 1
    prevClosestAirport = closest_airport(0)
    while True:

        # Process until we see the closest airport change
        while(closest_airport(currPt) == prevClosestAirport and currPt != (nLinePts-1) ):
            currPt += 1

        process_segment(rangeStartPt, currPt, prevClosestAirport)

        rangeStartPt = currPt
        prevClosestAirport = closest_airport(currPt)
        currPt += 1



    for iLinePt in range(1,nLinePts):
        print("hi")


    for i in range(dim.x_res):
        for j in range(dim.y_res):
            
            # Coordinates of this point
            pX = p_x[i]
            pY = p_x[j]


            probs[i,j] = prob

    return probs
