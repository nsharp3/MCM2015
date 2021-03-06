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

longDist = 10^12 # Arbitrary large distance for geometry calculations

# Calculate the probability that the plane is in each area
# Returns a lat x lon array with the likelihood that the plane
# crashed in that area 
def calc_init_values(dim_m, source, target, airports, m):
    
    print("\n=== Calculating values")

    prob_unguided = calc_init_values_unguided(dim_m, source, target)
    prob_unguided = mask_and_normalize_probs(dim_m, prob_unguided, m)


    prob_guided = calc_init_values_guided(dim_m, source, target, airports, m)
    prob_guided = mask_and_normalize_probs(dim_m, prob_guided, m)

    probTotal = planes.p_guided_failure * prob_guided + (1 - planes.p_guided_failure) * prob_unguided


    return probTotal

# Take a probability map over the search area. Set all areas on land to
# 0 and normalize the rest to sum to 1.
# Modifies the input data
def mask_and_normalize_probs(dim_m, probs, m):

    print("Masking and normalizing...")
    
    p_x = np.linspace(dim_m.x_min, dim_m.x_max, dim_m.x_res)
    p_y = np.linspace(dim_m.y_min, dim_m.y_max, dim_m.y_res)
   
    sumProb = 0

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            if(m.is_land(p_x[j], p_y[i])):
                probs[i,j] = 0

            sumProb += probs[i,j]

    if sumProb < 0.0000001:
        probs = probs * 0.0
    else:
        probs = probs * (1.0 / sumProb)

    return probs

def calc_init_values_unguided(dim, source, target):

    print("\n=== Calculating unguided probabilities")

    p_x = np.linspace(dim.x_min, dim.x_max, dim.x_res)
    p_y = np.linspace(dim.y_min, dim.y_max, dim.y_res)

    probs = np.zeros((dim.x_res, dim.y_res))

    flightLine = LineString([(source.x, source.y), (target.x, target.y)])
    
    # Compute area statistics along each element
    dX = p_y[1] - p_y[0]
    dY = p_x[1] - p_x[0]

    # Transform the element areas to be along the line
    th = np.arctan2(abs(target.y - source.y), abs(source.x - target.x))
    dAlongLine = np.cos(th)*dX + np.sin(th)*dY
    dPerpLine = np.sin(th)*dX + np.cos(th)*dY

    distrib = scipy.stats.norm(0, planes.unguided_crash_dev)
    
    for i in range(dim.x_res):
        for j in range(dim.y_res):
            
            # Coordinates of this point
            pX = p_y[i]
            pY = p_x[j]

            dist = Point(pX,pY).distance(flightLine)

            prob = (dAlongLine / flightLine.length) * \
                (distrib.cdf(dist + 0.5 * dPerpLine) - distrib.cdf(dist - 0.5 * dPerpLine))

            probs[i,j] = prob

    return probs


def calc_init_values_guided(dim, source, target, airports, m):

    print("\n=== Calculating guided probabilities")

    nLinePts = 1000 # Constant controlling granularity along line for computation

    p_x = np.linspace(dim.x_min, dim.x_max, dim.x_res)
    p_y = np.linspace(dim.y_min, dim.y_max, dim.y_res)

    probs = np.zeros((dim.x_res, dim.y_res))

    flightLine = LineString([(source.x, source.y), (target.x, target.y)])
    flightLineInf = line_inf(flightLine)


    # Compute area statistics along each element
    dX = p_y[1] - p_y[0]
    dY = p_x[1] - p_x[0]

    # Transform the element areas to be along the line
    th = np.arctan2(abs(target.y - source.y), abs(source.x - target.x))
    dAlongLine = np.cos(th)*dX + np.sin(th)*dY
    dPerpLine = np.sin(th)*dX + np.cos(th)*dY

    distrib = scipy.stats.norm(0, planes.unguided_crash_dev/10) # Smaller
    distribSmall = scipy.stats.norm(0, planes.unguided_crash_dev/20) # Much tighter distribution used for exterior segments
 
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
                minPort = ind

        return minPort

    # Helper function to process a subset of the line with a single
    # closest airport
    def process_interior_segment(startInd, endInd, airportInd):

        print("Processing interior segment...")
    
        this_seg_probs = np.zeros((dim.x_res, dim.y_res))

        airport = airports[airportInd]
        #airport = Point(airports[airportInd].x, airports[airportInd].y)

        # The line that is just this segment
        segLine = LineString([(lineX[startInd], lineY[startInd]), \
                              (lineX[endInd-1]  , lineY[endInd-1]  )])

        # The line which biscects the triangle and intersects the
        # midpoint of the segment
        midLine = LineString([(airport.x, airport.y), \
                              ( 0.5 * (lineX[startInd] + lineX[endInd-1]),\
                                0.5 * (lineY[startInd] + lineY[endInd-1]))])

        # The probability the the crash happens in this segment
        segCrashProb = (endInd - startInd) / float(nLinePts)

        # Used for contains detections
        triLine1 = LineString([(airport.x, airport.y), \
                               (lineX[startInd], lineY[startInd])])
        triLine2 = LineString([(airport.x, airport.y), \
                               (lineX[endInd-1], lineY[endInd-1])])

        for i in range(dim.x_res):
            for j in range(dim.y_res):
                
                # Coordinates of this point
                pX = p_y[i]
                pY = p_x[j]
                p = Point(pX, pY)

                # The line between this point and the airport
                #toAptLine = LineString([(airport.x, airport.y),(pX, pY)])
                toAptLine = LineString([(pX, pY), (airport.x, airport.y)])
                rayCastLine = LineString([(pX, pY), (0, 0)])

                
                # Check if this is on the correct side of the route
                if(toAptLine.intersects(flightLineInf)):
                    # This point is on the wrong side of the line, so
                    # ignore it
                    continue

                # Make sure the point is inside the triangle
                toAptLineRay = line_ray(toAptLine)

                rayCastCount = 0
                if(rayCastLine.intersects(triLine1)):
                    rayCastCount += 1
                if(rayCastLine.intersects(triLine2)):
                    rayCastCount += 1
                if(rayCastLine.intersects(segLine)):
                    rayCastCount += 1
                
                if(rayCastCount != 1):
                    # The point can't be within the triangle
                    continue

                # We now know that this point is within the fall zone.
                # Assign it a probability from the normal distribution.
                
                # Find the actual intersection along the route
                intersectionPt = toAptLineRay.intersection(segLine)

                # Find how far "along" the triangle we are
                perpDist = p.distance(segLine)
                actDist = p.distance(intersectionPt)

                # What fraction of the route segment does this element include
                triProg = perpDist / midLine.length
                triWidth = segLine.length * (1.01 - triProg)
                segFrac = min(0.1, dAlongLine / triWidth) # Handle clipping due to grid near airport
                #segFrac = dAlongLine / triWidth # Handle clipping due to grid near airport

                # The probability of landing in this element
                #pAlong = (distrib.cdf(actDist + 0.5*dPerpLine) - distrib.cdf(actDist - 0.5*dPerpLine))
                pAlong = abs(distrib.cdf(actDist + 0.5*dPerpLine) - distrib.cdf(actDist - 0.5*dPerpLine)) / (distrib.cdf(triLine1.length))
                pAlong = 1 
                pAcross = segFrac
                prob = pAlong * pAcross

                this_seg_probs[i,j] += prob
                #this_seg_probs[i,j] += 1

        return mask_and_normalize_probs(dim, this_seg_probs, m) * segCrashProb
        
    
    # Helper function to process a subset of the line where the closest
    # airport was either the source or the target
    # Note that this is used hackily backwards for the last segment
    def process_exterior_segment(startInd, endInd, airportInd):
        
        print("Processing exterior segment...")

        airport = airports[airportInd]
        startPt = Point(lineX[startInd], lineY[startInd])

        # The probability the the crash happens in this segment
        segCrashProb = abs(endInd - startInd) / float(nLinePts)
        
        # The line that is just this segment
        segLine = LineString([(lineX[startInd], lineY[startInd]), \
                              (lineX[endInd-1]  , lineY[endInd-1]  )])
   

        # Reflected point, used for clipping end
        dX = lineX[endInd-1] - lineX[startInd]
        dY = lineY[endInd-1] - lineY[startInd]
        refPt = Point(lineX[endInd-1] + dX, lineY[endInd-1] + dY)

        for i in range(dim.x_res):
            for j in range(dim.y_res):
                
                # Coordinates of this point
                pX = p_y[i]
                pY = p_x[j]
                pt = Point(pX, pY)

                # Skip points which are off the end of this line
                if(pt.distance(refPt) < pt.distance(startPt)):
                    continue

                dist = pt.distance(segLine)

                prob = segCrashProb * (dAlongLine / segLine.length) * \
                    (distribSmall.cdf(dist + 0.5 * dPerpLine) - distribSmall.cdf(dist - 0.5 * dPerpLine))

                probs[i,j] += prob
            

    # Walk along the points of the line, stopping to process each
    # segment for which a given airport is closest.
    rangeStartPt = 0
    currPt = 1
    prevClosestAirport = closest_airport(0)
    while True:

        # Process until we see the closest airport change
        while(closest_airport(currPt) == prevClosestAirport and currPt != (nLinePts-1) ):
            currPt += 1


        if(rangeStartPt == 0):
            process_exterior_segment(rangeStartPt, currPt, prevClosestAirport)
        elif(currPt == (nLinePts-1)):
            process_exterior_segment(currPt, rangeStartPt, prevClosestAirport)
        else:
            seg_probs = process_interior_segment(rangeStartPt, currPt, prevClosestAirport)
            probs += seg_probs

        rangeStartPt = currPt
        prevClosestAirport = closest_airport(currPt)
        currPt += 1

        if(currPt == nLinePts):
            break

    return probs

# delT is measured in days
def generate_current_mapping(dim_m, U, V, delT):

    print("\n=== Generating current mapping")

    currMap = {} # (start, end) --> proportion

    dX = float(dim_m.x_max - dim_m.x_min) / (dim_m.x_res)
    dY = float(dim_m.y_max - dim_m.y_min) / (dim_m.y_res)


    largestDel = 0
    sumP = 0

    for i in range(dim_m.x_res):
        for j in range(dim_m.y_res):

            u = abs(U[i,j])
            v = abs(V[i,j])

            uDir = sign(U[i,j])
            vDir = sign(V[i,j])
            
            delHatX = u * delT / dX
            delHatY = v * delT / dY

            if(delHatX > 1 or delHatY > 1):
                print("DelT is too large. Movement = " + str(max(delHatX,delHatY)))

            largestDel = max(largestDel, delHatX)
            largestDel = max(largestDel, delHatY)


            stay = (1-delHatX) * (1-delHatY)
            offX = delHatX * (1-delHatY)
            offY = delHatY * (1-delHatX)
            offXY = delHatX * delHatY

            sumP += stay + offX + offY + offXY

            currMap[((i,j),(i,j))] = stay
            currMap[((i,j),(i + uDir,j))] = offX
            currMap[((i,j),(i,j + vDir))] = offY
            currMap[((i,j),(i + uDir,j + vDir))] = offXY


    print("Largest current delta for interval is " + str(largestDel))

    sumP /= (dim_m.x_res * dim_m.y_res)
    print("Sum prob = " + str(sumP))
    #print(currMap)

    return currMap

def apply_current_mapping(dim_m, old_probs, currMap, m):

    print("\n=== Applying current mapping")

    new_probs = np.zeros(old_probs.shape)

    for inds, frac in currMap.iteritems():

        try:
            new_probs[inds[1]] += old_probs[inds[0]] * frac
        except:
            pass


    new_probs = mask_and_normalize_probs(dim_m, new_probs, m)
    
    return new_probs


# Extends a shapely line very far in either direction to simulate it
# being infinite
def line_inf(line):

    midX = 0.5 * (line.coords[0][0] + line.coords[1][0])
    midY = 0.5 * (line.coords[0][1] + line.coords[1][1])

    dX = (line.coords[1][0] - line.coords[0][0])
    dY = (line.coords[1][1] - line.coords[0][1])

    longLine = LineString([ (midX - dX * longDist, midY - dY * longDist), \
                            (midX + dX * longDist, midY + dY * longDist) ])

    return longLine

# Extends a shapely line very far in one direction to simulate it
# being an infinite ray
def line_ray(line):

    midX = 0.5 * (line.coords[0][0] + line.coords[1][0])
    midY = 0.5 * (line.coords[0][1] + line.coords[1][1])

    dX = (line.coords[1][0] - line.coords[0][0])
    dY = (line.coords[1][1] - line.coords[0][1])

    longLine = LineString([ (midX - dX, midY - dY), \
                            (midX + dX * longDist, midY + dY * longDist) ])

    return longLine

# Extends a shapely line by .99 in either direction to simulate it
# being smalelr
def line_slightly_smaller(line):

    midX = 0.5 * (line.coords[0][0] + line.coords[1][0])
    midY = 0.5 * (line.coords[0][1] + line.coords[1][1])

    dX = (line.coords[1][0] - line.coords[0][0])/2
    dY = (line.coords[1][1] - line.coords[0][1])/2

    longLine = LineString([ (midX - dX * 0.99, midY - dY * 0.99), \
                            (midX + dX * 0.99, midY + dY * 0.99) ])

    #print("in line = " + str(line))
    #print("out line = "  +str(longLine))


    return longLine


def sign(val):

    if val != 0:
        return np.sign(val)

    return 1.0
