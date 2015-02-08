# MCM 2015 Submission
#
# Authors:
# Nicholas Sharp (nsharp3@vt.edu)
# Brendan Avent
# Saurav Sharma

# Imports

from collections import namedtuple
from math import sqrt

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

import scipy.stats

from shapely.geometry import LineString
from shapely.geometry import Point


Point = namedtuple('Point', 'lat lon')
Dimension = namedtuple('Dimension', 'lat_min lat_max lat_res lon_min lon_max lon_res') # in lat/lon
Dimension_m = namedtuple('Dimension_m', 'y_min y_max y_res x_min x_max x_res') # in meters

# Constants that define the problem
# TODO resolve the problem of defining distance over lat/lon
cost_flight = 1.0   # Cost, measured in $ / distance from nearest airport
unguided_crash_dev = 300*10^3   # The devitaion of the normal distribution for the unguided crash sites
Pr_intact = 0.5 	# Probability that the type of crash left the plane relatively intact (1-Pr_destructive)
p_debris_float = 0	# Proportion of debris that floats after an "intact" type crash
search_plane_visibility = .95	# Probability that a search plane will spot debris in its area
search_plane_depth = .5 	# Parameter to control the cubic decrase in a search plane's usefulness of spotting underwater (intact) crash

def main():

    ## Problem parameters
    
    # The intended flight
    source = Point(14.7,100.6)
    target = Point(-3.6,105.3)
    
    # The problem domain and resolution
    lat_min = -10
    lat_max = 20
    lon_min = 90
    lon_max = 120
    lat_res = 100
    lon_res = 100
    dim = Dimension(lat_min, lat_max, lat_res, lon_min, lon_max, lon_res)
    
    # create Basemap instance.
    m = Basemap(projection='mill',\
                llcrnrlat=lat_min,urcrnrlat=lat_max,\
                llcrnrlon=lon_min,urcrnrlon=lon_max,\
                resolution='l')

    # Convert the problem domain to meters
    x_min, y_min = m(lon_min, lat_min)
    x_max, y_max = m(lon_max, lat_max)
    dim_m = Dimension_m(y_min, y_max, lat_res, x_min, x_max, lon_res)
    source_m = (m(source.lon,source.lat)[0], m(source.lon, source.lat)[1])
    target_m = (m(target.lon,target.lat)[0], m(target.lon, target.lat)[1])

    # Airport locations
    airports = get_airports(dim)


    ## Run calculations to evaluate the problem
    print("=== Evaluating problem\n")

    costs = calc_costs(dim, airports)

    init_vals = calc_init_values(dim_m, source_m, target_m)
   
    
    
    ## Plot
    
    # Set up the figure
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])    
   
    m.fillcontinents(zorder=1)
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(0.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    
    # Draw the line representing the flight
    (xSource, ySource) = m(source.lon, source.lat)
    (xTarget, yTarget) = m(target.lon, target.lat)
    m.plot([xSource, xTarget], [ySource, yTarget], lw=4, c='black', zorder=4)
    
    # Draw dots on airports
    aptCoords = [(p.lon, p.lat) for p in airports]
    aptCoords = np.array(aptCoords)
    x, y = m(aptCoords[:,0], aptCoords[:,1])
    m.scatter(x, y, 30, marker='s', color='red', zorder=5)


    # Draw a contour map of costs
    lons, lats = m.makegrid(lat_res, lon_res)
    x, y = m(lons, lats)
    cs = m.contourf(x, y, init_vals)

    
    plt.show()


def get_airports(dim):

    #TODO dummy data for now
    airports = []
    
    airports.append(Point(14.7,100.6))
    airports.append(Point(-3.6,105.3))
    airports.append(Point(2.8, 115.2))
    airports.append(Point(13.8, 93.2))

    return airports

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

# Calculate the probability that the plane is in each area
# Returns a lat x lon array with the likelihood that the plane
# crashed in that area 
def calc_init_values(dim, source, target):
    
    print("=== Calculating values")

    prob_unguided = calc_init_values_unguided(dim, source, target)

    return prob_unguided


def calc_init_values_unguided(dim, source, target):

    print("=== Calculating unguided values")

    p_x = np.linspace(dim.x_min, dim.x_max, dim.x_res)
    p_y = np.linspace(dim.y_min, dim.y_max, dim.y_res)

    probs = np.zeros((dim.x_res, dim.y_res))

    flightLine = LineString[(source.x, source.y), (target.x, target.y)]
    
    # Compute area statistics along each element
    dX = p_x[1] - p_x[0]
    dY = p_y[1] - p_y[0]

    # Transform the element areas to be along the line
    th = np.arctan2(abs(target.y - source.y), abs(source.x - target.x))
    dAlongLine = np.cos(th)*dX + np.sin(th)*dY
    dPerpLine = np.sin(th)*dX + np.cos(th)*dY

    distrib = scipy.stats.norm(0, unguided_crash_dev)
    
    for i in range(dim.lat_res):
        for j in range(dim.lon_res):
            
            # Coordinates of this point
            pX = p_x[i]
            pY = p_x[j]

            dist = Point(pX,pY).distance(flightLine)

            prob = dAlongLine / flightLine.length * \
                (distrib.cdf(dist + 0.5 * dPerpLine) + distrib.cdf(dist - 0.5 * dPerpLine))

            probs[i,j] = prob

    return prob

# Calculate probabilities of finding the crash with a search plane
# Returns a lat x lon array with the probabilities of finding the
# crash in that area
def search_plane_probabilities(dim, crash_probabilities, depth_data):

    print("=== Calculating search plane probabilities")

    search_probabilities = np.ones((dim.lat_res, dim.lon_res)) * float('inf')

    for i in range(dim.lat_res):
        for j in range(dim.lon_res):
        	Pr_locating_given_crash_and_intact_at_surface = (1-p_debris_float)*search_plane_visibility
        	Pr_locating_given_crash_and_intact_at_depth = p_debris_float*(1 / (1 + search_plane_depth*pow(depth_data[i,j],3)))
        	Pr_locating_given_crash_and_intact = Pr_locating_given_crash_and_intact_at_depth + \
                                                Pr_locating_given_crash_and_intact_at_surface
        	Pr_locating_given_crash_and_destructive = search_plane_visibility
            
            search_probabilities[i,j] = (Pr_locating_given_crash_and_intact * Pr_intact) + \
            							(Pr_locating_given_crash_and_destructive * (1-Pr_intact))


    print("=== Done calculating search plane probabilities")

    return search_probabilities

if __name__ == "__main__":
    main()
