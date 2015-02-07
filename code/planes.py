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


Point = namedtuple('Point', 'lat lon')
Dimension = namedtuple('Dimension', 'lat_min lat_max lat_res lon_min lon_max lon_res')

# Constants that define the problem
# TODO resolve the problem of defining distance over lat/lon
cost_flight = 1.0   # Cost, measured in $ / distance from nearest airport
Pr_intact = 0.5 	# Probability that the type of crash left the plane relatively intact (1-Pr_destructive)
alpha = 0	# Proportion of debris that floats after an "intact" type crash
search_plane_visibility = .95	# Probability that a search plane will spot debris in its area
search_plane_depth = .5 	# Parameter to control the cubic decrase in a search plane's usefulness of spotting underwater (intact) crash

def main():

    ## Problem parameters
    
    # The intended flight
    source = Point(4.7,93.6)

    target = Point(15.6,112.3)
    
    # The problem domain and resolution
    lat_min = -10
    lat_max = 20
    lon_min = 90
    lon_max = 120
    lat_res = 500
    lon_res = 500
    dim = Dimension(lat_min, lat_max, lat_res, lon_min, lon_max, lon_res)

    # Airport locations
    airports = get_airports(dim)



    ## Run calculations to evaluate the problem
    print("=== Evaluating problem\n")


    costs = calc_costs(dim, airports)

    init_vals = calc_init_values()
   

    
    ## Plot
    
    # Set up the figure
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    
    
    # create polar stereographic Basemap instance.
    m = Basemap(projection='mill',\
                llcrnrlat=lat_min,urcrnrlat=lat_max,\
                llcrnrlon=lon_min,urcrnrlon=lon_max,\
                resolution='l')
    
    m.fillcontinents()
    m.drawcoastlines()
    m.drawcountries()
    parallels = np.arange(0.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
   
    # Draw a contour map of costs
    lons, lats = m.makegrid(lat_res, lon_res)
    x, y = m(lons, lats)


    cs = m.contourf(x, y, costs)
    plt.show()


def get_airports(dim):

    #TODO dummy data for now
    airports = []

    airports.append(Point(4.7,93.6))
    airports.append(Point(15.6,112.3))
    airports.append(Point(8.8, 111.2))

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
def calc_init_values():
    
    print("=== Calculating values")


# Calculate probabilities of finding the crash with a search plane
# Returns a lat x lon array with the probabilities of finding the
# crash in that area
def search_plane_probabilities(dim, crash_probabilities, depth_data):

    print("=== Calculating search plane probabilities")

    p_lat = np.linspace(dim.lat_min, dim.lat_max, dim.lat_res)
    p_lon = np.linspace(dim.lon_min, dim.lon_max, dim.lon_res)

    search_probabilities = np.ones((dim.lat_res, dim.lon_res)) * float('inf')

    for i in range(dim.lat_res):
        for j in range(dim.lon_res):
        	Pr_locating_given_crash_and_intact_at_surface = (1-alpha)*search_plane_visibility
        	Pr_locating_given_crash_and_intact_at_depth = alpha*(1 / (1 + search_plane_depth*pow(depth_data[i,j],3)))
        	Pr_locating_given_crash_and_intact = Pr_locating_given_crash_and_intact_at_depth + \
        										 Pr_locating_given_crash_and_intact_at_surface
        	Pr_locating_given_crash_and_destructive = search_plane_visibility
            
            search_probabilities[i,j] = (Pr_locating_given_crash_and_intact * Pr_intact) + \
            							(Pr_locating_given_crash_and_destructive * (1-Pr_intact))


    print("=== Done calculating search plane probabilities")

    return search_probabilities

if __name__ == "__main__":
    main()
