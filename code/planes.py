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
    
    
    # create Basemap instance.
    m = Basemap(projection='mill',\
                llcrnrlat=lat_min,urcrnrlat=lat_max,\
                llcrnrlon=lon_min,urcrnrlon=lon_max,\
                resolution='l')
   
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
    cs = m.contourf(x, y, costs)

    
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
def calc_init_values():
    
    print("=== Calculating values")



if __name__ == "__main__":
    main()
