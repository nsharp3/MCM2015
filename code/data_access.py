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

# External libraries
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

import scipy.stats

from shapely.geometry import LineString
from shapely.geometry import Point


def get_airports(dim):

    #TODO dummy data for now
    airports = []
    
    airports.append(Point_l(14.7,100.6))
    airports.append(Point_l(-3.6,105.3))
    airports.append(Point_l(2.8, 115.2))
    airports.append(Point_l(13.8, 93.2))

    return airports
