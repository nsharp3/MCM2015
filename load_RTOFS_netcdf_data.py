from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import netCDF4

import os.path

plt.figure()

mydate='20150208';
server_prefix = 'http://nomads.ncep.noaa.gov:9090/dods/'
dataset = 'rtofs/rtofs_global'+mydate
#datavar = 'rtofs_glo_3dz_forecast_daily_uvel'
datavar = 'rtofs_glo_3dz_forecast_daily_vvel'

url= server_prefix + dataset + '/' + datavar
cacheFilename = 'cache/dataset' + '-' + datavar + '-%s.dump'

print("URL is: " + url)

if(os.path.isfile(cacheFilename%('data'))):

    print("Reading data from cache")
    
    lat = np.load(cacheFilename%('lat'))
    lon = np.load(cacheFilename%('lon'))
    data = np.load(cacheFilename%('data'))

else:

    print("No cached data found")
    print("Accessing data from server...")
    cacheFile = open(cacheFilename, 'w')
    file = netCDF4.Dataset(url)
    print("\tlat...")
    lat  = file.variables['lat'][:]
    print("\tlong...")
    lon  = file.variables['lon'][:]
    print("\ttemp...")
    data = file.variables['v'][1,1,:,:]
    print("\tdone.")

    file.close()

    # Save so we can read from cache later
    print("Saving data to cache")
    lat.dump(cacheFilename%('lat'))
    lon.dump(cacheFilename%('lon'))
    data.dump(cacheFilename%('data'))


m=Basemap(projection='mill',lat_ts=10, \
  llcrnrlon=lon.min(),urcrnrlon=lon.max(), \
  llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
  resolution='c')


Lon, Lat = meshgrid(lon,lat)
x, y = m(Lon,Lat)

print(data.shape)
print(data[1000:1050,2000:2050])

cs = m.pcolormesh(x,y,data,shading='flat', \
  cmap=plt.cm.jet)

colorbar(cs)
plt.title('Example 1: Global RTOFS SST from NOMADS')
plt.show()
