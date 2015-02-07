from geopy.geocoders import Nominatim
geolocator = Nominatim()
with open('cities.txt') as f:
	for line in f:
		try:
			location = geolocator.geocode(line)
			print((location.latitude, location.longitude))
		except:
			pass
