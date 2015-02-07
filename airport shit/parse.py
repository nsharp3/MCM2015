import re

with open('busiestCities.txt') as f:
	for line in f:
		l = re.sub(r'\(\w*\)', '', line)
		print(l[:-1])

