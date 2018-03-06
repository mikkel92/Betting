
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from selenium import webdriver

page_url = "https://www.sofascore.com/football/livescore"

driver = webdriver.PhantomJS()
driver.get(page_url)

# This will get the initial html - before javascript
driver.page_source
time.sleep(5) # sleep to make sure the page is loaded before executing java

# This will get the html after on-load javascript
html = driver.execute_script("return document.documentElement.innerHTML;")

code_string = unicodedata.normalize('NFKD', html).encode('ascii','ignore')
live_urls = []

for i_c in code_string.split('js-event-link js-event'):
	live_urls.append("https://www.sofascore.com" + str(i_c[0:300].split('href="')[-1].split('" data')[0][3:]))

no_stats_matches = 0 # counter for matches without statistics

for i_match in live_urls[1:]:

	if no_stats_matches > 2:
		print '3 matches in a row without statistics. STOPPING SCRIPT'
		break

	print 'getting match with link : %s' %i_match

	try:
		driver.get(i_match)
		driver.page_source
		time.sleep(5) # sleep to make sure the page is loaded before executing java
		# This will get the html after on-load javascript
		live_html = driver.execute_script("return document.documentElement.innerHTML;")
		live_code_string = unicodedata.normalize('NFKD', live_html).encode('ascii','ignore')
		#print live_code_string
		json_link = ("https://www.sofascore.com" + str(live_code_string.split('js-event-details-async-content" data-src="')[1].split('"')[0]))
		# NEEDS FIX, not a general link (Or maybe it is) #

		# Request with fake header, otherwise you will get an 403 HTTP error
		r = requests.get(json_link, headers={'User-Agent': 'Mozilla/5.0'})

		odds = []
		json_file = r.json()
		
		data = [i.split('[')[0] for i in json_file['statistics'].keys()]
		for i in json_file['odds']:
			if i['name'] == 'Next goal':
				print '---------------------'
				for j in i['live']:
					for k in j['odds']:
						print k['decimalValue']
						odds.append(k['decimalValue'])

		
	except:

		no_stats_matches += 1
		print 'could not get data from this match'
		continue
	
	no_stats_matches = 0

	data_frame = []
	for i_k in range(0,len(data)):
		
		if type(json_file['statistics'][data[i_k]]) == int: 

			data_frame.append([data[i_k],json_file['statistics'][data[i_k]]])
		else: continue

	# rearange the data_frame a bit
	data_frame = pd.DataFrame(data_frame)
	data_frame = data_frame[data_frame[0] != 'provider']
	data_frame = data_frame.sort_values(by=[0])
	data_frame.columns = ['variable','value']
	data_frame = data_frame.set_index('variable')
	#print data_frame

	significance = 1.5 # how well does a team need to perform before the script should tell us?

	TotalShotsOnGoal = float(data_frame["value"]["homeTotalShotsOnGoal"] + data_frame["value"]["awayTotalShotsOnGoal"])
	print TotalShotsOnGoal, data_frame["value"]["homeTotalShotsOnGoal"], data_frame["value"]["awayTotalShotsOnGoal"]
	
	if float(odds[0]) * data_frame["value"]["homeTotalShotsOnGoal"] / TotalShotsOnGoal > significance :
		print "interesting odds on home team next goal"

	if float(odds[2]) * data_frame["value"]["awayTotalShotsOnGoal"] / TotalShotsOnGoal > significance : 	
		print "interesting odds on away team next goal" 

	"""
	# plotting
	ticks = [unicodedata.normalize('NFKD', i).encode('ascii','ignore') for i in data_frame['value']]
	plt.plot(data_frame['value'][0:len(data_frame)/2],'o',label='away')
	plt.plot(data_frame['value'][len(data_frame)/2:],'+',label='home')
	plt.legend()
	plt.xticks(np.linspace(0,len(ticks)/2,len(ticks)/2),ticks[0:len(ticks)/2],rotation=90)
	plt.show(block=False)
	"""

raw_input( ' ... ' )

####
# TODO:
# Compare with next goal odds.
# Use Ball possession, red cards, shots on goal, shots of goal as stats
# compare to previous half.
# get asain odds from pages.






















