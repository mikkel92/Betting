
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
html1 = driver.page_source
time.sleep(5) # sleep to make sure the page is loaded before executing java

# This will get the html after on-load javascript
html2 = driver.execute_script("return document.documentElement.innerHTML;")

code_string = unicodedata.normalize('NFKD', html2).encode('ascii','ignore')
live_urls = []

for i_c in code_string.split('js-event-link js-event'):
	live_urls.append("https://www.sofascore.com" + str(i_c[0:300].split('href="')[-1].split('" data')[0][3:]))



for i_match in live_urls[1:]:
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
			print '---------------------'
			
			for j in i['live']:
				for k in j['odds']:
					print k['decimalValue']
					odds.append(k['decimalValue'])

		
	except:
		print 'could not get data from this match'
		continue
	
	data_frame = []
	for i_k in range(0,len(data)):
		
		if type(json_file['statistics'][data[i_k]]) == int: 

			data_frame.append([data[i_k],json_file['statistics'][data[i_k]]])
		else: continue

	data_frame = pd.DataFrame(data_frame)
	data_frame = data_frame[data_frame[0] != 'provider']
	data_frame = data_frame.sort_values(by=[0])
	print data_frame # 0 is names on variables, 1 is variable value

	ticks = [unicodedata.normalize('NFKD', i).encode('ascii','ignore') for i in data_frame[0]]
	plt.plot(data_frame[1][0:len(data_frame)/2],'o',label='away')
	plt.plot(data_frame[1][len(data_frame)/2:],'+',label='home')
	plt.legend()
	plt.xticks(np.linspace(0,len(ticks)/2,len(ticks)/2),ticks[0:len(ticks)/2],rotation=90)
	plt.show(block=False)

raw_input( ' ... ' )




