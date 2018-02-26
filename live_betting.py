
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
time.sleep(2) # sleep to make sure the page is loaded before executing java

# This will get the html after on-load javascript
html2 = driver.execute_script("return document.documentElement.innerHTML;")

code_string = unicodedata.normalize('NFKD', html2).encode('ascii','ignore')
live_urls = []

for i_c in code_string.split('js-event-link js-event'):
	live_urls.append(i_c[0:300].split('href="')[-1].split('" data')[0][3:])
	
driver.quit()

for url in live_urls:
	url_to_open = "https://www.sofascore.com" + str(url)

	driver.get(url_to_open)

"""
# Request with fake header, otherwise you will get an 403 HTTP error
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

odds = []
json_file = r.json()
for i in json_file['odds']:
	print '---------------------'
	
	for j in i['live']:
		for k in j['odds']:
			print k['decimalValue']
			odds.append(k['decimalValue'])

data = [i.split('[')[0] for i in json_file['statistics'].keys()]

data_frame = []
for i_k in range(0,len(data)):
	
	if type(json_file['statistics'][data[i_k]]) == int: 

		data_frame.append([data[i_k],json_file['statistics'][data[i_k]]])
	else: continue

data_frame = pd.DataFrame(data_frame)
data_frame.sort_values(by=[0])

data_frame.plot(x=data_frame[0])
#plt.xticks(data_frame[0])
plt.show(block=False)

"""
raw_input( ' ... ' )




