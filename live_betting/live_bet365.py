
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchAttributeException
from selenium.webdriver.common.keys import Keys
import urllib, json
from datetime import datetime

def get_match_data(button,browser):

	# click mathc button and wait a bit for it to load
	start_time = datetime.now()
	button.click()
	time.sleep(0.2)
	inner_html = str(button.get_attribute('innerHTML').encode(encoding='UTF-8',errors='strict'))

	# change to stats tab i live stream is an option for the match
	if "ipn-ScoreDisplayStandard_AVIcon" in inner_html:
		stats_tab = browser.find_element_by_class_name("lv-ButtonBar_MatchLive")
		stats_tab.click()
		time.sleep(0.2)

	# try to scrape the match data
	try:
		event_data = browser.find_elements_by_class_name("ml1-AllStats")
		event_odds = browser.find_elements_by_class_name("ipe-EventViewDetail_MarketGrid")
		match_data = ([event_data[0].text, event_odds[0].text])

	except: 
		match_data = "failed"

	if "ipn-ScoreDisplayStandard " not in inner_html:
		match_data = "not_soccer_match"
	
	time.sleep(np.random.rand()*0.5)
	return match_data



def scrape_betting():

	page_url_mobile = "https://mobile.bet365.dk/#type=InPlay;"
	page_url = "https://www.bet365.dk/#/IP/"

	browser = webdriver.Chrome()  # choose web browser
	browser.get(page_url) # get the url for the corrosponding league
	browser.get(page_url)

	time.sleep(3) #

	# Click on the tab to web scrape
	se_begivenhed_button = browser.find_elements_by_class_name("ip-ControlBar_BBarItem")
	se_begivenhed_button[1].click()
	time.sleep(3)

	
	
	# Click on every live event in the live betting tab
	event_buttons = browser.find_elements_by_class_name("ipn-FixtureButton")
	failed_loads = []
	page_fails = ([0,0])

	for counter, button in enumerate(event_buttons):
		print counter

		match_data = get_match_data(button,browser)
		
		if match_data == "not_soccer_match":
			print "Done scraping soccer matches"
			break
		elif match_data == "failed":
			match_data = get_match_data(button,browser)
			if match_data == "failed":
				failed_loads.append(button)
				print "failed to get page"
				continue
		else:
			continue
	print match_data	
		
	
	for button in failed_loads:

		match_data = get_match_data(button,browser)
		if match_data == "failed":
			page_fails[1] += 1
			print "failed to get page second time"
			

	print page_fails

	#print source
	browser.close()
	"""

	html = browser.execute_script("return document.documentElement.innerHTML;")

	# This will get the html after on-load javascript
	#html = driver.execute_script("return window.performance.getEntries();")
	#print html

	"""
scrape_betting()


