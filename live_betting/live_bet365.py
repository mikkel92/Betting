
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
import pyautogui
from datetime import datetime


def get_live_urls():

	page_url_mobile = "https://mobile.bet365.dk/#type=InPlay;"
	page_url = "https://www.bet365.dk/#/IP/"

	browser = webdriver.Chrome()  # choose web browser
	browser.get(page_url) # get the url for the corrosponding league
	browser.get(page_url)

	# Get browser to front of your screen
	pyautogui.moveTo(100, 200) # make the window come to front of screen for mouseover
	pyautogui.dragTo(100, 200, button='left')
	time.sleep(2)

	# Click on the tab to web scrape
	se_begivenhed_button = browser.find_elements_by_class_name("ip-ControlBar_BBarItem")
	se_begivenhed_button[1].click()
	time.sleep(2)
	
	# Click on every live event in the live betting tab
	event_buttons = browser.find_elements_by_class_name("ipn-FixtureButton")
	for button in event_buttons:
		
		start_time = datetime.now()
		button.click()
		event_data = browser.find_elements_by_class_name("ipe-EventViewDetail_MarketGrid")
		print event_data[0]
		match_data = event_data[0].text
		print match_data
		time.sleep(np.random.rand()*0.5 +.2)
		break
	
	source = browser.page_source
	#print source
	browser.close()
	"""

	html = browser.execute_script("return document.documentElement.innerHTML;")

	# This will get the html after on-load javascript
	#html = driver.execute_script("return window.performance.getEntries();")
	#print html

	"""
get_live_urls()


