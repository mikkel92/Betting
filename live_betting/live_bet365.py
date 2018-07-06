
import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import os, re
import selenium
from selenium import webdriver
from selenium.common.exceptions import NoSuchAttributeException
from selenium.webdriver.common.keys import Keys
import urllib, json
import mechanize
import pyautogui
from bs4 import BeautifulSoup
import spynner



def get_live_urls():

	page_url_mobile = "https://mobile.bet365.dk/#type=InPlay;"
	page_url = "https://www.bet365.dk/#/IP/"
	"""
	driver = webdriver.PhantomJS()
	driver.get(page_url)
	"""

	browser = webdriver.Chrome()  # choose web browser
	browser.get(page_url) # get the url for the corrosponding league
	browser.get(page_url)
	pyautogui.moveTo(100, 200) # make the window come to front of screen for mouseover
	pyautogui.dragTo(100, 200, button='left')
	time.sleep(2)
	 
	
	pyautogui.moveTo(120, 240) # make the curser move on homepage for javescript readout
	pyautogui.moveTo(120, 270, 0.2)
	pyautogui.dragTo(120, 270, button='left')
	pyautogui.moveTo(100, 500, 0.2)
	pyautogui.dragTo(100, 500, button='left')
	time.sleep(1)
	a = browser.find_elements_by_class_name("ipn-FixtureButton")

	#print a 
	for button in a:
		button.click()
		#time.sleep(np.random.rand()*0.5 +.2)

	#source = browser.page_source
	#print source
	browser.close()
	"""
	source = browser.page_source
	print source 
	html = browser.execute_script("return document.documentElement.innerHTML;")
	#browser.find_element_by_id("col-content") # find elements in javescript table
	#a = browser.find_element_by_class_name("gl-Participant_Name")
	print html
	odds_soup = BeautifulSoup(browser.page_source, 'lxml') # read it with beautiful soup
	odds_table = odds_soup.find_all('td',{'class': 'odds-nowrp'})
	
	browser.close()
	"""
	"""
	#print (json_file.split('englishName":"Football"')[-1][-2:])
	#print json_file

	#time.sleep(5) # sleep to make sure the page is loaded before executing java
	
	# This will get the html after on-load javascript
	#html = driver.execute_script("return window.performance.getEntries();")
	#print html
	

	#print live_ids
	"""
get_live_urls()


