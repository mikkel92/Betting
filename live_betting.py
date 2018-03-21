
import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import json
from selenium import webdriver
from datetime import datetime
import os

def get_live_urls():

	page_url = "https://www.sofascore.com/football/livescore"

	driver = webdriver.PhantomJS()
	driver.get(page_url)

	# This will get the initial html - before javascript
	driver.page_source
	time.sleep(5) # sleep to make sure the page is loaded before executing java

	# This will get the html after on-load javascript
	html = driver.execute_script("return document.documentElement.innerHTML;")

	code_string = unicodedata.normalize('NFKD', html).encode('ascii','ignore')
	has_statistics = code_string.split('tab-event-widget-statistics')[0][-100:]
	live_urls = []

	# get the urls for the live matches that you want. Below are some selection criteria
	for i_c in code_string.split('js-event-link js-event'):

		match_time = i_c.split('event-live" title="')[-1][0:2]

		if match_time == 'HL': # get stats for a match where it is currently half time
			live_urls.append("https://www.sofascore.com" + str(i_c[0:300].split('href="')[-1].split('" data')[0][3:]))
			
		try:
			match_time = float(match_time)

			if float(match_time) < 30: continue # dont get data for matches where less than x mins have been played
			else: # get stats for rest
				live_urls.append("https://www.sofascore.com" + str(i_c[0:300].split('href="')[-1].split('" data')[0][3:]))
		
		except: continue
		
	return live_urls

def rearange_livedata(data_frame):
	
	data_frame = pd.DataFrame(data_frame)
	data_frame = data_frame[data_frame[0] != 'provider']
	data_frame = data_frame.sort_values(by=[0])
	data_frame.columns = ['variable','value']
	data_frame = data_frame.set_index('variable')

	for name in data_frame.index:
		data_frame = data_frame.rename({name : name[4:]})

	away_data = data_frame[0:len(data_frame)/2]
	home_data = data_frame[len(data_frame)/2:]

	return home_data, away_data

	
def estimate_betting_value(live_urls,debug=False):	

	
	print "\n \n \n"
	
	current_month = str(datetime.now().month)
	current_year = str(datetime.now().year)

	directory = "/Users/mjensen/Desktop/Universitet/Adaboost/live_data/" + current_year + "/" + current_month

	if not os.path.exists(directory):
   		os.makedirs(directory)
	
	no_stats_matches = 0 # counter for matches without statistics
	driver = webdriver.PhantomJS()

	for i_match in live_urls:
		current_datetime = str(datetime.now())[11:19]
		print_data = False
		second_half = False

		if no_stats_matches > 4:
			print('5 matches in a row without statistics. STOPPING SCRIPT')
			break

		# ----------------------------------------------------------- #
		# Try to load the data
		# ----------------------------------------------------------- #

		try:
			if debug: print "getting live page code string"

			driver.get(i_match)
			driver.page_source
			
			time.sleep(5) # sleep to make sure the page is loaded before executing java
			# This will get the html after on-load javascript
			live_html = driver.execute_script("return document.documentElement.innerHTML;")
			live_code_string = unicodedata.normalize('NFKD', live_html).encode('ascii','ignore')
			
			match_time = live_code_string.split('header-timer-container live ">')[-1][0:5]
			 
			# Get current score from source code 
			#TODO: doesn't work every time, fix
			try:
				current_score_home = int(live_code_string.split('h1 event-live')[-1][81])
				current_score_away = int(live_code_string.split('h1 event-live')[-1][195])
				
				
			except:
				print "couldn't load current score"
			
			if debug: print "getting link to jason file"
			json_link = ("https://www.sofascore.com" + str(live_code_string.split('js-event-details-async-content" data-src="')[1].split('"')[0]))
			# NEEDS FIX, not a general link (Or maybe it is) #

			# Request with fake header, otherwise you will get an 403 HTTP error
			r = requests.get(json_link, headers={'User-Agent': 'Mozilla/5.0'})
			odds = []
			json_file = r.json()
			
			
			if debug: print "getting data for odds"
			data = [i.split('[')[0] for i in json_file['statistics'].keys()]
			for i in json_file['odds']:
				if i['name'] == 'Next goal':
					
					for j in i['live']:
						for k in j['odds']:
							#print(k['decimalValue']
							odds.append(k['decimalValue'])
			
			if len(odds) == 0:
				print('could not get odds from this match')
				continue
			
		except Exception, e:

			no_stats_matches += 1
			continue

		file_name = directory + "/" + str(datetime.now().day) + "_" + current_datetime + "_" + i_match[26:-8] + ".txt"

		with open(file_name, 'w') as outfile:
			json.dump(json_file, outfile)
		 
		

		# ----------------------------------------------------------- #
		# Try to load the data
		# ----------------------------------------------------------- #


		no_stats_matches = 0

		data_frame = []
		data_first_half = []
		data_second_half = []
		for i_k in range(0,len(data)):
			
			if type(json_file['statistics'][data[i_k]]) == int: 

				data_frame.append([data[i_k],json_file['statistics'][data[i_k]]])

			elif data[i_k] == 'period1':
				data_H1 = json_file['statistics']['period1'].keys()
				for i_d in range(0,len(data_H1)):
					data_first_half.append([data_H1[i_d], json_file['statistics']['period1'][data_H1[i_d]]]) 

			elif data[i_k] == 'period2':
				data_H2 = json_file['statistics']['period2'].keys()
				for i_d in range(0,len(data_H2)):
					data_second_half.append([data_H2[i_d], json_file['statistics']['period2'][data_H2[i_d]]]) 

			else: continue

		home_data, away_data = rearange_livedata(data_frame)
		
		try:
			if float(match_time[0:2]) > 55.:
				second_half = True
				home_data_H1, away_data_H1 = rearange_livedata(data_first_half)
				home_data_H2, away_data_H2 = rearange_livedata(data_second_half)
		except:
			try:
				if match_time == 'Pause' and abs(current_score_home - current_score_away) > 2:
					print "One team is far ahead, loosing team might score after HT"
			except:
				a=1
		# set up selection rules for decent bets

		variables_to_use = ('HitWoodwork','ShotsOnGoal', 'AttInBoxBlocked', 'BlockedScoringAttempt',
							'ShotsOffGoal', 'TotalShotsOnGoal', 'TotalShotsInsideBox',
							'TotalShotsOutsideBox', 'Offsides', 'AccuratePassesPercent',
							'DuelWonPercent', 'AerialWonPercent', 'CornerKicks')

		variable_weight = (20. ,20., 15., 15.,
						   14., 0., 10.,
						   7., 4., 0.1,
						   0.1, 0.1, 2.)

		home_score = 0
		away_score = 0
		if second_half:
			home_score_H1 = 0
			away_score_H1 = 0
			home_score_H2 = 0
			away_score_H2 = 0

		variable_warning_str = 'WARNING, no data for variables: \n'

		exceptions = 0
		for i_v in range(0,len(variables_to_use)):

			try:
				home_score += home_data['value'][variables_to_use[i_v]] * variable_weight[i_v]
				away_score += away_data['value'][variables_to_use[i_v]] * variable_weight[i_v]

				if second_half:
					home_score_H1 += home_data_H1['value'][variables_to_use[i_v]] * variable_weight[i_v]
					away_score_H1 += away_data_H1['value'][variables_to_use[i_v]] * variable_weight[i_v]
					home_score_H2 += home_data_H2['value'][variables_to_use[i_v]] * variable_weight[i_v]
					away_score_H2 += away_data_H2['value'][variables_to_use[i_v]] * variable_weight[i_v]

			except:
				variable_warning_str += variables_to_use[i_v] + ' , '
				exceptions += 1
				continue

		if exceptions > 4:
			continue		

		if second_half:
			try:
				score_H1 = (home_score_H1 / (home_score_H1 + away_score_H1))
				score_H2 = (home_score_H2 / (home_score_H2 + away_score_H2))
				compare_half =  score_H1 - score_H2
				print score_H1, score_H2, compare_half
				if compare_half > 0:
					print "away team is doing %1.2f percent better than in first half" %(abs(compare_half)) 
				else: 	
					print "home team is doing %1.2f percent better than in first half" %(abs(compare_half))

				if abs(compare_half) > 0.15:
					print i_match[26:-8]
			except: 
				a = 1

		combined_score = home_score + away_score
		asian_correction = 0.93
		try:
			odds_home = float(odds[0]) * asian_correction
			odds_away = float(odds[2]) * asian_correction
		except:
			print('could not get odds from this match')
			continue
		# estimate asian from odds on next goal
		combined_odds = odds_home + odds_away
		#print(home_score, combined_odds / odds_away)
		#print(away_score, combined_odds / odds_home)

		significance = 1.3 # decide how much better the team has to perform, relative to odds

		if (combined_odds / odds_away) * home_score / combined_score > significance :
			print_data = True
			print("interesting odds on home team next goal")
			print("approx odds: %1.2f,  home team score: %1.2f pecent,  bet value: %1.2f " %((combined_odds / odds_away),(home_score/combined_score) * 100., (combined_odds / odds_away) * (home_score/combined_score) ))

			try:
				if float(home_data['value']['RedCards']) > 0:
					print('WARNING: home team has %i red cards' %int(home_data['value']['RedCards']))
			except:
				a = 1

		if (combined_odds / odds_home) * away_score / combined_score > significance : 
			print_data = True
			print("interesting odds on away team next goal" )
			print("approx odds: %1.2f,  away team score: %1.2f percent,  bet value: %1.2f " %((combined_odds / odds_home),(away_score/combined_score) * 100., (combined_odds / odds_home) * (away_score/combined_score) ))
			
			try :
				if float(away_data['value']['RedCards']) > 0:
					print('WARNING: away team has %i red cards' %int(away_data['value']['RedCards']))
			except:
				a = 1

		if print_data:
			
			print('\ngetting match with link : %s' %i_match)
			print match_time	
			print(variable_warning_str)
			try: 
				print('current score: %i - %i' %(current_score_home, current_score_away))
			except:
				a = 1
			print('---------------------')

		"""
		# plotting
		ticks = [unicodedata.normalize('NFKD', i).encode('ascii','ignore') for i in data_frame['value']]
		plt.plot(data_frame['value'][0:len(data_frame)/2],'o',label='away')
		plt.plot(data_frame['value'][len(data_frame)/2:],'+',label='home')
		plt.legend()
		plt.xticks(np.linspace(0,len(ticks)/2,len(ticks)/2),ticks[0:len(ticks)/2],rotation=90)
		plt.show(block=False)
		"""

live_urls = get_live_urls()
estimate_betting_value(live_urls)


raw_input( ' ... ' )

####
# TODO:
# get bet365 stats and odds
# is NextScoreHome/NextScoreAway == AsianHome/AsianAway? 
# compare to previous half.
# get asain odds from pages.






















