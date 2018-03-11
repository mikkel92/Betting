
import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from selenium import webdriver

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
	
def estimate_betting_value(live_urls):	

	
		
	no_stats_matches = 0 # counter for matches without statistics
	driver = webdriver.PhantomJS()

	for i_match in live_urls:

		if no_stats_matches > 2:
			print('3 matches in a row without statistics. STOPPING SCRIPT')
			break

		print('---------------------')
		print('\ngetting match with link : %s' %i_match)

		try:
			driver.get(i_match)
			driver.page_source
			time.sleep(5) # sleep to make sure the page is loaded before executing java
			# This will get the html after on-load javascript
			live_html = driver.execute_script("return document.documentElement.innerHTML;")
			live_code_string = unicodedata.normalize('NFKD', live_html).encode('ascii','ignore')
			
			# Get current score from source code
			current_score_home = int(live_code_string.split('h1 event-live')[-1][81])
			current_score_away = int(live_code_string.split('h1 event-live')[-1][195])
			
			
			# get 
			json_link = ("https://www.sofascore.com" + str(live_code_string.split('js-event-details-async-content" data-src="')[1].split('"')[0]))
			# NEEDS FIX, not a general link (Or maybe it is) #

			# Request with fake header, otherwise you will get an 403 HTTP error
			r = requests.get(json_link, headers={'User-Agent': 'Mozilla/5.0'})
			odds = []
			json_file = r.json()
			
			
			data = [i.split('[')[0] for i in json_file['statistics'].keys()]
			for i in json_file['odds']:
				if i['name'] == 'Next goal':
					
					for j in i['live']:
						for k in j['odds']:
							#print(k['decimalValue']
							odds.append(k['decimalValue'])
			
			if len(odds) == 0:
				print('could not get data from this match')
				continue
			
		except:

			no_stats_matches += 1
			print('could not get data from this match')
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

		for name in data_frame.index:
			data_frame = data_frame.rename({name : name[4:]})

		away_data = data_frame[0:len(data_frame)/2]
		home_data = data_frame[len(data_frame)/2:]


		# set up selection rules for decent bets

		variables_to_use = ('HitWoodwork','ShotsOnGoal', 'AttInBoxBlocked', 'BlockedScoringAttempt',
							'ShotsOffGoal', 'TotalShotsOnGoal', 'TotalShotsInsideBox',
							'TotalShotsOutsideBox', 'Offsides', 'AccuratePassesPercent',
							'DuelWonPercent', 'AerialWonPercent', 'CornerKicks')

		variable_weight = (20. ,20., 15., 15.,
						   14., 10., 10.,
						   7., 4., 0.1,
						   0.1, 0.1, 2.)

		home_score = 0
		away_score = 0

		variable_warning_str = 'WARNING, no data for variables: \n'

		for i_v in range(0,len(variables_to_use)):

			try:
				home_score += home_data['value'][variables_to_use[i_v]] * variable_weight[i_v]
				away_score += away_data['value'][variables_to_use[i_v]] * variable_weight[i_v]
			
			except:
				variable_warning_str += variables_to_use[i_v] + ' , '
				continue



		combined_score = home_score + away_score
		asian_correction = 0.93

		odds_home = float(odds[0]) * asian_correction
		odds_away = float(odds[2]) * asian_correction
		# estimate asian from odds on next goal
		combined_odds = odds_home + odds_away
		#print(home_score, combined_odds / odds_away)
		#print(away_score, combined_odds / odds_home)

		significance = 1.0 # decide how much better the team has to perform, relative to odds

		if (combined_odds / odds_away) * home_score / combined_score > significance :
			print('current score: %i - %i' %(current_score_home, current_score_away))
			print("interesting odds on home team next goal")
			print("approx odds: %1.2f,  home team score: %1.2f pecent,  bet value: %1.2f " %((combined_odds / odds_away),(home_score/combined_score) * 100., (combined_odds / odds_away) * (home_score/combined_score) ))
			print(variable_warning_str)

			try:
				if float(home_data['value']['RedCards']) > 0:
					print('WARNING: home team has %i red cards' %int(home_data['value']['RedCards']))
			except:
				a = 1

		if (combined_odds / odds_home) * away_score / combined_score > significance : 
			print('current score: %i - %i' %(current_score_home, current_score_away))	
			print("interesting odds on away team next goal" )
			print("approx odds: %1.2f,  away team score: %1.2f percent,  bet value: %1.2f " %((combined_odds / odds_home),(away_score/combined_score) * 100., (combined_odds / odds_home) * (away_score/combined_score) ))
			print(variable_warning_str)
			
			try :
				if float(away_data['value']['RedCards']) > 0:
					print('WARNING: away team has %i red cards' %int(away_data['value']['RedCards']))
			except:
				a = 1
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
# is NextScoreHome/NextScoreAway == AsianHome/AsianAway? 
# compare to previous half.
# get asain odds from pages.






















