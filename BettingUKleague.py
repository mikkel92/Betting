from bs4 import BeautifulSoup
#from urllib2 import urlopen
import pandas as pd
import numpy as np
import scipy as sp
import requests
import urllib
import re
import os
import glob
#from xgboost import XGBClassifier 
from lxml import html
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
#from selenium import webdriver
#from selenium.webdriver.common.action_chains import ActionChains
from statsmodels.discrete.discrete_model import Logit


class Estimate_Outcome(object):

    """
    Calculate the probabilities of soccer game outcomes

    Parameters
    ----------

    country : Name of country to estimate matches for

    compare : 0 or 1
        Compare with the actual outcome?
        If compare = 0 the matches for this week will be estimated
        If compare = 1 the matches in round number **round_to_estimate** will be estimated and compared
        with the true outcome of the matches afterwards

    safety : A number between 0.0 and 1.0
        How certain is the outcome?
        Sets how certain you want to be on the outcome of a match. Only matches with one of the three
        possible outcomes getting a score above the set safety will count.

    min_odds : A number > 0
        What is the minimum odds of an outcome that you are willing to bet on?

    round_to_estimate : Real positive integer between 0 and rounds played this season
        Which round do you want to estimate the outcomes of? 
        The widetable is only made from games played before the **round_to_estimate**

    n_rounds_training : Real positive integer between 0 and **round_to_estimate**
        Sets how many rounds back the training data will go
        The widetable is only made from games played after **round_to_estimate** - **n_rounds_training**

    """

    def __init__(self, country, data_year ,compare=1, min_odds=2.5, round_to_estimate='None', n_rounds_training='all',all_vs_all=True):
        self.country = country 
        self.round_to_estimate = round_to_estimate
        self.n_rounds_training = n_rounds_training
        self.compare = compare
        self.min_odds = min_odds
        self.all_vs_all = all_vs_all
        self.data_year = data_year
        #if self.round_to_estimate <= self.n_rounds_training and self.n_rounds_training != 'all':
        #    raise Exception( "Should not train on the round that is estimated")
    
    def load_results(self):
        
        data_path = '/home/daniel/Betting/Betting/data/premier_league/%s' %self.data_year
            
        league_data = pd.read_csv(data_path)  
        odds_columns = ["B365H","B365D","B365A"]

        tmp_odds_array = []
        for i_d in range(0,len(league_data[odds_columns])): 
            tmp_odds_array.append([league_data["B365H"][i_d],league_data["B365D"][i_d],league_data["B365A"][i_d]])

        all_results_odds = (tmp_odds_array)
        columns_to_use = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
                            "HTHG","HTAG","HTR","HS","AS","HST",
                            "AST","HF","AF","HC","AC","HY","AY","HR","AR",
                            ]
        all_results = league_data[columns_to_use]
        #print tmp_odds_array
        #print all_results[0]

        return all_results, all_results_odds


    def make_widetable(self, all_results):

        
        sort_teams = np.unique((all_results["HomeTeam"][0:50]))
        n_teams = len(sort_teams)
        
        if self.round_to_estimate == 'None':
            widetable_iterator_finnish = len(all_results)
        else:
            widetable_iterator_finnish = int((self.round_to_estimate - 1) * n_teams / 2) # needs improvement

        if self.n_rounds_training == 'all':
            widetable_iterator_start = 0
        else:
            widetable_iterator_start = int(widetable_iterator_finnish - (self.n_rounds_training * n_teams / 2))


        widetable = ( {
            "Team":[sort_teams],
            "P":np.zeros(len(sort_teams)),
            "W":np.zeros(len(sort_teams)),
            "D":np.zeros(len(sort_teams)),
            "L":np.zeros(len(sort_teams)),
            "GF":np.zeros(len(sort_teams)),
            "GA":np.zeros(len(sort_teams)),
            "GD":np.zeros(len(sort_teams)),
            "Pts":np.zeros(len(sort_teams)),
            "Wh":np.zeros(len(sort_teams)),
            "Dh":np.zeros(len(sort_teams)),
            "Lh":np.zeros(len(sort_teams)),
            "GFh":np.zeros(len(sort_teams)),
            "GAh":np.zeros(len(sort_teams)),
            "Wa":np.zeros(len(sort_teams)),
            "Da":np.zeros(len(sort_teams)),
            "La":np.zeros(len(sort_teams)),
            "GFa":np.zeros(len(sort_teams)),
            "GAa":np.zeros(len(sort_teams)),
            "HS":np.zeros(len(sort_teams)),
            "AS":np.zeros(len(sort_teams)),
            "HST":np.zeros(len(sort_teams)),
            "AST":np.zeros(len(sort_teams)),
            "HF":np.zeros(len(sort_teams)),
            "AF":np.zeros(len(sort_teams)),
            "HC":np.zeros(len(sort_teams)),
            "AC":np.zeros(len(sort_teams)),
            "HY":np.zeros(len(sort_teams)),
            "AY":np.zeros(len(sort_teams)),
            "HR":np.zeros(len(sort_teams)),
            "AR":np.zeros(len(sort_teams)),




        })
        
        #print widetable_iterator_start, widetable_iterator_finnish
        #print all_results["HomeTeam"][widetable_iterator_start], all_results["AwayTeam"][widetable_iterator_start]
        #print all_results["HomeTeam"][widetable_iterator_finnish-1], all_results["AwayTeam"][widetable_iterator_finnish-1]
        for i_i in range(widetable_iterator_start, widetable_iterator_finnish):            
            
            weight = 1. # float(((i_i - widetable_iterator_start) * 2 / n_teams) + 1.)/ float((widetable_iterator_finnish - widetable_iterator_start) / n_teams * 2)
            
            for i_t in range(0, len(sort_teams)):
                match_GD = float(all_results["FTHG"][i_i]) - float(all_results["FTAG"][i_i])
                
                if sort_teams[i_t] == all_results["HomeTeam"][i_i]:
                    widetable["P"][i_t] += 1 * weight
                    widetable["GD"][i_t] += match_GD #* weight

                    if match_GD > 0:
                        widetable["Pts"][i_t] += 3 * weight
                        widetable["W"][i_t] += 1 * weight
                        widetable["Wh"][i_t] += 1 * weight
                    elif match_GD == 0:
                        widetable["Pts"][i_t] += 1 * weight
                        widetable["D"][i_t] += 1 * weight
                        widetable["Dh"][i_t] += 1 * weight
                    else: 
                        widetable["Pts"][i_t] += 0 * weight
                        widetable["L"][i_t] += 1 * weight
                        widetable["Lh"][i_t] += 1 * weight

                    widetable["HS"][i_t] += float(all_results["HS"][i_i]) * weight
                    widetable["HST"][i_t] += float(all_results["HST"][i_i]) * weight
                    widetable["HF"][i_t] += float(all_results["HF"][i_i]) * weight
                    widetable["HC"][i_t] += float(all_results["HC"][i_i]) * weight
                    widetable["HY"][i_t] += float(all_results["HY"][i_i]) * weight
                    widetable["HR"][i_t] += float(all_results["HR"][i_i]) * weight

                    widetable["GF"][i_t] += float(all_results["FTHG"][i_i]) * weight
                    widetable["GA"][i_t] += float(all_results["FTAG"][i_i]) * weight
                    widetable["GFh"][i_t] += float(all_results["FTHG"][i_i]) * weight
                    widetable["GAh"][i_t] += float(all_results["FTAG"][i_i]) * weight

                if sort_teams[i_t] == all_results["AwayTeam"][i_i]:
                    widetable["P"][i_t] += 1 * weight
                    widetable["GD"][i_t] -= match_GD #* weight

                    if match_GD > 0:
                        widetable["Pts"][i_t] += 0 * weight
                        widetable["L"][i_t] += 1 * weight
                        widetable["La"][i_t] += 1 * weight
                    elif match_GD == 0:
                        widetable["Pts"][i_t] += 1 * weight
                        widetable["D"][i_t] += 1 * weight
                        widetable["Da"][i_t] += 1 * weight
                    else: 
                        widetable["Pts"][i_t] += 3 * weight
                        widetable["W"][i_t] += 1 * weight
                        widetable["Wa"][i_t] += 1 * weight

                    widetable["AS"][i_t] += float(all_results["AS"][i_i]) * weight
                    widetable["AST"][i_t] += float(all_results["AST"][i_i]) * weight
                    widetable["AF"][i_t] += float(all_results["AF"][i_i]) * weight
                    widetable["AC"][i_t] += float(all_results["AC"][i_i]) * weight
                    widetable["AY"][i_t] += float(all_results["AY"][i_i]) * weight
                    widetable["AR"][i_t] += float(all_results["AR"][i_i]) * weight

                    widetable["GF"][i_t] += float(all_results["FTAG"][i_i]) * weight
                    widetable["GA"][i_t] += float(all_results["FTHG"][i_i]) * weight
                    widetable["GFa"][i_t] += float(all_results["FTAG"][i_i]) * weight
                    widetable["GAa"][i_t] += float(all_results["FTHG"][i_i]) * weight
        for i_v in widetable:
            if i_v != "Team":
                widetable[i_v] = widetable[i_v] / (max(widetable[i_v]) + 0.0000001)
        #print widetable[-1]
        
        return widetable

    def masseys(self,input_results,widetable):

        
        n_teams = len(np.unique((widetable["Team"][0])))
        n_round = self.round_to_estimate # figure which round it is from how many games each team has played  
        nr_of_matches = int(n_teams / 2) # number of matches per round in league    
        matches_so_far = n_round * nr_of_matches  # mathces played this season
        training_start = (n_round - self.n_rounds_training) * nr_of_matches
        n_train_matches = matches_so_far - training_start # nr of matches used for training

        data_masseys = {}
        for team in widetable["Team"][0]:
            data_masseys[team] = []

        match_outcomes = []
        for i_m in range(training_start - nr_of_matches, matches_so_far - nr_of_matches): # loop over matches played before the week we're trying to estimate the outcomes of
            match_outcome = float(input_results["FTHG"][i_m]) - float(input_results["FTAG"][i_m]) # calculated by how much which team won
            match_outcomes.append(match_outcome)

            for team in widetable["Team"][0]: 
                if input_results["HomeTeam"][i_m] == team: # append 1 if team was home
                    data_masseys[team].append(1)
                elif input_results["AwayTeam"][i_m] == team: # append -1 if team was away
                    data_masseys[team].append(-1)
                else:
                    data_masseys[team].append(0) # append 0 to all other teams
        
        data_masseys = pd.DataFrame(data_masseys)
        data_masseys['Intercept'] = 1.0 
        match_outcomes = pd.Series(match_outcomes)
        match_outcomes = (match_outcomes - min(match_outcomes)) / (abs(max(match_outcomes) - min(match_outcomes)))

        regression = Logit(match_outcomes,data_masseys).fit_regularized(method = 'l1', disp=False)
        params = regression.params
        del params['Intercept']
          
        #print np.exp(params)
        return np.exp(params)

    def make_match_features(self,widetable,masseys):

        
        
        home_data = np.array([widetable["Team"][0],masseys,widetable['Wa'],widetable['Da']
                             ,widetable['La'],widetable['GFa'],widetable['GAa']
                             ,widetable['P'],widetable['W'],widetable['L'],widetable['GF']
                             ,widetable['GA'],widetable['Pts'],widetable['Wh'],widetable['Dh']
                             ,widetable['Lh'],widetable['GFh'],widetable['GAh']
                             ,widetable['HS'],widetable['HF'],widetable['HY']
                             ,widetable['AS'],widetable['AF'],widetable['AY']
                             ,widetable['HST'],widetable['HC'],widetable['HR']
                             ,widetable['AST'],widetable['AC'],widetable['AR'],widetable['GD']])
        
        away_data = home_data

        """
        # data for how well the away team plays when it's away 
        home_data = np.array([widetable["Team"][0],widetable['Wa'],widetable['Da']
                             ,widetable['La'],widetable['GFa'],widetable['GAa']
                             ,widetable['P'],widetable['W'],widetable['L'],widetable['GF']
                             ,widetable['GA'],widetable['Pts'],widetable['Wh'],widetable['Dh']
                             ,widetable['Lh'],widetable['GFh'],widetable['GAh']
                             ,widetable['HS'],widetable['HF'],widetable['HY']
                             ,widetable['AS'],widetable['AF'],widetable['AY']
                             ,widetable['HST'],widetable['HC'],widetable['HR']
                             ,widetable['AST'],widetable['AC'],widetable['AR'],widetable['GD']])
        # data for how well the home team plays when it's home
        away_data = np.array([widetable["Team"][0],widetable['Wh'],widetable['Dh']
                             ,widetable['Lh'],widetable['GFh'],widetable['GAh']
                             ,widetable['P'],widetable['W'],widetable['L'],widetable['GF']
                             ,widetable['GA'],widetable['Pts'],widetable['Wa'],widetable['Da']
                             ,widetable['La'],widetable['GFa'],widetable['GAa']
                             ,widetable['HS'],widetable['HF'],widetable['HY']
                             ,widetable['AS'],widetable['AF'],widetable['AY']
                             ,widetable['HST'],widetable['HC'],widetable['HR']
                             ,widetable['AST'],widetable['AC'],widetable['AR'],widetable['GD']])
        """
     

        return home_data, away_data


    def make_training_sample(self,widetable,input_results):
        

        train_matches = [] # list for appending the data for the teams to the according games (training sample)
        score_last_week = [] # array for outcome of individual matches
        
        
        input_data = input_results[0]
        input_odds = input_results[1]
        
        n_teams = len(np.unique((widetable["Team"][0]))) # Get number of teams in league
        n_round = self.round_to_estimate # figure which round it is from how many games each team has played 
        nr_of_matches = int(n_teams / 2) # number of matches per round in league
        matches_so_far = n_round * nr_of_matches  # mathces played this season     
        training_start = (n_round - self.n_rounds_training) * nr_of_matches 
        #print matches_so_far

        # label the outcome of a match as 0: home team won, 1: draw, 2: away team won
        if self.all_vs_all:
            for i_m in range(training_start - nr_of_matches, matches_so_far - nr_of_matches): # loop over matches played before the week we're trying to estimate the outcomes of
                
                match_outcome = float(input_data["FTHG"][i_m]) - float(input_data["FTAG"][i_m]) # calculate goals(team_home) - goals(team_away)
                if match_outcome > 0: # if home team score more goals than away team
                    score_last_week.append(0)
                elif match_outcome == 0: # if the teams score the same number of goals
                    score_last_week.append(1)
                else: # if the team away score more goals than home team
                    score_last_week.append(2)

        # label matches where an outcome with odds > min_odds is the result of a match as won, and everything else as lost
        else:
            for i_m in range(training_start - nr_of_matches, matches_so_far - nr_of_matches): # loop over matches played before the week we're trying to estimate the outcomes of
                
                match_outcome = float(input_data["FTHG"][i_m]) - float(input_data["FTAG"][i_m]) # calculate goals(team_home) - goals(team_away)
                if match_outcome > 0: # if home team score more goals than away team
                    if input_odds[i_m][0] > self.min_odds:
                        score_last_week.append([1,0,0])
                    else: score_last_week.append([0,0,0])
                    continue
                elif match_outcome == 0: # if the teams score the same number of goals
                    if input_odds[i_m][1] > self.min_odds:
                        score_last_week.append([0,1,0])
                    else: score_last_week.append([0,0,0])
                    continue
                elif match_outcome < 0: # if the team away score more goals than home team
                    if input_odds[i_m][2] > self.min_odds:
                        score_last_week.append([0,0,1])
                    else: score_last_week.append([0,0,0])
                    continue
        
        # defining the teams playing eachother in the round    
        competitors = []
        for i_m in range(training_start - nr_of_matches, matches_so_far): # loop over mathces played
                competitors.append([str(input_data["HomeTeam"][i_m]),str(input_data["AwayTeam"][i_m])]) # append the names of the competing teams to list

        # get the features on which you want to train, specified in the make_match_features function
        masseys = Estimate_Outcome.masseys(self,input_data,widetable)
        t_h_data, t_a_data = Estimate_Outcome.make_match_features(self,widetable,masseys)

        delete_row = len(t_h_data) - 1

        # make the training sample:
        for i_m in range(0, self.n_rounds_training * nr_of_matches):
            team_home = competitors[i_m][0] # home team
            team_away = competitors[i_m][1] # away team
            
            # fix data arangements
            home_team_data = t_h_data.transpose()
            away_team_data = t_a_data.transpose()
            
            # find the data for only the two teams that played against each other
            for i_t in range(0, n_teams): # loop over teams
                if team_home == str(home_team_data[i_t][0].encode('utf-8')): # when the home teams data comes
                    t_h = (home_team_data[i_t]) # append to new variable
                if team_away == str(away_team_data[i_t][0].encode('utf-8')): # when the away teams data comes
                    t_a = (away_team_data[i_t]) # append to new variable

            t_h_a = np.concatenate((t_h, t_a), axis=0) # put the data together and delete team names
            t_h_a = np.delete(t_h_a, 0)
            t_h_a = np.delete(t_h_a, delete_row)
            t_h_a = np.concatenate((t_h_a,input_odds[i_m + training_start - nr_of_matches]), axis=0)
            
            train_matches.append(t_h_a) # append the data for the teams in the match played
    
    
        """
    
        reg = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=50, 
                                         algorithm='SAMME', random_state=0)    
        
        reg.fit(train_matches, score_last_week) # train on the training sample
        boosted_decisions = reg.decision_function(train_matches) # decide the outcome of new matches
        
        # printing precission of testing on training data
        correctly_estimated_train_matches = 0
        for i_t in range(0,len(train_matches)):
            if boosted_decisions[i_t][score_last_week[i_t]] == max(boosted_decisions[i_t]):
                correctly_estimated_train_matches += 1.
        print 'Train sample precision: %1.2f' % (correctly_estimated_train_matches / float(len(train_matches)))
        print len(train_matches)
        """
        return train_matches, score_last_week

    def rest(self,widetable,input_results,train_matches):    
        
        train_outcome_h = [] # List for home training data
        train_outcome_d = [] # List for draw training data
        train_outcome_a = [] # List for away training data
        
        for i_m in train_matches[1]:
            train_outcome_h.append(i_m[0])
            train_outcome_d.append(i_m[1])
            train_outcome_a.append(i_m[2])

        decide_matches = []
        decide_matches_outcome = []
        decide_matches_odds = []

        input_data = input_results[0]
        input_odds = input_results[1]


        
        # ------------------ dublicate, needs fix --------------------- #
        # defining the teams playing eachother in the round   

        n_teams = len(np.unique((widetable["Team"][0]))) # Get number of teams in league
        n_round = self.round_to_estimate # figure which round it is from how many games each team has played 
        nr_of_matches = int(n_teams / 2) # number of matches per round in league
        matches_so_far = n_round * nr_of_matches  # mathces played this season 

        competitors = []
        for i_m in range(0, matches_so_far): # loop over mathces played
            competitors.append([str(input_data["HomeTeam"][i_m]),str(input_data["AwayTeam"][i_m])]) # append the names of the competing teams to list

        masseys = Estimate_Outcome.masseys(self,input_data,widetable)
        t_h_data, t_a_data = Estimate_Outcome.make_match_features(self,widetable,masseys)

        delete_row = len(t_h_data) - 1

        # make the training sample:
        for i_m in range(0, matches_so_far - nr_of_matches):
            team_home = competitors[i_m][0] # home team
            team_away = competitors[i_m][1] # away team
            
            

            
            home_team_data = t_h_data.transpose()
            away_team_data = t_a_data.transpose()

        # make the sample for decision (same as above, but for new matches)
        # Get the outcome of matches which are played in the round you are trying to estimate 
        
            
            
        
        for i_m in range( matches_so_far - nr_of_matches, matches_so_far):

            match_outcome = float(input_data["FTHG"][i_m]) - float(input_data["FTAG"][i_m])
            if match_outcome > 0:
                decide_matches_outcome.append(0)
            elif match_outcome == 0:
                decide_matches_outcome.append(1)
            else:
                decide_matches_outcome.append(2)

            decide_matches_odds.append(input_odds[i_m])

            for i_t in range(0, n_teams):
                if competitors[i_m][0] == str(home_team_data[i_t][0].encode('utf-8')):
                    t_h = (home_team_data[i_t]) 
                if competitors[i_m][1] == str(away_team_data[i_t][0].encode('utf-8')):
                    t_a = (away_team_data[i_t])
            t_h_a = np.concatenate((t_h, t_a), axis=0)
            t_h_a = np.delete(t_h_a, 0)
            t_h_a = np.delete(t_h_a, delete_row)
            t_h_a = np.concatenate((t_h_a,input_odds[i_m]), axis=0)

            decide_matches.append(t_h_a)

        results_h = 0
        results_d = 0
        results_a = 0

        Rstate = 1.

        for i_r in range(0,int(Rstate)): # loop over different Rstates in AdaBoost to eliminate fluctuations introduced by a random number generator
            
            
            """   
            reg = XGBClassifier(max_depth=30, n_estimators=300, learning_rate=1) 

            reg.fit(np.array(train_matches[0]), train_outcome_h) # train on the training sample
            boosted_decisions = reg.predict(np.array(decide_matches)) # decide the outcome of new matches
            results_h += boosted_decisions # add the results together from the different Rstates

            reg.fit(np.array(train_matches[0]), train_outcome_d) # train on the training sample
            boosted_decisions = reg.predict(np.array(decide_matches)) # decide the outcome of new matches
            results_d += boosted_decisions # add the results together from the different Rstates

            reg.fit(np.array(train_matches[0]), train_outcome_a) # train on the training sample
            boosted_decisions = reg.predict(np.array(decide_matches)) # decide the outcome of new matches
            results_a += boosted_decisions # add the results together from the different Rstates
            """

            
            #adaboost

            reg = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=30, learning_rate=1.0, 
                                            algorithm='SAMME', random_state=i_r) 

            reg.fit(train_matches[0], train_outcome_h) # train on the training sample
            
            boosted_decisions = reg.decision_function(decide_matches) # decide the outcome of new matches
            results_h += boosted_decisions

            reg.fit(train_matches[0], train_outcome_d) # train on the training sample
            boosted_decisions = reg.decision_function(decide_matches) # decide the outcome of new matches
            results_d += boosted_decisions # add the results together from the different Rstates

            reg.fit(train_matches[0], train_outcome_a) # train on the training sample
            boosted_decisions = reg.decision_function(decide_matches) # decide the outcome of new matches
            results_a += boosted_decisions # add the results together from the different Rstates
            
        if self.compare == 0: # if you don't want to compare with existing results (estimating games that haven't been played)
            
            print(country) # print name of the country
            print('Number of teams in %s: %i' %(countrydata, n_teams))
            print('Round to estimate %i' %(n_round))
            
            estimated_results = [] # list for appending the estimated outcomes
            for i_m in range(matches_so_far - nr_of_matches, matches_so_far):
                # append team_home , BDT score, team_away
                estimated_results.append([competitors[i_m][0], results[i_m - matches_so_far + nr_of_matches] / Rstate, competitors[i_m][1]])

            # ------------------------------------- #
            # Get odds for liga from oddsportal
            # ------------------------------------- #

            # list for all the individual urls on oddsportal
            url_list = {
                        'china': 'http://www.oddsportal.com/soccer/china/super-league/',
                        'southkorea': 'http://www.oddsportal.com/soccer/south-korea/k-league-classic/',
                        'finland': 'http://www.oddsportal.com/soccer/finland/veikkausliiga/',
                        'thailand': 'http://www.oddsportal.com/soccer/thailand/thai-premier-league/',
                        'brazil': 'http://www.oddsportal.com/soccer/brazil/serie-a/',
                        'brazil2': 'http://www.oddsportal.com/soccer/brazil/serie-b/',
                        'brazil3': 'http://www.oddsportal.com/soccer/brazil/serie-c/',
                        'iceland': 'http://www.oddsportal.com/soccer/iceland/pepsideild/',
                        'lithuania': 'http://www.oddsportal.com/soccer/lithuania/a-lyga/',
                        'japan': 'http://www.oddsportal.com/soccer/japan/j-league/',
                        'japan2': '',
                        'iceland': 'http://www.oddsportal.com/soccer/iceland/pepsideild/'
                            }
            

            browser = webdriver.Chrome()  # choose web browser
            browser.get(url_list[country]) # get the url for the corrosponding league

            pyautogui.moveTo(100, 200) # make the window come to front of screen for mouseover
            pyautogui.dragTo(100, 200, button='left') 
            pyautogui.moveTo(685, 540) # make the curser move on homepage for javescript readout
            pyautogui.moveTo(685, 570, 0.2)


            browser.find_element_by_id("col-content") # find elements in javescript table
            browser.find_element_by_class_name("odds-nowrp")
            odds_soup = BeautifulSoup(browser.page_source, 'lxml') # read it with beautiful soup
            odds_table = odds_soup.find_all('td',{'class': 'odds-nowrp'})

            browser.close()

            odds_data = ([],[],[],[],[]) # list for appending odds data

            # append the data from oddsportal to the odds_data array
            for i_ot in range(0,len(odds_table)): # is this even used for anything?
                odds_data[0].append(odds_table[i_ot].find_all()[0].get_text())

            for i_ot in range(0,len(odds_table),3):
                odds_data[1].append(odds_data[0][i_ot]) # odds for home_team winning ( 1 )
                odds_data[2].append(odds_data[0][i_ot+1]) # odds for a draw ( X )
                odds_data[3].append(odds_data[0][i_ot+2]) # odds for away_team winning ( 2 )

            for i_od in range(0,len(odds_data[1])): # append the team names to array
                odds_data[4].append(odds_soup.find_all('td',{'class': 'name table-participant'})[i_od].get_text())
                
            # fix issue with team names being read out wierdly. NEEDS IMPROVEMENT. maybe some function that matches some of the team names but not all
            team_names = []
            for i_od in odds_data[4]:
                if len(i_od) > 100:
                    team_names.append(unicode((i_od.encode('ascii','replace')).split(';')[-1], 'utf-8'))
                    
                else: team_names.append(i_od)

            for i_t in range(len(team_names)): # order the teams nicer in array
                team_names[i_t]=re.split(' - ',team_names[i_t])

            odds_array = [] # array for odds on a match and the teams names
            names_stats = [] # team names on soccerstats.com
            names_stats_seperate = [] # team names on soccerstats.com in seperated columns
            names_odds = [] # team names on oddsportal.com

            for i_t in range(0, len(team_names)): # append things to lists
                odds_array.append([team_names[i_t][0].encode('utf-8'),odds_data[1][i_t],odds_data[2][i_t],odds_data[3][i_t],team_names[i_t][1].encode('utf-8')])
                names_odds.append([odds_array[i_t][0] + ' ' + odds_array[i_t][4]])
            
            for i_t in range(0,nr_of_matches):
                names_stats.append([estimated_results[i_t][0] + ' ' + estimated_results[i_t][2]])
                names_stats_seperate.append([estimated_results[i_t][0], estimated_results[i_t][2]])
            # If some of the name matches, use that same name in all arrays
            
            for i_s in range(0,len(names_stats)): # loop over names from soccerstats
                name_match = False # The name has not been matched in the list from oddsportal
                for i_o in range(0,len(names_odds)): # loop over names from oddsportal
                    if name_match:
                        break # break if the name has been matches in former iteration
                    
                    # check if some of the name from soccerstats matches with some of the name from oddsportal
                    teams = [x for x in names_stats[i_s][0].split() if any(y for y in names_odds[i_o][0].split() if x in y)]

                    # if there was a match, call the team the same in both arrays
                    # if all but one name or surname in the two teams match
                    if len(teams) > max((len(names_stats[i_s][0].split()),len(names_odds[i_o][0].split()))) - 2:
                        odds_array[i_o][0] = names_stats_seperate[i_s][0]
                        odds_array[i_o][4] = names_stats_seperate[i_s][1]
                        name_match = True
                        names_odds[i_o][0] = '0'
                        break
                        

            # put all the data into one dataframe
            
            betting_value = [] # array for how good a bet on an outcome would be
            
            print( ' | '.join((('Estimated outcomes').center(63),('Value score').center(20),('Odds on match').center(63))))
            
            for i_p in range(0,len(estimated_results)):
                printed = False
                for i_o in range(0,len(odds_array)):
                    if estimated_results[i_p][0] == odds_array[i_o][0] and estimated_results[i_p][2] == odds_array[i_o][4]:
                
                        # calculate 'chance of outcome' * odds
                        betting_value = ([round(float(estimated_results[i_p][1][0]) * float(odds_array[i_o][1]), 2), round(float(estimated_results[i_p][1][1]) * 
                                                float(odds_array[i_o][2]), 2), round(float(estimated_results[i_p][1][2]) * float(odds_array[i_o][3]), 2)]) 

                        print_data = (map(str, (estimated_results[i_p][0], estimated_results[i_p][1][0], estimated_results[i_p][1][1],
                                        estimated_results[i_p][1][2], estimated_results[i_p][2], betting_value, odds_array[i_o][0], odds_array[i_o][1],
                                        odds_array[i_o][2], odds_array[i_o][3], odds_array[i_o][4])))
                        for i_pd in range(0,len(print_data)):
                            if 0 < i_pd < 4 or 6 < i_pd < 10:
                                print_data[i_pd] = print_data[i_pd][:5].center(7)
                            elif i_pd == 5:
                                print_data[i_pd] = print_data[i_pd].center(20)
                            else: 
                                print_data[i_pd] = print_data[i_pd][:15].center(15)
                        print( ' | '.join(print_data))
                        Printed = True
                        break
                else: # if a name match on the two homepages, just print the estimated outcome
                    print_data = (map(str, (estimated_results[i_p][0], estimated_results[i_p][1][0], estimated_results[i_p][1][1],
                                        estimated_results[i_p][1][2], estimated_results[i_p][2])))
                    for i_pd in range(0,len(print_data)):
                        if 0 < i_pd < 4:
                            print_data[i_pd] = print_data[i_pd][:5].center(7)
                        else:
                            print_data[i_pd] = print_data[i_pd][:15].center(15)
                    print( ' | '.join(print_data))

        if self.compare == 1: # if you want to compare with the actualt relsults of the matches ( for checking how often AdaBoost estimates correctly)

            # Write to text file
            return_array = []
            for i_m in range(0,len(results_h)):
                return_array.append([self.country,results_h[i_m]/Rstate,results_d[i_m]/Rstate,results_a[i_m]/Rstate,decide_matches_outcome[i_m],decide_matches_odds[i_m]])

            return return_array
            """
            text_file = open(text_file_name, "w")
            for i_m in range(0,nr_of_matches):
                #to_write = str(self.country,results[i_m],score_current[matches_so_far - nr_of_matches + i_m],input_results["odds"][matches_so_far - nr_of_matches + i_m])

                text_file.write("%s%s%f%s" %(self.country,results[i_m],score_current[matches_so_far - nr_of_matches + i_m],input_results["odds"][matches_so_far - nr_of_matches + i_m]))
            
            text_file.close()

            if self.all_vs_all:

                n_bets = np.array([[0],[0],[0]]) # number of bets made
                avg_odds = np.array([[0.0],[0.0],[0.0]])
                n_wins = 0 # number of bets made on the correct outcome
                confusion_matrix = np.zeros([3,3]) # array for confusion matrix
                
                for i_m in range(0, nr_of_matches): # loop over matches
                    #if max(results[i_m]) > self.safety * Rstate:
                    # TODO clean this next bit
                    # loop over indicies in confusion matrix
                    if len(np.shape(results)) < 2:
                        break
                    results_looper = (0.0,1.0,2.0)
                    for i_c in range(0,3): # loop over actual outcome of matches
                        for i_r in range(0,3): # loop over possible outcome of matches
                            if input_results["odds"][matches_so_far - nr_of_matches + i_m][i_r] > self.min_odds:# * max(results[i_m]) / Rstate > 1.5: 
                                #print input_results["odds"][matches_so_far - nr_of_matches + i_m][i_r] * max(results[i_m]) / Rstate, input_results["odds"][matches_so_far - nr_of_matches + i_m][i_r]
                    # if the home team won and the home team got the highest score from AdaBoost and that score is above safety
                                if score_current[matches_so_far - nr_of_matches + i_m] == results_looper[i_c] and results[i_m][i_r] == max(results[i_m]) and results[i_m][i_r] > self.safety * Rstate:
                                    confusion_matrix[i_r][i_c] += 1.0
                                    n_bets[i_r] += 1.0
                        if score_current[matches_so_far - nr_of_matches + i_m] == results_looper[i_c] and results[i_m][i_c] == max(results[i_m]) and results[i_m][i_c] > self.safety * Rstate:
                            if input_results["odds"][matches_so_far - nr_of_matches + i_m][i_c] > self.min_odds:# * max(results[i_m]) / Rstate > 1.5: 
                                avg_odds[i_c] += input_results["odds"][matches_so_far - nr_of_matches + i_m][i_c]

                
                #print confusion_matrix
                #print n_bets


                estimated_results = [] # list for appending the estimated outcomes 
                for i_m in range(0, nr_of_matches): # loop over matches
                    estimated_results.append([competitors[ (n_round - 1) * nr_of_matches + i_m][0], results[i_m] / Rstate, competitors[ (n_round - 1) * nr_of_matches + i_m][1], score_current[matches_so_far - nr_of_matches + i_m]])
                #print estimated_results

                if max(n_bets) != 0: # if a match was qualifies to bet on
                    #Winpercent = n_wins/n_bets
                    
                    #print('fraction of games correctly estimated: %f. safety: %f' %((n_wins/n_bets),safety))
                    return confusion_matrix, n_bets, avg_odds
                
                if max(n_bets) == 0: # if no matches were qualified to bet on   
                    return (0,0,0)

            else:

            """






