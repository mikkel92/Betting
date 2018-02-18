#from BettingUKleague import Estimate_Outcome

import matplotlib.pyplot as plt
from matplotlib import gridspec
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
import time
import os
import glob
from joblib import Parallel, delayed
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.metrics import make_scorer



# Timer 
start_time = time.time()


BDT_cut = 0.5 # cut over which you bet
y_ax = 100

all_data = []
# list for home (first) and away(second) games that has a score higher than cut
over_cut_games = { 
                    "game_nr_home":[],
                    "winnings_home":[],
                    "game_nr_away":[],
                    "winnings_away":[]
                    }

# path with data
data_path = '/Users/mjensen/Desktop/Universitet/Adaboost/BDT_scores/premier_league/'
files = glob.glob(data_path + 'One*' + '*10*' + '*season*' + '*0.0*') # the data files you want
for data_file in files:
    # lead columns in data file
    data = np.loadtxt(data_file,dtype={'names': ('country','BDT_score_h','BDT_score_d','BDT_score_a','match_outcome','odds_home','odds_draw','odds_away'),
                      'formats': ('S8', 'f4', 'f4', 'f4', 'i1', 'f4', 'f4','f4')})
    for data_row in data:
        all_data.append(data_row)

    test_min_odds = float(data_file[-7:-4])

    # arrays for signal and background data
    BDT_signal = {
                    "Home":[[],[]],
                    "Draw":[],
                    "Away":[[],[]],
                    "All":[]  }

    BDT_background = {
                    "Home":[[],[]],
                    "Draw":[],
                    "Away":[[],[]],
                    "All":[]  }

    # number of games where the winning team had less than min_odds as their odds
    games_under_min_odds = 0


    home_data_odds = 0 # combined odds for home team winnings
    home_data_above_BDT = 0 # nr of home matches above BDT cut
    away_data_odds = 0 # combined odds for away team winnings
    away_data_above_BDT = 0 # nr of away matches above BDT cut

    train_outcome_h = [] # labels for events for training on the 1 vs all bdt scores

    for i_d in range(0,len(data)): 

        if int(data["match_outcome"][i_d]) == 0 and data["odds_home"][i_d] > test_min_odds: # if home won and odds are above min odds
            train_outcome_h.append(1) # the match is a signal
            BDT_signal["Home"][0].append(data["BDT_score_h"][i_d]) # save bdt score for home team
            BDT_signal["Home"][1].append(data["BDT_score_a"][i_d]) # and away team
            if data["BDT_score_h"][i_d] > BDT_cut: # if the bdt score of the home team was above cut
                home_data_odds += data["odds_home"][i_d] # save odds from winning
                home_data_above_BDT += 1.0 # count number of home winnings above bdt cut
                over_cut_games["game_nr_home"].append([i_d,data["odds_home"][i_d],test_min_odds])
                
        else:
            train_outcome_h.append(0)
            BDT_background["Home"][0].append(data["BDT_score_h"][i_d])
            BDT_background["Home"][1].append(data["BDT_score_a"][i_d])
            if data["BDT_score_h"][i_d] > BDT_cut:
                home_data_above_BDT += 1.0
                over_cut_games["game_nr_home"].append([i_d,0.,test_min_odds])

        if int(data["match_outcome"][i_d]) == 1 and data["odds_draw"][i_d] > test_min_odds:
            BDT_signal["Draw"].append(data["BDT_score_d"][i_d])
        else:
            BDT_background["Draw"].append(data["BDT_score_d"][i_d])

        if int(data["match_outcome"][i_d]) == 2 and data["odds_away"][i_d] > test_min_odds:
            BDT_signal["Away"][0].append(data["BDT_score_a"][i_d])
            BDT_signal["Away"][1].append(data["BDT_score_h"][i_d])
            if data["BDT_score_a"][i_d] > BDT_cut:
                away_data_odds += data["odds_away"][i_d]
                away_data_above_BDT += 1.0
                over_cut_games["game_nr_away"].append([i_d,1.,test_min_odds,data["odds_away"][i_d]])
                
        else:
            BDT_background["Away"][0].append(data["BDT_score_a"][i_d])
            BDT_background["Away"][1].append(data["BDT_score_h"][i_d])
            if data["BDT_score_a"][i_d] > BDT_cut:
                away_data_above_BDT += 1.0
                over_cut_games["game_nr_away"].append([i_d,0.,test_min_odds,data["odds_away"][i_d]])
              

        if min(data["odds_home"][i_d],data["odds_draw"][i_d],data["odds_away"][i_d]) < test_min_odds:
            games_under_min_odds += 1
    print data_file
    print "signal events home: %i  --  background events home: %i" %(len(BDT_signal["Home"][0]), len(BDT_background["Home"][0]))
    print "signal events draw: %i  --  background events draw: %i" %(len(BDT_signal["Draw"]), len(BDT_background["Draw"]))
    print "signal events away: %i  --  background events away: %i" %(len(BDT_signal["Away"][0]), len(BDT_background["Away"][0]))
    print
    print "Home: combined odds: %f win: %f percent profit %f" %(home_data_odds, home_data_odds - home_data_above_BDT, home_data_odds / home_data_above_BDT)
    print "Away: combined odds: %f win: %f percent profit %f" %(away_data_odds, away_data_odds - away_data_above_BDT, away_data_odds / away_data_above_BDT)
    print "games under min_odds %i" %games_under_min_odds

accuracy_home = []
accuracy_draw = []
accuracy_away = []
for i_d in all_data:
    max_score = max ((i_d[1],i_d[2],i_d[3]))
    
    if i_d[1] == max_score:
        if i_d[4] == 0:
            accuracy_home.append(1.)
        else: accuracy_home.append(0.)

    if i_d[2] == max_score:
        if i_d[4] == 1:
            accuracy_draw.append(1.)
        else: accuracy_draw.append(0.)

    if i_d[3] == max_score:
        if i_d[4] == 2:
            accuracy_away.append(1.)
        else: accuracy_away.append(0.)

print "accuracy home: %1.2f -- fraction of home games won: %1.2f" % (sum(accuracy_home) / len(accuracy_home), float(len(BDT_signal["Home"][0]))/ float(len(BDT_background["Home"][0]) + len(BDT_signal["Home"][0])))
print "accuracy draw: %1.2f -- fraction of draw games won: %1.2f" % (sum(accuracy_draw) / len(accuracy_draw), float(len(BDT_signal["Draw"]))/ float(len(BDT_background["Draw"]) + len(BDT_signal["Draw"])))
print "accuracy away: %1.2f -- fraction of away games won: %1.2f" % (sum(accuracy_away) / len(accuracy_away), float(len(BDT_signal["Away"][0]))/ float(len(BDT_background["Away"][0]) + len(BDT_signal["Away"][0])))


unique_games_home = [0]
unique_odds_home = []

over_cut_games["game_nr_home"] = sorted(over_cut_games["game_nr_home"])

# sort games to not double count if over bdt cut for multiple min_odd
for i_r in range(0,len(over_cut_games["game_nr_home"])):
    if over_cut_games["game_nr_home"][i_r][0] != unique_games_home[-1]:
        unique_odds_home.append(over_cut_games["game_nr_home"][i_r][1])
    #print sorted(over_cut_games["game_nr_home"])[i_r]
    unique_games_home.append(over_cut_games["game_nr_home"][i_r][0])
    
unique_games_away = [0]
unique_odds_away = []


over_cut_games["game_nr_away"] = sorted(over_cut_games["game_nr_away"])
for i_r in np.linspace(len(over_cut_games["game_nr_away"]),1,len(over_cut_games["game_nr_away"])):
    
    i_r = int(i_r - 1)
    if over_cut_games["game_nr_away"][i_r][0] != unique_games_away[-1]:
        unique_odds_away.append(over_cut_games["game_nr_away"][i_r])
    unique_games_away.append(over_cut_games["game_nr_away"][i_r][0])

betting_cash = 100. # starting cash
bets_won = 0 # number of bets won
bets_lost = 0 # number of bets lost


for i_m in range(0,len(unique_odds_away)):  
    kelly_crit = 10. / betting_cash #(((0.8 / (unique_odds_away[i_m][2])) * unique_odds_away[i_m][3]) - 1.) / (unique_odds_away[i_m][3] - 1.) # kelly criterion
    #if kelly_crit < 0:
    #    continue
    if unique_odds_away[i_m][1] > 0:
        betting_cash += betting_cash * kelly_crit * unique_odds_away[i_m][3] - betting_cash * kelly_crit 
        bets_won += 1
        #print "win: %1.2f kelly_crit: %1.3f bets won: %i odds: %f" %(betting_cash, kelly_crit, bets_won, unique_odds_away[i_m][3])
    else:    
        betting_cash -= betting_cash * kelly_crit
        bets_lost += 1
        #print "loss: %1.2f kelly_crit: %1.3f bets_lost: %i odds: %f" %(betting_cash, kelly_crit, bets_lost, unique_odds_away[i_m][3])

print "winnings: %1.2f" %betting_cash

combined_odds_away = []
for i_d in range(0,len(unique_odds_away)):
    if unique_odds_away[i_d][1] > 0:
        combined_odds_away.append(unique_odds_away[i_d][3])

print sum(unique_odds_home),len(unique_odds_home)
print sum(combined_odds_away),len(unique_odds_away)

"""
plt.figure(figsize=(10,9))
plt.subplot(221)

plt.scatter(BDT_background["Home"][0],BDT_background["Home"][1],c='r')
plt.scatter(BDT_signal["Home"][0],BDT_signal["Home"][1],c='g')

plt.subplot(222)
plt.scatter(BDT_signal["Away"][0],BDT_signal["Away"][1],c='g')
plt.scatter(BDT_background["Away"][0],BDT_background["Away"][1],c='r')
plt.show()
"""


n_train = 2000
BDT_data = []
scan_params = False

for i_d in range(0,len(data)):
    BDT_data.append([data["BDT_score_h"][i_d],data["BDT_score_d"][i_d],data["BDT_score_a"][i_d]])
print np.shape(BDT_data)


def Adaboost(p,x):

    signal_BDT = []
    signal_over_cut = []
    background_BDT = []
    background_over_cut = []
    
    reg = AdaBoostClassifier(DecisionTreeClassifier(max_depth=int(p[0])), n_estimators=int(p[1]),
                             learning_rate=p[2], algorithm='SAMME')
    reg.fit(BDT_data[0:n_train], train_outcome_h[0:n_train])
    boosted_decisions = reg.decision_function(BDT_data[n_train:len(data)])

    for i_d in range(n_train,len(data)):
        if int(data["match_outcome"][i_d]) == 0 and data["odds_home"][i_d] > test_min_odds:
            signal_BDT.append(boosted_decisions[i_d - n_train])
            if boosted_decisions[i_d - n_train] > p[3]:
                signal_over_cut.append(boosted_decisions[i_d - n_train])
        
        else: 
            background_BDT.append(boosted_decisions[i_d - n_train])
            if boosted_decisions[i_d - n_train] > p[3]:
                background_over_cut.append(boosted_decisions[i_d - n_train])

    if len(background_over_cut) == 0 or len(signal_over_cut) == 0:
        return signal_BDT, background_BDT, 10
    else: return signal_BDT, background_BDT, float(len(background_over_cut)) / float(len(signal_over_cut)) 

if scan_params:
    best = 2.
    current = 2.
    for i_p1 in range(2,7):
        for i_p2 in np.linspace(5,50,10):
            print best , current
            for i_p3 in np.linspace(0.1,1.0,5):
                for i_p4 in np.linspace(0.5,1.0,6):
                    current = Adaboost((i_p1,i_p2,i_p3,i_p4),BDT_data)
                if current < best:
                    best = current
                    print (i_p1,i_p2,i_p3,i_p4)[2]
#signal_BDT, background_BDT, lal = Adaboost((3,40,1.0,0.3),BDT_data)

plt.figure(figsize=(10,9))

plt.subplot(221)
bins = np.linspace(-1,1,41)
plt.hist(np.array(BDT_signal["Home"][0]),histtype='step',linewidth=2.0,label='signal',bins=bins,color='r',alpha=0.5)
plt.hist(np.array(BDT_background["Home"][0]),histtype='step',linewidth=2.0,label='background',bins=bins,color='b',alpha=0.5)
plt.legend(loc=2)
plt.ylim(0,y_ax)
plt.title('Home')
plt.xlabel('BDT score')
plt.ylabel('frequence')

plt.subplot(223)
plt.hist(np.array(BDT_signal["Draw"]),histtype='step',linewidth=2.0,label='signal',bins=bins,color='r',alpha=0.5)
plt.hist(np.array(BDT_background["Draw"]),histtype='step',linewidth=2.0,label='background',bins=bins,color='b',alpha=0.5)
plt.legend(loc=2)
plt.ylim(0,y_ax)
plt.title('Draw')
plt.xlabel('BDT score')
plt.ylabel('frequence')


plt.subplot(222)
plt.hist(np.array(BDT_signal["Away"][0]),histtype='step',linewidth=2.0,label='signal',bins=bins,color='r',alpha=0.5)
plt.hist(np.array(BDT_background["Away"][0]),histtype='step',linewidth=2.0,label='background',bins=bins,color='b',alpha=0.5)
plt.legend(loc=2)
plt.ylim(0,y_ax)
plt.title('Away')
plt.xlabel('BDT score')
plt.ylabel('frequence')
plt.show(block=False)
"""
plt.subplot(224)
plt.hist(np.array(signal_BDT),histtype='step',linewidth=2.0,label='signal',bins=bins,color='r',alpha=0.5)
plt.hist(np.array(background_BDT),histtype='step',linewidth=2.0,label='background',bins=bins,color='b',alpha=0.5)
plt.legend(loc=2)
plt.ylim(0,30)
plt.title('Double training Home')
plt.xlabel('BDT score')
plt.ylabel('frequence')
plt.show(block=False)
"""
raw_input( ' ... ' )
plt.close('all')


