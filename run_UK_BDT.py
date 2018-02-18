from BettingUKleague import Estimate_Outcome
import matplotlib.pyplot as plt
from matplotlib import gridspec
from multiprocessing import Pool
import numpy as np
import time
import os
import glob
from joblib import Parallel, delayed

# Timer 
start_time = time.time()



def main(country,data_year,round_to_estimate,n_rounds_training,min_odds,train_size='full'):
    
    outcome = Estimate_Outcome(country,data_year,round_to_estimate=round_to_estimate,n_rounds_training=n_rounds_training,min_odds=min_odds,all_vs_all=False)
    season_results = outcome.load_results()
    widetable = outcome.make_widetable(season_results[0])
    #masseys = outcome.masseys(season_results[0],widetable)
    if train_size == 'full': # obsolete atm
        train_sample = outcome.make_training_sample(widetable,season_results)
    elif train_size == 'season':
        train_sample = outcome.make_training_sample(widetable,season_results)
    else: raise Exception("train_size must be 'full' or 'season'") 
    rest = outcome.rest(widetable,season_results,train_sample)
    return rest

all_vs_all = False
test_training = 10
test_min_odds = 0.0
test_train_size = 'season'


#for test_min_odds in np.linspace(1.5,3.5,21):
print test_min_odds
if all_vs_all:
	text_file_name = "All_vs_all_training_%s_rounds_sample_%s_min_odds_%1.1f.txt" % (test_training,test_train_size,test_min_odds)
else: 
	text_file_name = "One_vs_all_training_%s_rounds_sample_%s_min_odds_%1.1f.txt" % (test_training,test_train_size,test_min_odds)

data_array = []
#Make it all saved in a big array

country = 'UK'
# path with data
data_path = '/Users/mjensen/Desktop/Universitet/Adaboost/data/premier_league/'


for file in os.listdir(data_path):
    if file.endswith(".csv"):
        print file

        rest = Parallel(n_jobs=-1)(delayed((main))(country,file,i_r,test_training,min_odds=test_min_odds,train_size=test_train_size) for i_r in range(test_training + 1,37))

        for i_d in rest:
        	for i_m in i_d:
        		data_array.append([i_m[0],i_m[1],i_m[2],i_m[3],i_m[4],i_m[5][0],i_m[5][1],i_m[5][2]])

print time.time() - start_time

save_path = '/Users/mjensen/Desktop/Universitet/Adaboost/BDT_scores/premier_league/'
np.savetxt(save_path + text_file_name, data_array,fmt='%s')         

print "saved file at: %s as %s " %(save_path, text_file_name)
