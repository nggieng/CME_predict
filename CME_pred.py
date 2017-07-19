# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 10:51:12 2017

@author: nggieng
based on Bobra, M.
First, we'll import some modules.
"""

import numpy as np, matplotlib.pylab as plt, matplotlib.mlab as mlab, pandas as pd, mpld3, requests, urllib, json
from datetime import datetime as dt_obj
from datetime import timedelta
from sklearn import svm
from sklearn import cross_validation
from mpld3 import plugins
from sunpy.time import TimeRange
import sunpy.instr.goes
from scipy.stats import pearsonr as pearse

#%matplotlib inline
get_ipython().magic('matplotlib inline')
#%config InlineBackend.figure_format='retina'
get_ipython().magic("config InlineBackend.figure_format='retina'")
"""
Now we'll gather the data.
Data extended from  "startSearchDate": "2010-05-01",
                "endSearchDate": "2017-06-30"
"""

url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/search/"
search_param = {"eventName": "Solar Flare",
                "startSearchDate": "2010-05-01",
                "endSearchDate": "2017-06-30"}

t_start = search_param['startSearchDate']
t_end   = search_param['endSearchDate']

response = requests.post(url, data=search_param)
html = response.content
events = pd.read_html(html, attrs={'id':'FLR_table'})[0]

# get rid of the NaNs, as their presence disallows str.contains() function to be executed properly:
events = events.fillna('nothing')

# find all the events with CMEs
events = events[events['Directly Linked Event(s)'].str.contains("CME")]

# find all the flares that are M1.0-class or greater
events = events[events['Class'].str.contains("M|X")]

n_elements = len(events)
print "There are", n_elements, "possible CME events (positive class)."

# the following output is in the Stanford Digital Repository:
print "Class, Peak Time, Active Region Number, Directly Linked Event(s)"
for i in range(len(events)):
    print events['Class'].iloc[i],events['Peak Time'].iloc[i],events['Active Region Number'].iloc[i],events['Directly Linked Event(s)'].iloc[i]
    
    
#Check detail on Bobra's code
#Verify the DONKI data by comparing it to the GOES data.
#If, for any given event, both data are the same, then it's correct.
#To do this, we'll use the instr.goes.get_goes_event_list() function in SunPy
#So let's first convert all our times into datetime_objects.

#Positive class will be flaring active regions that did produce a CME.
#Negative class will be flaring active regions that did not produce a CME. 
def parse_tai_string(tstr):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    return dt_obj(year,month,day,hour,minute)

# create an array of datetime objects 
x = np.array([(parse_tai_string(events['Peak Time'].iloc[i])) for i in range(n_elements)])

# Case 1: CME and Flare exist but NOAA active region number does not exist in DONKI database
number_of_donki_mistakes = 0
for i in range(n_elements):
    if (('CME' in str(events['Directly Linked Event(s)'].iloc[i])) and ('nothing' in str(events['Active Region Number'].iloc[i]))):
        time = x[i]
        time_range = TimeRange(time,time)
        listofresults = sunpy.instr.goes.get_goes_event_list(time_range,'M1')
        if (listofresults[0]['noaa_active_region'] == 0):
            continue
        else:
            print "Missing NOAA number:",events['Active Region Number'].iloc[i],events['Class'].iloc[i],events['Peak Time'].iloc[i],"should be",listofresults[0]['noaa_active_region'],"; changing now."
            events['Active Region Number'].iloc[i] = listofresults[0]['noaa_active_region']
            number_of_donki_mistakes += 1
print 'There are',number_of_donki_mistakes,'DONKI mistakes so far.'

# Grab all the data from the GOES database
time_range = TimeRange(t_start,t_end)
listofresults = sunpy.instr.goes.get_goes_event_list(time_range,'M1')
print 'Grabbed all the GOES data; there are',len(listofresults),'events.'
# the following output is in the Stanford Digital Repository:
print 'NOAA Active Region Number, Class, Peak Time'
for i in range(len(listofresults)):#
    print listofresults[i]['noaa_active_region'],listofresults[i]['goes_class'],listofresults[i]['peak_time']
    
    
# Case 2: NOAA active region number is wrong in DONKI database
for i in range(len(listofresults)):
    match_peak_times = np.where(x == listofresults[i]['peak_time'])
    if (match_peak_times[0].size == 0):
        continue
    j = match_peak_times[0][0]
    if (events['Active Region Number'].iloc[j] == 'nothing'):
        continue
    if ((listofresults[i]['noaa_active_region']) != int(events['Active Region Number'].iloc[j])):
        print 'Messed up NOAA number:',int(events['Active Region Number'].iloc[j]),events['Class'].iloc[j],events['Peak Time'].iloc[j],"should be",listofresults[i]['noaa_active_region'],"; changing now."
        events['Active Region Number'].iloc[j] = listofresults[i]['noaa_active_region']
        number_of_donki_mistakes += 1
print 'There are',number_of_donki_mistakes,'DONKI mistakes so far.'

# Case 3: The flare peak time is wrong in the DONKI database.
goes_peak_times = np.array([listofresults[i]['peak_time'] for i in range(len(listofresults))])
for i in range(n_elements):
    this_peak_time = x[i]
    max_peak_time = this_peak_time + timedelta(0,0,0,0,6)
    min_peak_time = this_peak_time - timedelta(0,0,0,0,6)
    match_peak_times = (np.logical_and(goes_peak_times <= max_peak_time, goes_peak_times >= min_peak_time))
    if not any(match_peak_times):
        print 'DONKI flare peak time of',this_peak_time,'is not an actual GOES flare peak time. Flag event.'
        events['Peak Time'].iloc[i] = 'nothing'
        number_of_donki_mistakes += 1
print 'There are',number_of_donki_mistakes,'DONKI mistakes.'

# check HARPS numbers 
answer = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt',sep=' ')

# also drop elements where the noaa active region equals 'nothing' or 0
# the following assumes one match for match_nothing_ar or match_zero (could generalize this)
match_nothing_ar = np.where(events['Active Region Number'] == 'nothing')
match_zero = np.where(events['Active Region Number'] == 0)
if (match_nothing_ar[0].size > 0):
    events = events[events['Active Region Number'] != 'nothing']
if (match_zero[0].size > 0):
    events = events[events['Active Region Number'] != 0]

match_nothing_time = np.where(events['Peak Time'] == 'nothing')
if (match_nothing_time[0].size > 0):
    events = events[events['Peak Time'] != 'nothing']

n_elements = len(events)

# Now, let's determine at which time we'd like to predict CMEs.
timedelayvariable = 24

# convert the GOES Peak Time format into one that JSOC can understand
# subtract timedelayvariable hours before the GOES Peak Time and convert into a list of strings
def create_tai_string(x):
    t_rec = [] 
    for i in range(len(x)):
        x[i] = (x[i] - timedelta(hours=timedelayvariable))
        t_rec.append(x[i].strftime('%Y.%m.%d_%H:%M_TAI'))
    print "All times have been converted."
    return t_rec

# create an array of datetime objects 
x = np.array([(parse_tai_string(events['Peak Time'].iloc[i])) for i in range(n_elements)])
t_rec = create_tai_string(x)
# All times have been converted.

# time true value
events = events.rename(columns={'Peak Time': 'One Day Before Peak Time'})
events['One Day Before Peak Time'] = t_rec

#########################################################################
# grab the SDO data from the JSOC database by executing the JSON queries#
#########################################################################

def get_the_jsoc_data():
    
    catalog_data = []
    classification = []
    
    for i in range(n_elements):
    
        print "=====",i,"====="
        # next match NOAA_ARS to HARPNUM
        idx = answer[answer['NOAA_ARS'].str.contains(str(int(listofactiveregions[i])))]
       
        # if there's no HARPNUM, quit
        if (idx.empty == True):
            print 'skip: there are no matching HARPNUMs for', str(int(listofactiveregions[i]))
            continue
        
        #construct jsoc_info queries and query jsoc database; we are querying for 25 keywords
        url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.sharp_720s["+str(idx.HARPNUM.values[0])+"]["+t_rec[i]+"][? (CODEVER7 !~ '1.1 ') and (abs(OBS_VR)< 3500) and (QUALITY<65536) ?]&op=rs_list&key=USFLUX,MEANGBT,MEANJZH,MEANPOT,SHRGT45,TOTUSJH,MEANGBH,MEANALP,MEANGAM,MEANGBZ,MEANJZD,TOTUSJZ,SAVNCPP,TOTPOT,MEANSHR,AREA_ACR,R_VALUE,ABSNJZH"
        response = urllib.urlopen(url)
        status = response.getcode()
    
        # if there's no response at this time, quit
        if status!= 200:
            print 'skip: failed to find JSOC data for HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]
            continue
    
        # read the JSON output
        data = json.loads(response.read())
    
        # if there are no data at this time, quit
        if data['count'] == 0:
            print 'skip: there are no data for HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]
            continue
    
        # check to see if the active region is too close to the limb
        # we can compute the latitude of an active region in stonyhurst coordinates as follows:
        # latitude_stonyhurst = CRVAL1 - CRLN_OBS
        # for this we have to query the CEA series (but above we queried the other series as the CEA series does not have CODEVER5 in it)

        url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.sharp_cea_720s["+str(idx.HARPNUM.values[0])+"]["+t_rec[i]+"][? (abs(OBS_VR)< 3500) and (QUALITY<65536) ?]&op=rs_list&key=CRVAL1,CRLN_OBS"
        response = urllib.urlopen(url)
        status = response.getcode()
        
        # if there's no response at this time, quit
        if status!= 200:
            print 'skip: failed to find CEA JSOC data for HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]
            continue
    
        # read the JSON output
        latitude_information = json.loads(response.read())

        # if there are no data at this time, quit
        if latitude_information['count'] == 0:
            print 'skip: there are no data for HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]
            continue

        CRVAL1 = float(latitude_information['keywords'][0]['values'][0])
        CRLN_OBS = float(latitude_information['keywords'][1]['values'][0])
        if (np.absolute(CRVAL1 - CRLN_OBS) > 70.0):
            print 'skip: latitude is out of range for HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]
            continue
            
        if ('MISSING' in str(data['keywords'])):
            print 'skip: there are some missing keywords for HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]
            continue

        print 'printing data for NOAA Active Region number',str(int(listofactiveregions[i])),'and HARPNUM',idx.HARPNUM.values[0],'at time', t_rec[i]

        individual_flare_data = []
        for j in range(18):
            individual_flare_data.append(float(data['keywords'][j]['values'][0]))
    
        catalog_data.append(list(individual_flare_data))
    
        single_class_instance = [idx.HARPNUM.values[0],str(int(listofactiveregions[i])),listofgoesclasses[i],t_rec[i]]
        classification.append(single_class_instance)

    return catalog_data, classification

# fed the data into function
listofactiveregions = list(events['Active Region Number'].values.flatten())
listofgoesclasses = list(events['Class'].values.flatten())

# call the function
positive_result = get_the_jsoc_data()

# associated positive class
CME_data = positive_result[0]
positive_class = positive_result[1]
# the following output is in the Stanford Digital Repository:
print "There are", len(CME_data), "CME events (positive class)."
print "HARPNUM, NOAA Number, Class, Peak Time, USFLUX, MEANGBT, MEANJZH, MEANPOT, SHRGT45, TOTUSJH, MEANGBH, MEANALP, MEANGAM, MEANGBZ, MEANJZD, TOTUSJZ, SAVNCPP, TOTPOT, MEANSHR, AREA_ACR, R_VALUE, ABSNJZH"
for i in range(len(CME_data)):
    print positive_class[i][0], positive_class[i][1], positive_class[i][2], positive_class[i][3],positive_result[0][i][0],positive_result[0][i][1],positive_result[0][i][2],positive_result[0][i][3],positive_result[0][i][4],positive_result[0][i][5],positive_result[0][i][6],positive_result[0][i][7],positive_result[0][i][8],positive_result[0][i][9],positive_result[0][i][10],positive_result[0][i][11],positive_result[0][i][12],positive_result[0][i][13],positive_result[0][i][14],positive_result[0][i][15],positive_result[0][i][16],positive_result[0][i][17]

####################
# Negative class
####################
# select peak times that belong to both classes
all_peak_times = np.array([(listofresults[i]['peak_time']) for i in range(len(listofresults))])

#select peak times that belong to the positive class
positive_peak_times = np.array([(parse_tai_string(positive_class[i][3])) + timedelta(hours=timedelayvariable) for i in range(len(positive_class))])

negative_class_possibilities = [] 
counter_positive = 0
counter_negative = 0
for i in range(len(listofresults)):
    if (listofresults[i]['noaa_active_region'] < 10000):
        continue
    this_peak_time = all_peak_times[i]
    max_peak_time = this_peak_time + timedelta(0,0,0,0,6)
    min_peak_time = this_peak_time - timedelta(0,0,0,0,6)
    match_peak_times = np.where((np.logical_and(positive_peak_times <= max_peak_time, positive_peak_times >= min_peak_time)) == True)
    if (match_peak_times[0].shape[0] == 1):
        counter_positive +=1
    else:
        counter_negative += 1
        this_instance = [listofresults[i]['noaa_active_region'],listofresults[i]['goes_class'],listofresults[i]['peak_time']]
        negative_class_possibilities.append(this_instance)
print "There are", counter_positive,"maximal events in the positive class (the true number may be less than this)."
print "There are",counter_negative,"possible events in the negative class."

#compute times that are one day before the flare peak time: 
#create an array of datetime objects 
x = np.array([negative_class_possibilities[i][2] for i in range(len(negative_class_possibilities))])
t_rec = create_tai_string(x)
n_elements = len(t_rec)

#query the JSOC database to see if these data are present:
listofactiveregions = list(negative_class_possibilities[i][0] for i in range(n_elements))
listofgoesclasses = list(negative_class_possibilities[i][1] for i in range(n_elements))

negative_result = get_the_jsoc_data()

#events associated with the negative class:
no_CME_data = negative_result[0]
negative_class = negative_result[1]
# the following output is in the Stanford Digital Repository:
print "There are", len(no_CME_data), "no-CME events (negative class)."
print "HARPNUM, NOAA Number, Class, Peak Time, USFLUX, MEANGBT, MEANJZH, MEANPOT, SHRGT45, TOTUSJH, MEANGBH, MEANALP, MEANGAM, MEANGBZ, MEANJZD, TOTUSJZ, SAVNCPP, TOTPOT, MEANSHR, AREA_ACR, R_VALUE, ABSNJZH"
for i in range(len(no_CME_data)):
    print negative_class[i][0], negative_class[i][1], negative_class[i][2], negative_class[i][3],negative_result[0][i][0],negative_result[0][i][1],negative_result[0][i][2],negative_result[0][i][3],negative_result[0][i][4],negative_result[0][i][5],negative_result[0][i][6],negative_result[0][i][7],negative_result[0][i][8],negative_result[0][i][9],negative_result[0][i][10],negative_result[0][i][11],negative_result[0][i][12],negative_result[0][i][13],negative_result[0][i][14],negative_result[0][i][15],negative_result[0][i][16],negative_result[0][i][17]
    
###################
#feature selection#
###################

def create_flare_class(type_of_class):
    total_flare_class = []
    for i in range(len(type_of_class)):
        magnitude = float(type_of_class[i][2][1:4])
        flareclass = type_of_class[i][2][0]
        if (flareclass == 'M'):
            factor = 1.0
        if (flareclass == 'X'):
            factor = 10.0
        total_flare_class.append(factor*magnitude)
    return total_flare_class

positive_flare_class = np.array(create_flare_class(positive_class))
negative_flare_class = np.array(create_flare_class(negative_class))

CME_data = np.array(CME_data)
no_CME_data = np.array(no_CME_data)

CME_data = np.column_stack((CME_data, positive_flare_class))
no_CME_data = np.column_stack((no_CME_data, negative_flare_class))
print "Now we have", CME_data.shape[1], "features."

# how many flares associated with CME?
def normalize_the_data(flare_data):
    flare_data = np.array(flare_data)
    n_elements = flare_data.shape[0]
    for j in range(flare_data.shape[1]):
        standard_deviation_of_this_feature = np.std(flare_data[:,j])
        median_of_this_feature = np.median(flare_data[:,j])
        for i in range(n_elements):
            flare_data[i,j] = (flare_data[i,j] - median_of_this_feature) / (standard_deviation_of_this_feature)
    return flare_data

no_CME_data = normalize_the_data(no_CME_data)
CME_data = normalize_the_data(CME_data)

print "There are", no_CME_data.shape[0], "flares with no associated CMEs."
print "There are", CME_data.shape[0], "flares with associated CMEs."

#Feature for the active regions that both flared and produced a CME (green)
# and for the active regions that flared but did not produce a CME (red):

sharps = ['Total unsigned flux', 'Mean gradient of total field', 
'Mean current helicity (Bz contribution)', 'Mean photospheric magnetic free energy',
'Fraction of Area with Shear > 45 deg', 'Total unsigned current helicity',
'Mean gradient of horizontal field', 'Mean characteristic twist parameter, alpha',
'Mean angle of field from radial', 'Mean gradient of vertical field', 
'Mean vertical current density', 'Total unsigned vertical current', 
'Sum of the modulus of the net current per polarity',
'Total photospheric magnetic free energy density', 'Mean shear angle',
'Area of strong field pixels in the active region', 'Sum of flux near polarity inversion line',
'Absolute value of the net current helicity','Flare index']

i=6

# For the positive class (green)
mu_fl = np.mean(CME_data[:,i])
sigma_fl = np.std(CME_data[:,i])
num_bins = 15
n_fl, bins_fl, patches_fl = plt.hist(CME_data[:,i], num_bins, normed=1, facecolor='green', alpha=0.5)
y_fl = mlab.normpdf(bins_fl, mu_fl, sigma_fl)
plt.plot(bins_fl, y_fl, 'g--',label='positive class')

# For the negative class (red)
mu_nofl = np.mean(no_CME_data[:,i])
sigma_nofl = np.std(no_CME_data[:,i])
n_nofl, bins_nofl, patches_nofl = plt.hist(no_CME_data[:,i], num_bins, normed=1, facecolor='red', alpha=0.5)
y_nofl = mlab.normpdf(bins_nofl, mu_nofl, sigma_nofl)
plt.plot(bins_nofl, y_nofl, 'r--',label='negative class')

text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
plt.xlabel('Normalized '+sharps[i],**text_style)
plt.ylabel('Number (normalized)', labelpad=20,**text_style)
fig = plt.gcf()
fig.set_size_inches(10,5)
fig.savefig('fscore_tmp.png',bbox_inches='tight')
legend = plt.legend(loc='upper right', fontsize=12, framealpha=0.0,title='')
legend.get_frame().set_linewidth(0.0)
mpld3.enable_notebook()

#Computing the Univariate F-score for feature selection
from sklearn.feature_selection import SelectKBest, f_classif  # import the feature selection method
N_features = 19                                               # select the number of features 
Nfl = CME_data.shape[0]; Nnofl = no_CME_data.shape[0]
yfl = np.ones(Nfl); ynofl = np.zeros(Nnofl)
selector = SelectKBest(f_classif, k=N_features)               # k is the number of features
selector.fit(np.concatenate((CME_data,no_CME_data),axis=0), np.concatenate((yfl, ynofl), axis=0))
scores=selector.scores_
print scores

#interpret the scores in plot:
mpld3.disable_notebook()
plt.clf()
order = np.argsort(scores)
orderedsharps = [sharps[i] for i in order]
y_pos2 = np.arange(19)
plt.barh(y_pos2, sorted(scores/np.max(scores)), align='center')
plt.ylim((-1, 19))
plt.yticks(y_pos2, orderedsharps)
plt.xlabel('Normalized Fisher Score', fontsize=15)
plt.title('Ranking of SHARP features', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(8,10)
fig.savefig('sharp_ranking_48hours.png',bbox_inches='tight')
plt.show()

#Pearson linear correlation coefficients
xdata = np.concatenate((CME_data, no_CME_data), axis=0)
ydata = np.concatenate((np.ones(Nfl), np.zeros(Nnofl)), axis=0)

for i in range(len(sharps)):
    for j in range(len(sharps)):
        x = pearse(xdata[:,i],xdata[:,j])
        print "The correlation between",sharps[i],"and",sharps[j],"is",x[0],"."

#how many features?
CME_data = CME_data[:,0:18]
no_CME_data = no_CME_data[:,0:18]
print "Now we are back to", CME_data.shape[1], "features."

#run the support vector machine on the data
number_of_examples = Nfl + Nnofl
C = 4.0; gamma = 0.075; class_weight = {1:6.5}
clf = svm.SVC(C=C, gamma=gamma, kernel='rbf', class_weight=class_weight, cache_size=500, max_iter=-1, shrinking=True, tol=1e-8)

#performance ?
def confusion_table(pred, labels):
    """
    computes the number of TP, TN, FP, FN events given the arrays with predictions and true labels
    and returns the true skill score
  
    Args:
    pred: np array with predictions (1 for flare, 0 for nonflare)
    labels: np array with true labels (1 for flare, 0 for nonflare)
  
    Returns: true negative, false positive, true positive, false negative
    """  
    Nobs = len(pred)
    TN = 0.; TP = 0.; FP = 0.; FN = 0.
    for i in range(Nobs):
        if (pred[i] == 0 and labels[i] == 0):
            TN += 1
        elif (pred[i] == 1 and labels[i] == 0):
            FP += 1
        elif (pred[i] == 1 and labels[i] == 1):
            TP += 1 
        elif (pred[i] == 0 and labels[i] == 1):
            FN += 1
        else:
            print "Error! Observation could not be classified."
    return TN,FP,TP,FN

#True Skill Score (TSS)
array_of_avg_TSS = np.ndarray([50])
array_of_std_TSS = np.ndarray([50])
pred = -np.ones(number_of_examples)
xdata = np.concatenate((CME_data, no_CME_data), axis=0)
ydata = np.concatenate((np.ones(Nfl), np.zeros(Nnofl)), axis=0)
shuffle_index = np.arange(number_of_examples)                  # shuffle the data indices 
np.random.shuffle(shuffle_index)
ydata_shuffled = ydata[shuffle_index]
xdata_shuffled = xdata[shuffle_index,:]
for k in range(2,52):
    skf = cross_validation.StratifiedKFold(ydata_shuffled, n_folds=k)
    these_TSS_for_this_k = []
    for j, i in skf: 
        train = xdata_shuffled[j]; test = xdata_shuffled[i]     # test is examples in testing set; train is examples in training set
        ytrain = ydata_shuffled[j]; ytest = ydata_shuffled[i]   # ytest is labels in testing set; ytrain is labels in training set
        clf.fit(train, ytrain)
        pred[i] = clf.predict(test)
        TN,FP,TP,FN = confusion_table(pred[i], ydata_shuffled[i])
        if (((TP+FN) == 0.0) or (FP+TN)==0.0):
            these_TSS_for_this_k.append(-1.0)
            continue
        else:
            these_TSS_for_this_k.append(TP/(TP+FN) - FP/(FP+TN))
    TSS_k = np.array(these_TSS_for_this_k)
    array_of_avg_TSS[k-2]=np.mean(TSS_k)
    array_of_std_TSS[k-2]=np.std(TSS_k)
    
#plot mean TSS
fig, ax = plt.subplots(figsize=(10,8))      # define the size of the figure
orangered = (1.0,0.27,0,1.0)                # create an orange-red color
cornblue  = (0.39,0.58,0.93,1.0)            # create a cornflower-blue color

# define some style elements
marker_style_red  = dict(linestyle='', markersize=8, fillstyle='full',color=orangered,markeredgecolor=orangered)
marker_style_blue = dict(linestyle='', markersize=8, fillstyle='full',color=cornblue,markeredgecolor=cornblue)
text_style = dict(fontsize=16, fontdict={'family': 'monospace'})

# ascribe the data to the axes
k = np.arange(50)+2
for i in range(50):
    if (array_of_avg_TSS[i] > array_of_std_TSS[i]):
        ax.errorbar(k[i], array_of_avg_TSS[i], yerr=array_of_std_TSS[i], linestyle='',color=orangered)
        ax.plot(k[i], array_of_avg_TSS[i],'o',**marker_style_red)
    if (array_of_avg_TSS[i] <= array_of_std_TSS[i]):
        ax.errorbar(k[i], array_of_avg_TSS[i], yerr=array_of_std_TSS[i], linestyle='',color=cornblue)
        ax.plot(k[i], array_of_avg_TSS[i],'o',**marker_style_blue)
plt.xlim(xmax = 52, xmin = 0)

# label the axes and the plot
ax.set_xlabel('k',**text_style)
ax.set_ylabel('TSS',labelpad=20,**text_style)
plt.title(r'TSS per k using stratified k-fold cross-validation',**text_style)
fig = plt.gcf()
fig.set_size_inches(10,5)
mpld3.enable_notebook()

print "The TSS equals",array_of_avg_TSS[8],"plus or minus",array_of_std_TSS[8]
