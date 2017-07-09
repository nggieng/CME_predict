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


