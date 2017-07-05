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
"""

url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/search/"
search_param = {"eventName": "Solar Flare",
                "startSearchDate": "2010-05-01",
                "endSearchDate": "2015-07-01"}

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