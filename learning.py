#!/usr/bin/env python
# coding: utf-8

# In[147]:


import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
clf = RandomForestClassifier(n_estimators = 100) 
from sklearn import metrics


def load(data):
    extraced_ssid_level_pivot = data.pivot(columns=['bssid'], values='level')
    output =pd.concat([data, extraced_ssid_level_pivot] , axis=1)
    if 'level' in output:
        del output['level']
    if 'ssid' in output:
        del output['ssid']
    if 'bssid' in output:
        del output['bssid']
    return output


def get_features_and_target(df , target_to_predicte):
    target=df[target_to_predicte]
    if target_to_predicte in df:
        del df[target_to_predicte]
    features=df
    return features, target


# load data from database in json
import http.client
import requests
r = requests.get('http://127.0.0.1:8091/android/wap')
print(r.status_code)
# parse data from json  to dataframe
frame = pd.DataFrame(r.json())
print(frame)
# group by date of scan 
#dataset = df.groupby('date').agg('max')
dataset = load(frame)
print("dataset before group by date " , dataset)

dataset['date'] = dataset['date'].astype('float')

dataset = dataset.groupby('date').agg("max")
dataset = dataset.fillna(100)
#dataset = dataset.groupby('date').agg('max')
print( dataset.iloc[0])
