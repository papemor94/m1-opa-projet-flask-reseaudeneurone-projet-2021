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


def load2 (data2):
    extraced_ssid_level_pivot =data2.pivot(columns=['bssid'], values='level')
    output =pd.concat([data2, extraced_ssid_level_pivot] , axis=1)
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
r = requests.get('http://127.0.0.1:8091/android/wpa')
print(r.status_code)
print (r.json())
# parse data from json  to dataframe
df = pd.DataFrame(r.json())
print(df)
# group by date of scan 
dataset = df.groupby('date').agg('max')
df2 = load2(dataset)
df2 = df2.fillna(100)
print(df2.iloc[:-1])

#  here we can train the model

from sklearn.neighbors import KNeighborsClassifier
def learn_and_generate_model() : 
    X, y =get_features_and_target(dataset, 'salle')
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, np.asarray(y_train, dtype="|S6"))
    y_pred =  neigh.predict(X_test)
    # metrics are used to find accuracy or error   
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(np.asarray(y_test, dtype="|S6"), y_pred))
    from sklearn.metrics import confusion_matrix
    return neigh , confusion_matrix(np.asarray(y_test, dtype="|S6"), y_pred)

learn_and_generate_model()