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

def load (input):
    
    data = pd.read_csv(input, delimiter = ",")
    extraced_ssid_level_pivot =data.pivot(columns=['bssid'], values='level')
    
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


dataset  = load('donnees.txt') ;   # on souhaitera ajouter des donnees dans ce dataset
dataset['salleid'] = dataset['salleid'].astype(str)

dataset = dataset.groupby('positionid').agg("max").fillna(100)
dataset

# 8
X, y =get_features_and_target(dataset, 'salleid')
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=20, max_depth=8)
clf.fit(X_train, y_train)

y_pred =clf.predict(X_test)
print(clf.score(X_test, y_test))
y_pred


# In[3]:


import pickle
from sklearn import tree
pickle.dump(clf, open('model2.pkl','wb')) 


# In[4]:


#data = dataset['00:4e:35:c8:ce:24']=9999
def clean_input(dataset , X_dict): 
    X_dataframe = pd.DataFrame.from_dict(X_dict)
    for feature in X_dataframe: 
        if feature in dataset:
            print(feature)
        else:
            del X_dataframe[feature]
    return  X_dataframe

def threat_input(dataset , X_input):
    X_input= clean_input(dataset , X_input)
    key_array = []
    for feature in X_input:
        key_array.append(feature)
    values = X_input.values
    maj_dataset =  dataset.append(dict(zip(key_array, *values)), ignore_index=True)
    formated_input = maj_dataset.replace(np.nan , 100)
    formated_input = formated_input[-1:].values
    return formated_input


# In[5]:



# In[6]:


from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from pandas import json_normalize

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def makecalc():
    print(request.json)
    observation = threat_input(dataset , request.json)
    print(threat_input(dataset , request.json))
    print ("predicted" , clf.predict(observation) , type(clf.predict(observation)))
    return np.array2string(clf.predict(observation))

@app.route('/app', methods=['GET','POST'])
def makecalcc():
    print(request.json)
    return "computed"
if __name__ == '__main__':
    modelfile = 'model2.pkl'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=False, host='0.0.0.0')
# ici le serveur traite les donnees qu'il recoit 

