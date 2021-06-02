#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
clf = RandomForestClassifier(n_estimators = 100) 
from sklearn import metrics
import pickle
from sklearn import tree
try:
     import flask
except:
    get_ipython().system(' conda install --yes flask         ')

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from pandas import json_normalize

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

# recupere les trois max et leur index   cle valeur 
def get_max(predicted_proba , labels):
    dic = dict()
    for i in range(3):
        #print(predicted_proba)
        max_value = max(predicted_proba)
        max_index = predicted_proba.index(max_value)
        #print(max_index)
        key = labels[max_index]
        dic[key] = max_value*100
        predicted_proba[max_index]=-1
    return dic

dataset  = load('donnees.txt') ;   # on souhaitera ajouter des donnees dans ce dataset
dataset['salleid'] = dataset['salleid'].astype(str)

dataset = dataset.groupby('positionid').agg("max").fillna(100)
dataset

# 8
X, y =get_features_and_target(dataset, 'salleid')
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)
clf = RandomForestClassifier(criterion='gini', max_depth=8,n_estimators=50)
clf.fit(X_train, y_train)

y_pred =clf.predict(X_test)
print(clf.score(X_test, y_test))
print(y_pred)


# In[2]:


res  = clf.predict_proba(X_test)


# In[ ]:


labels = clf.classes_


# In[9]:


get_max(res[0].tolist(), labels)


# In[5]:


pickle.dump(clf, open('modelfinal.pkl','wb')) 


# In[6]:


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
    formated_input = maj_dataset.replace(np.nan,100)
    formated_input = formated_input[-1:].values
    return formated_input


# In[7]:


app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def makecalc():
    #print(request.json)
    #print(dataset)
    observation = threat_input(dataset , request.json)
    #print(threat_input(dataset , request.json))
    print ("predicted" , model.predict(observation) , type(model.predict(observation)))
    proba  = model.predict_proba(observation)
    print ("result" , get_max(proba[0].tolist()  , labels))
    return json.dumps(get_max(proba[0].tolist()  , labels))
@app.route('/app', methods=['GET','POST'])
def makecalcc():
    print(request.json)
    return "computed"
if __name__ == '__main__':
    modelfile = 'modelfinal.pkl'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=False, host='0.0.0.0' , port=6000)
# ici le serveur traite les donnees qu'il recoit 


# In[8]:


try:
     import pandas_profiling
except:
    get_ipython().system(' conda install --yes pandas-profiling   ')
try:
     import geopandas
except:
    get_ipython().system(' conda install --yes geopandas    ')


# In[ ]:


import pandas as pd  
import numpy as np
import pandas_profiling
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


pandas_profiling.ProfileReport(dataset)

