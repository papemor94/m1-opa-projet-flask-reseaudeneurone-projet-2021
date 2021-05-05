#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 


from sklearn.ensemble import RandomForestClassifier
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
#dataset['salleid'] = dataset['salleid'].astype(str)


# In[2]:



dataset = dataset.groupby('positionid').agg("max").fillna(100)
dataset


# In[3]:


from sklearn.neighbors import KNeighborsClassifier
def learn_and_generate_model() : 
    X, y =get_features_and_target(dataset, 'salleid')
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.50, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, np.asarray(y_train, dtype="|S6"))
    y_pred =  neigh.predict(X_test)
    # metrics are used to find accuracy or error   
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(np.asarray(y_test, dtype="|S6"), y_pred))
    from sklearn.metrics import confusion_matrix
    return neigh , confusion_matrix(np.asarray(y_test, dtype="|S6"), y_pred)

learn_and_generate_model()


# In[4]:


import pickle
from sklearn import tree
pickle.dump(clf, open('model.pkl','wb'))


# In[11]:


from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from pandas import json_normalize

arr =np.array([[1, 2, 3], [4, 5, 6]])
app = Flask(__name__)

#res = request.json()
#df = pd.DataFrame(data)

@app.route('/predict', methods=['GET','POST'])
def makecalc():
    print(request.json)
    majdataset = dataset.append(request.json, ignore_index=True)
    majdataset = majdataset.fillna(100)
    last_scan = majdataset[-1:].values
    salle_predicted = model.predict(last_scan)
    salle_predicted_json = np.array2string((salle_predicted))
    return request.json

@app.route('/app', methods=['GET','POST'])
def makecalcc():
    print(request.json)
    dataset.append(request.json, ignore_index=True)
    print(dataset)
    return request.json
if __name__ == '__main__':
    modelfile = 'model.pkl'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=False, host='0.0.0.0')
# ici le serveur traite les donnees qu'il recoit 


# In[ ]:


newDfObj = dataset.append({'salleid': '255',
                         'centrefrequence0': '5888'}, ignore_index=True)
newDfObj
newDfObj =newDfObj.fillna(100)
del newDfObj['salleid']  
 
inp = newDfObj[-1:].values
inp


# In[49]:


c = {'0': {'ssid': 'UnivToulon', 'bssid': '1c:28:af:ce:e7:01', 'level': -33, 'centerFreq0': 2437, 'frequency': 2437}, '1':
 {'ssid': 'eduroam', 'bssid': '1c:28:af:ce:e7:00', 'level': -33, 'centerFreq0': 2437, 'frequency': 2437}, 
'2': {'ssid': 'visiteurs', 'bssid': '1c:28:af:ce:e7:02', 'level': -33, 'centerFreq0': 2437, 'frequency': 2437}}


# In[73]:


c['0']
new = pd.DataFrame.from_dict(c)


d = dict()
for i in range (len(c)):
    print(i)
    print(c['0']['bssid'])
    print(c['0']['level'])
    #d[c['0']['bssid']].append(c['0']['level'])
    d.update( {'Germany' : 49} )
majdataset = dataset.append(request.json, ignore_index=True)
    


# In[34]:


extraced_ssid_level_pivot =df.pivot(columns=['bssid'], values='level')


# In[21]:


dataset


# In[27]:


majdataset = dataset.append(c['1'], ignore_index=True)
majdataset = majdataset.append(c['1'], ignore_index=True)
majdataset


# In[1]:


0

