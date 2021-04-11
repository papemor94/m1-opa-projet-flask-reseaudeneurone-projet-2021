import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
clf = RandomForestClassifier(n_estimators = 100) 
from sklearn import metrics

def load(input_file):
    # comma delimited is the default
    df = pd.read_csv(input_file, header = 0)
    df = pd.read_csv(input_file, header = 0, delimiter = ";")

    df = df[:-19000]

    # remove the non-numeric columns
    df = df._get_numeric_data()
    
    # put the numeric column names in a python list
    features_names = list(df.columns.values)
    # create a numpy array with the numeric values for input into scikit-learn
    return df

def get_features_and_target(df , target_to_predicte):
    target=df[target_to_predicte]
    del df[target_to_predicte]
    features=df
    return features, target

df = load("trainingData.csv")
X, y =get_features_and_target(df, 'LONGITUDE')
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)



# performing predictions on the test dataset
clf.fit(X_train, np.asarray(y_train, dtype="|S6"))
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error   
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(np.asarray(y_test, dtype="|S6"), y_pred))


print("on a utilisé ", len(df), "donnees")

# confusion matrix 

from sklearn.metrics import confusion_matrix
confusion_matrix(np.asarray(y_test, dtype="|S6"), y_pred)