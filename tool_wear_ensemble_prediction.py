import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib

def classification_report_output(name, y_actual, y_pred, target_names=None):

    print(classification_report(y_true=y_actual, y_pred=y_pred, 
                                digits=4, target_names=target_names))

    f = open('Ensemble_Model_Classification_Report.txt', 'a')
    f.writelines(f"########## {name} ##########\n")
    f.writelines(f"{classification_report(y_true=y_actual, y_pred=y_pred, digits=4, target_names=target_names)}\n")
    f.writelines(f"############################\n")
    f.close()

df = pd.read_csv('dataset/aggregated_test_cleaned.csv')

X = df.drop('TARGET', axis=1)
y = df['TARGET']

model_voting = joblib.load('models/voting_clf.pkl')
model_knn = joblib.load('models/knn_model.pkl')
model_rf = joblib.load('models/rf_model.pkl')
model_lightgmb = joblib.load('models/lightgbm_model.pkl')


print("########## Ensemble Model ##########")
classification_report_output(name="Voting Classifier-Soft",
                            y_actual=y,
                            y_pred=model_voting.predict(X), target_names=["Unworn","Worn"])
print("############################")

print("########## KNN ##########")
classification_report_output(name="KNN",
                            y_actual=y,
                            y_pred=model_knn.predict(X), target_names=["Unworn","Worn"])
print("############################")

print("########## Random Forest ##########")
classification_report_output(name="RF",
                            y_actual=y,
                            y_pred=model_rf.predict(X), target_names=["Unworn","Worn"])
print("############################")

print("########## LightGBM ##########")
classification_report_output(name="LightGBM",
                            y_actual=y,
                            y_pred=model_lightgmb.predict(X), target_names=["Unworn","Worn"])
print("############################")