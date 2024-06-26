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

    f = open('Ensemble_Model_classification_report.txt', 'a')
    f.writelines(f"########## {name} ##########\n")
    f.writelines(f"{classification_report(y_true=y_actual, y_pred=y_pred, digits=4, target_names=target_names)}\n")
    f.writelines(f"############################\n")
    f.close()

df = pd.read_csv('dataset/combined_cleaned_without_droplist.csv') # Without Dropped_list

X = df.drop('TARGET', axis=1)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

model = joblib.load('results_clearout/3-Fold Stratify/voting_clf.pkl')


classification_report_output(name="Voting Classifier-Soft",
                            y_actual=y_test,
                            y_pred=model.predict(X_test), target_names=["Unworn","Worn"])
    
