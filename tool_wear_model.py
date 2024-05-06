import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from shutil import get_terminal_size

from sklearn.model_selection import train_test_split, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Show all the decimals
pd.set_option('display.width', get_terminal_size()[0]) # Get bigger terminal display width

# Load the data
df = pd.read_csv('dataset/combined_cleaned.csv')

print(df.head())

#######################################
#### MODEL BUILDING AND EVALUATION ####
#######################################

# 1. Base Models
# 2. Automated Hyperparameter Optimization
# 3. Stacking & Ensemble Learning

# 1. Split Dataset

X = df.drop('TARGET', axis=1)
y = df['TARGET']


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring, n_jobs=-1)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
        
        f = open('Estimators.txt', 'a')
        f.writelines(f"RMSE: {round(cv_results['test_score'].mean(), 4)} ({name})\n")
        f.close()

base_models(X, y, scoring="accuracy")