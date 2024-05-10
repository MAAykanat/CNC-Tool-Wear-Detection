import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from shutil import get_terminal_size

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
print(df.dtypes)
#######################################
#### MODEL BUILDING AND EVALUATION ####
#######################################

# 1. Base Models
# 2. Automated Hyperparameter Optimization
# 3. Stacking & Ensemble Learning

# 1. Split Dataset

X = df.drop('TARGET', axis=1)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def base_models(X, y, scoring="roc_auc", cv=10, all_metrics=False):
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

    if (all_metrics == True):
        for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ({name}) ")
            print(f"F1: {round(cv_results['test_f1'].mean(), 4)} ({name}) ")
            print(f"ROC_AUC: {round(cv_results['test_roc_auc'].mean(), 4)} ({name}) ")
            
            f = open('Estimators.txt', 'a')
            f.writelines(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ({name})\n")
            f.writelines(f"F1: {round(cv_results['test_f1'].mean(), 4)} ({name})\n")
            f.writelines(f"ROC_AUC: {round(cv_results['test_roc_auc'].mean(), 4)} ({name})\n")
            f.close()

    else:
        for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
            
            f = open('Estimators.txt', 'a')
            f.writelines(f"Score: {round(cv_results['test_score'].mean(), 4)} ({name})\n")
            f.close()

base_models(X_train, y_train, scoring=["accuracy", "f1", "roc_auc" ], all_metrics=True)

# 2. Automated Hyperparameter Optimization
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, classifiers, cv=3, scoring="roc_auc", all_metrics=False):
    print("Hyperparameter Optimization....")
    best_models = {}

    if (all_metrics == True):
        for name, classifier, params in classifiers:
            print(f"########## {name} ##########")
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            print(f"Accuracy (Before): {round(cv_results['test_accuracy'].mean(), 4)}")
            print(f"F1 (Before): {round(cv_results['test_f1'].mean(), 4)}")
            print(f"ROC_AUC (Before): {round(cv_results['test_roc_auc'].mean(), 4)}")

            gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
            final_model = classifier.set_params(**gs_best.best_params_)

            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ({name}) ")
            print(f"F1: {round(cv_results['test_f1'].mean(), 4)} ({name}) ")
            print(f"ROC_AUC: {round(cv_results['test_roc_auc'].mean(), 4)} ({name})")
            print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

            f = open('Estimators_Hyperparameter.txt', 'a')
            f.writelines(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ({name})\n")
            f.writelines(f"F1: {round(cv_results['test_f1'].mean(), 4)} ({name})\n")
            f.writelines(f"ROC_AUC: {round(cv_results['test_roc_auc'].mean(), 4)} ({name})\n")
            f.writelines(f"{name} best params: {gs_best.best_params_}\n")
            f.close()

            best_models[name] = final_model
    else:
        for name, classifier, params in classifiers:
            print(f"########## {name} ##########")
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

            gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
            final_model = classifier.set_params(**gs_best.best_params_)

            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)

            print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
            print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

            f = open('Estimators_Hyperparameter.txt', 'a')
            f.writelines(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}\n")
            f.writelines(f"{name} best params: {gs_best.best_params_}\n")
            f.close()
            best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X_train, y_train, classifiers=classifiers, cv=10, scoring=["accuracy", "f1", "roc_auc" ], all_metrics=True)

print(best_models["CART"].fit(X_train,y_train).feature_importances_)
print(best_models["RF"].fit(X_train,y_train).feature_importances_)
print(best_models["XGBoost"].fit(X_train,y_train).feature_importances_)
print(best_models["LightGBM"].fit(X_train,y_train).feature_importances_)


################################
### PLOT- FEATURE IMPORTANCE ###
################################

def plot_importance(model, features, name, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features_{}'.format(name))
    plt.tight_layout()
    if save:
        plt.savefig('results/importances_{}.png'.format(name))
    plt.show()

for model in best_models:
    if model != "KNN":
        final_model = best_models[model].fit(X_train, y_train)
        plot_importance(final_model, X_train, name = model, save=True)
    else:
        pass

##############################
### PLOT- CONFUSION MATRIX ###
##############################
def plot_confusion_matrix(name, y_actual, y_pred, cmap='viridis', save=False):
    cm = confusion_matrix(y_actual, y_pred)

    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                        cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap=cmap)
    plt.title('{}'.format(name), fontsize=10)
    if save:
        plt.savefig('results/confusion_matrix_{}.png'.format(name))
        plt.savefig('results/confusion_matrix_{}.tiff'.format(name))
    plt.show()

for model in best_models:
    model_fit=best_models[model].fit(X_train, y_train)
    plot_confusion_matrix(name=model, 
                          y_actual=y_test, 
                          y_pred=model_fit.predict(X_test), 
                          save=True)
