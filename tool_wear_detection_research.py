import numpy as no
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.5f' % x) # Show all the decimals

# Load the dataset

df_train = pd.read_csv('dataset/train.csv')

total = 0

for i in range(1,10):
    globals()['df_%s' % i] = pd.read_csv(f'dataset/experiment_0{i}.csv')
    total =total + globals()['df_%s' % i].shape[0]

for i in range(10,19):
    globals()['df_%s' % i] = pd.read_csv(f'dataset/experiment_{i}.csv')
    total =total + globals()['df_%s' % i].shape[0]

for i in range(1,19):
    print(f"df_{i} shape: {globals()['df_%s' % i].shape}")


#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# 1. General Picture of the Dataset
# 2. Catch Numeric and Categorical Value
# 3. Catetorical Variables Analysis
# 4. Numeric Variables Analysis
# 5. Target Variable Analysis (Dependent Variable) - Categorical
# 6. Target Variable Analysis (Dependent Variable) - Numeric
# 7. Outlier Detection
# 8. Missing Value Analysis
# 9. Correlation Matrix

#############
# Train SET #
#############

# 1. General Picture of the Dataset
# 2. Missing Value Analysis - Handle

# 1. General Picture of the Dataset
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df_train)

# 2. Missing Value Analysis - Handle

print(df_train.isnull().sum())

# There are 4 missing values in passed_visual_inspection column

# Fill NaN values with "no"

df_train["passed_visual_inspection"] = df_train["passed_visual_inspection"].fillna("no")

print(df_train.isnull().sum())


