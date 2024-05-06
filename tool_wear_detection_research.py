# Author: Muhammet Ali Aykanat - https://github.com/MAAykanat
# Date: 06/05/2024
# Version: 1.0
# Description: Tool Wear Detection Research
# Target: Tool Condition (Worn/Unworn)

# Features: 52
# Observations: 25286

#############
### Notes ###
#############

### 1. The dataset is combined.csv from train and experiment datasets
### 2. Convert TARGET to binary (0,1) --> 0: Unworn, 1: Worn
### 3. Drop list: 
# 'MACHINING_PROCESS' -->  Not sure what it is and how it is related to the target
# 'Z1_CURRENTFEEDBACK', --> All same - 0.0
# 'Z1_DCBUSVOLTAGE', --> All same - 0.0
# 'Z1_OUTPUTCURRENT', --> All same - 0.0
# 'Z1_OUTPUTVOLTAGE', --> All same - 0.0
# 'S1_COMMANDACCELERATION' --> Actual value is 'S1_ACTUALPOSITION' is there, no need to command
# 'S1_SYSTEMINERTIA' --> All same - 12.0
# 'M1_CURRENT_PROGRAM_NUMBER' --> There is miss leading information all should be 1
# 'M1_SEQUENCE_NUMBER' --> There is miss leading information all should be 1
# 'M1_CURRENT_FEEDRATE' --> There is miss leading information, correct one is 'FEEDRATE'
# 'EXP_NO' --> Experiment number is not needed, it is cardinal variable

### 4. List should convert to numeric:
# 'FEEDRATE'
# 'CLAMP_PRESSURE'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from shutil import get_terminal_size
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Show all the decimals
pd.set_option('display.width', get_terminal_size()[0]) # Get bigger terminal display width

# Load the dataset
df = pd.read_csv('dataset/combined.csv')

# Capitalize the column names
df.columns = df.columns.str.upper()

# Notes Implementation

df["TARGET"] = df["TARGET"].apply(lambda x: 1 if x == "worn" else 0)

drop_list = ['MACHINING_PROCESS', 'Z1_CURRENTFEEDBACK', 'Z1_DCBUSVOLTAGE', 
             'Z1_OUTPUTCURRENT', 'Z1_OUTPUTVOLTAGE', 'S1_COMMANDACCELERATION', 
             'S1_SYSTEMINERTIA', 'M1_CURRENT_PROGRAM_NUMBER', 'M1_SEQUENCE_NUMBER', 
             'M1_CURRENT_FEEDRATE', "EXP_NO"]

df.drop(drop_list, axis=1, inplace=True)

cat_to_numeric = ['FEEDRATE', 'CLAMP_PRESSURE'] # Convert to numeric

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

check_df(df)

# Shape: (25286, 52)
"""
MACHINING_PROCESS             object
EXP_NO                         int64
FEEDRATE                       int64
CLAMP_PRESSURE               float64
TARGET                        object
Others are                   float64
"""
# No missing values in the dataset

# 2. Catch Numeric and Categorical Value

"""
Observations: 25286
Variables: 52
categorical_cols: 12
num_cols: 40
categorical_but_cardinal: 0
numeric_but_categorical: 10
"""

def grap_column_names(dataframe, categorical_th=10, cardinal_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included."""

    """
    Cardinal Variables: Variables that are categorical and do not carry information,
    that is, have too many classes, are called variables with high cardinality.
    """

    """
    Returns
    ------
        categorical_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        categorical_but_cardinal: list
                Categorical variables with high cardinality list
    """
    # categorical_cols, categorical_but_cardinal
    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numeric_but_categorical = [col for col in dataframe.columns if dataframe[col].nunique() < categorical_th and
                   dataframe[col].dtypes != "O"]
    categorical_but_cardinal = [col for col in dataframe.columns if dataframe[col].nunique() > cardinal_th and
                   dataframe[col].dtypes == "O"]
    categorical_cols = categorical_cols + numeric_but_categorical
    categorical_cols = [col for col in categorical_cols if col not in categorical_but_cardinal]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in numeric_but_categorical]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'categorical_cols: {len(categorical_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'categorical_but_cardinal: {len(categorical_but_cardinal)}')
    print(f'numeric_but_categorical: {len(numeric_but_categorical)}')

    return categorical_cols, num_cols, categorical_but_cardinal

cat_cols, num_cols, cat_but_car = grap_column_names(df)

print("\n\nCategorical Columns: \n", cat_cols)
print("\n\nNumeric Columns: \n", num_cols)
[print("\n\nCategorical but Cardinal EMPTY!!!\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 3. Categorical Variables Analysis

def cat_summary(dataframe, col_name, plot=False):
    """
    This function shows the frequency of categorical variables.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    plot : bool, optional
        The default is False.
    Returns
    -------
    None.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)
print("#"*50)
# Drop cols should be Numeric
cat_cols = [col for col in cat_cols if col not in cat_to_numeric]

print(cat_cols)

# 4. Numeric Variables Analysis

for col in cat_to_numeric:
    num_cols.append(col)

def numerical_col_summary(dataframe, col_name, plot=False):

    """
    This function shows the frequency of numerical variables.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    plot : bool, optional
        The default is False.
    Returns
    -------
    None.
    """
    print(dataframe[col_name].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)
    print("##########################################")
    if plot:
        sns.histplot(dataframe[col_name], kde=True)
        plt.xlabel(col_name)
        plt.title(f"{col_name} Distribution")
        plt.show()

for col in num_cols:
    numerical_col_summary(df,col)
print("#"*50)

# 5. Target Variable Analysis (Dependent Variable) - Categorical

"Non-Sense to analyze the target variable, There is only 1 categorical variable"

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    This function shows the mean of the target variable according to the categorical variable.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    categorical_col : str
        The name of the categorical variable.
    Returns
    -------
    None.
    """
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "TARGET", col)

print("#"*50)

