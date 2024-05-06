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

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

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

df["TARGET"] = df["TARGET"].apply(lambda x: 1 if x == "worn" else 0) # Convert to binary

drop_list = ['MACHINING_PROCESS', 'Z1_CURRENTFEEDBACK', 'Z1_DCBUSVOLTAGE', 
             'Z1_OUTPUTCURRENT', 'Z1_OUTPUTVOLTAGE', 'S1_COMMANDACCELERATION', 
             'S1_SYSTEMINERTIA', 'M1_CURRENT_PROGRAM_NUMBER', 'M1_SEQUENCE_NUMBER', 
             'M1_CURRENT_FEEDRATE', "EXP_NO", "Z1_COMMANDVELOCITY", "Z1_COMMANDACCELERATION"]

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

# 6. Target Variable Analysis (Dependent Variable) - Numeric
def target_summary_with_num(dataframe, target, numerical_col):
    """
    This function shows the average of numerical variables according to the target variable.
    
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    numerical_col : str
        The name of the numerical variable.
    Returns
    -------
    None.
    """
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "median", "count"]}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "TARGET", col)
print("#"*50)


# 7. Outlier Detection
# Outlier List below
"""
X1_ACTUALPOSITION: False, X1_ACTUALVELOCITY: True, X1_ACTUALACCELERATION: True
X1_COMMANDPOSITION: False, X1_COMMANDVELOCITY: True, X1_COMMANDACCELERATION: True
X1_CURRENTFEEDBACK: True, X1_DCBUSVOLTAGE: True, X1_OUTPUTCURRENT: False
X1_OUTPUTVOLTAGE: True, X1_OUTPUTPOWER: True, Y1_ACTUALPOSITION: False
Y1_ACTUALVELOCITY: True, Y1_ACTUALACCELERATION: True, Y1_COMMANDPOSITION: False
Y1_COMMANDVELOCITY: True, Y1_COMMANDACCELERATION: True, Y1_CURRENTFEEDBACK: True
Y1_DCBUSVOLTAGE: True, Y1_OUTPUTCURRENT: False, Y1_OUTPUTVOLTAGE: True
Y1_OUTPUTPOWER: True, Z1_ACTUALPOSITION: False, Z1_ACTUALVELOCITY: True
Z1_ACTUALACCELERATION: True, Z1_COMMANDPOSITION: False, Z1_COMMANDVELOCITY: True
Z1_COMMANDACCELERATION: True, S1_ACTUALPOSITION: False, S1_ACTUALVELOCITY: False
S1_ACTUALACCELERATION: False, S1_COMMANDPOSITION: False, S1_COMMANDVELOCITY: False
S1_CURRENTFEEDBACK: True, S1_DCBUSVOLTAGE: True, S1_OUTPUTCURRENT: True
S1_OUTPUTVOLTAGE: False, S1_OUTPUTPOWER: True, FEEDRATE: False
CLAMP_PRESSURE: False
"""

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    This function calculates the lower and upper limits for the outliers.

    Calculation:
    Interquantile range = q3 - q1
    Up limit = q3 + 1.5 * interquantile range
    Low limit = q1 - 1.5 * interquantile range

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    q1 : float, optional
        The default is 0.05.
    q3 : float, optional
        The default is 0.95.
    Returns
    -------
    low_limit, up_limit : float
        The lower and upper limits for the outliers.
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
        This function checks dataframe has outlier or not.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    Returns
    -------
    bool
        True if the dataframe has outlier, False otherwise.
    """

    lower_limit, upper_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)

    if dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)].any(axis=None):
        # print(f'{col_name} have outlier')
        return True
    else:
        return False
    
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    This function replaces the outliers with the lower and upper limits.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    variable : str
        The name of the column to be analyzed.
    q1 : float, optional
        The default is 0.05.
    q3 : float, optional
        The default is 0.95.
    Returns 
    -------
    None
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

# Replace with thresholds
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Handle the outliers
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

# 8. Missing Value Analysis
# No missing values in the dataset
print(df.isnull().sum())

# 9. Correlation Matrix
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    This function returns the columns that have a correlation higher than the threshold value.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    plot : bool, optional
        The default is False.
    corr_th : float, optional
        The default is 0.90.
    Returns
    -------
    drop_list : list
        The list of columns that have a correlation higher than the threshold value.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

df_corr = df.corr()

# f, ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="magma")
# ax.set_title("Correlation Heatmap", color="black", fontsize=10)
# plt.show()

drop_corr_list=high_correlated_cols(df, plot=False, corr_th=0.95) # 0.95 selected since it is meaningful in statisticaly
print(drop_corr_list)

# Drop List
"""
['X1_COMMANDPOSITION', 'X1_COMMANDVELOCITY', 'Y1_COMMANDPOSITION', 
'Y1_COMMANDVELOCITY', 'Z1_COMMANDPOSITION', 'S1_COMMANDPOSITION', 
'S1_COMMANDVELOCITY', 'S1_DCBUSVOLTAGE', 'S1_OUTPUTVOLTAGE', 'S1_OUTPUTPOWER']
"""

print(df.shape) # (25286, 41)
df.drop(drop_corr_list, axis=1, inplace=True)
print(df.shape) # (25286, 31)

#######################################
######### FEATURE ENGINEERING #########
#######################################

# There are 6 steps to be taken in the Feature Engineering process.
# 1. Missing Values
# 2. Outlier Values Analysis
# 3. Feature Generation
# 4. Encoding
# 5. Standardization
# 6. Save the Dataset

# 1. Missing Values
# No missing values in the dataset

# 2. Outlier Values Analysis
# Outliers are handled in the previous steps

# 3. Feature Generation
# It will be implemanted lately after research.

# 4. Encoding
# No need for encoding, there is only one categorical variable (TARGET)
# It is already converted to binary at line 59

# 5. Standardization

cat_cols, num_cols, cat_but_car = grap_column_names(df)

for col in cat_cols:
    cat_cols = [col for col in cat_cols if col not in cat_to_numeric]

for col in cat_to_numeric:
    num_cols.append(col)

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())
print(df.shape)

# 6. Save the Dataset
df.to_csv('dataset/combined_cleaned.csv', index=False)