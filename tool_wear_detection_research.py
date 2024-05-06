import numpy as no
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.5f' % x) # Show all the decimals

# Load the dataset
df = pd.read_csv('dataset/combined.csv')

# Capitalize the column names
df.columns = df.columns.str.upper()

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

