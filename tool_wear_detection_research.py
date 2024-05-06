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

print(df.head())
print(df["target"].unique())
