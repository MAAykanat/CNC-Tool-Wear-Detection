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

