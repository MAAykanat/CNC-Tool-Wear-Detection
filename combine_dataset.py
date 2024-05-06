import numpy as np
import pandas as pd

# Load the dataset
# Train set and Experiment sets are loaded

df_train = pd.read_csv('dataset/train.csv')

total = 0
for i in range(1,10):
    globals()['df_%s' % i] = pd.read_csv(f'dataset/experiment_0{i}.csv')
    total = total + globals()['df_%s' % i].shape[0]

for i in range(10,19):
    globals()['df_%s' % i] = pd.read_csv(f'dataset/experiment_{i}.csv')
    total = total + globals()['df_%s' % i].shape[0]

for i in range(1,19):
    print(f"df_{i} shape: {globals()['df_%s' % i].shape}")

