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
    # print(f"df_{i} columns: {globals()['df_%s' % i].isnull().sum()}") # No missing values

########################
# Missing Value Handle #
########################

print(df_train.isnull().sum()) # There are 4 missing values in passed_visual_inspection column

# Fill missing values with "no"

df_train['passed_visual_inspection'] = df_train['passed_visual_inspection'].fillna('no')

# Check the missing values again
print(df_train.isnull().sum()) # There are no missing values in the dataset

##########################
# Combine all dataframes #
##########################

frames= []
for i in range(1,19):
    globals()['df_%s' % i]["exp_no"]=i
    globals()['df_%s' % i]["feedrate"] = df_train["feedrate"][i-1]
    globals()['df_%s' % i]["clamp_pressure"] = df_train["clamp_pressure"][i-1]
    globals()['df_%s' % i]["target"]="worn" if df_train["tool_condition"][i-1] == "worn" else "unworn"
    frames.append(globals()['df_%s' % i])

df = pd.concat(frames, axis=0, ignore_index=True)

print(df.head())
print(df.shape)

# Save the dataset

df.to_csv('dataset/combined.csv', index=False)

############################
####### END OF FILE ########
############################
