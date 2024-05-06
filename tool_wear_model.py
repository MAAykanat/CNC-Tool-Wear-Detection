import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from shutil import get_terminal_size

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Show all the decimals
pd.set_option('display.width', get_terminal_size()[0]) # Get bigger terminal display width

# Load the data

df = pd.read_csv('dataset/combined_cleaned.csv')

# Display the first 5 rows of the data
print(df.head())