# Table of Contents

   * [About Repo](#about-repo)
   * [Dataset](#dataset)
   * [Requirements](#requirements)
   * [Code Explanation](#code-explanation)
   * [Methods](#methods)
   * [#Implemented Machine Learning Algorithms](#implementedmachine-learning-algorithms)
   * [Authors](#authors)

# About Repo

    Hands-on implementation and classification on tool wear of CNC milling machine. 

# Dataset

Link of dataset:

    ```
    https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill
    ```

Dataset created by System-level Manufacturing and Automation Research Testbed (SMART) at the University of Michigan in April 2018.

**Inputs (features)**

**No :** experiment number

**material :** wax

**feed_rate :** relative velocity of the cutting tool along the workpiece (mm/s)

**clamp_pressure :** pressure used to hold the workpiece in the vise (bar)

**Outputs (predictions)**

**tool_condition :** label for unworn and worn tools

**machining_completed :** indicator for if machining was completed without the workpiece moving out of the pneumatic vise

**passed_visual_inspection:** indicator for if the workpiece passed visual inspection, only available for experiments where machining was completed

- **18 different** experiment has been implemented.
- By train.csv, results are **labelled**.

# Requirements

Use given requirement.txt file in repo.

- Python Version :

  ```
  Python 3.10.11
  ```

- Using pip :

  ```terminal
  pip install -r requirements/requirements.txt
  ```

# Code Explanation
    3 main code to handle research
        - combine_dataset.py
        - tool_wear_detection_research.py
        - tool_wear_model.py:
- combine_dataset.py:
    
    To handle dataset, experiments and results are combined to begin research. 
    Dataset set combine by using **18 experiment (experiment_{}.csv)** and their **tool_condition** on **train.csv**

- tool_wear_detection_research.py

    **EDA (Exploratory Data Analysis)** to dataset both for experiments and train sets.
    
    **Feature Engineering** to feed and train the model. 

    Output will be use to model.

    Output: 
    - combined_cleaned.csv
    - combined_cleaned_without_droplist.csv

- tool_wear_model.py: 

    Automated code for modelling, hyperparameter optimization and evaluation results.
    
    Results can be given with/without graph as desired.

# Methods
- There is created drop list according to correlation is higher than 90%. 
    
    - By using this list 2 different dataset created
        - combined_cleaned.csv --> drop Drop_list 
        - combined_cleaned_without_droplist.csv --> not drop Drop_list

    - Model splitted into train and test sets (80-20)
        - With stratify
        - Without stratfy

- 10 Fold-Cross Validation implemented

# Implemented Machine Learning Algorithms

- KNN
- Decision Tree (CART)
- Random Forest
- XGBoost
- LightGBM

