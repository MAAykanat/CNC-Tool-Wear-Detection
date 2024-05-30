# Table of Contents

   * [About Repo](#about-repo)
   * [Dataset](#dataset)
   * [Requirements](#requirements)
   * [Code Explanation](#code-explanation)
   * [Methods](#methods)
   * [Implemented Machine Learning Algorithms](#implemented-machine-learning-algorithms)
   * [Results](#results)
   * [Discussion](#discussion)
   * [Referance](#referance)
   * [Authors](#authors)

# About Repo

    Hands-on implementation and classification on tool wear of CNC milling machine. 

# Dataset

Link of dataset:

   
    https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill
   

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

# Results
Drop List:
- MACHINING_PROCESS,
- Z1_CURRENTFEEDBACK,
- Z1_DCBUSVOLTAGE,
- Z1_OUTPUTCURRENT,
- Z1_OUTPUTVOLTAGE,
- S1_COMMANDACCELERATION,
- S1_SYSTEMINERTIA,
- M1_CURRENT_PROGRAM_NUMBER,
- M1_SEQUENCE_NUMBER,
- M1_CURRENT_FEEDRATE,
- EXP_NO,
- Z1_COMMANDVELOCITY,
- Z1_COMMANDACCELERATION

EXP_NO definitely should dropped out, since it memorize according to experiment numbers.

---

    Without Drop List 

<table>
  <tr>
    <td></td>
    <td style="text-align:center;" colspan="2">KNN</td>
    <td style="text-align:center;" colspan="2">Decision Tree</td>
    <td style="text-align:center;" colspan="2">Random Forest</td>
    <td style="text-align:center;" colspan="2">XGBoost</td>
    <td style="text-align:center;" colspan="2">LightGBM</td>
  </tr>
  <tr>
    <td><b>How to Splitted</b></td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
  </tr>
  <tr>
    <td>Base Model</td>
    <td>0.9125</td>
    <td><b>0.9147</b></td>
    <td>0.9865</td>
    <td><b>0.9884</b></td>
    <td>0.9939</td>
    <td><b>0.9943</b></td>
    <td>0.9958</td>
    <td><b>0.996</b></td>
    <td>0.9956</td>
    <td><b>0.9959</b></td>
  </tr>
  <tr>
    <td>Hyperparameter</td>
    <td>0.9249</td>
    <td><b>0.928</b></td>
    <td>0.9867</td>
    <td><b>0.9883</b></td>
    <td>0.9922</td>
    <td><b>0.9932</b></td>
    <td>0.9963</td>
    <td><b>0.9968</b></td>
    <td>0.9974</td>
    <td><b>0.9978</b></td>
  </tr>
  <tr>
    <td>Test</td>
    <td>0.9278</td>
    <td><b>0.9306</b></td>
    <td>0.9881</td>
    <td>0.9866</td>
    <td><b>0.9949</b></td>
    <td>0.9921</td>
    <td><b>0.997</b></td>
    <td>0.9962</td>
    <td><b>0.9986</b></td>
    <td>0.9972</td>
  </tr>
</table>

    With Drop List 

<table>
  <tr>
    <td></td>
    <td style="text-align:center;" colspan="2">KNN</td>
    <td style="text-align:center;" colspan="2">Decision Tree</td>
    <td style="text-align:center;" colspan="2">Random Forest</td>
    <td style="text-align:center;" colspan="2">XGBoost</td>
    <td style="text-align:center;" colspan="2">LightGBM</td>
  </tr>
  <tr>
    <td><b>How to Splitted</b></td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
    <td>TVT</td>
    <td>Stratify</td>
  </tr>
  <tr>
    <td>Base Model</td>
    <td><b>0.8718</b></td>
    <td>0.8687</td>
    <td>0.987</td>
    <td><b>0.9895</b></td>
    <td>0.9887</td>
    <td><b>0.9899</b></td>
    <td>0.9945</td>
    <td><b>0.9951</b></td>
    <td><b>0.9951</b></td>
    <td>0.995</td>
  </tr>
  <tr>
    <td>Hyperparameter</td>
    <td><b>0.8722</b></td>
    <td>0.8716</td>
    <td><b>0.9905</b></td>
    <td>0.9898</td>
    <td>0.9905</td>
    <td><b>0.9906</b></td>
    <td>0.9951</td>
    <td><b>0.9952</b></td>
    <td>0.997</td>
    <td><b>0.9973</b></td>
  </tr>
  <tr>
    <td>Test</td>
    <td><b>0.8778</b></td>
    <td>0.8764</td>
    <td>0.9885</td>
    <td><b>0.9893</b></td>
    <td>0.9931</td>
    <td><b>0.9891</b></td>
    <td>0.9974</td>
    <td>0.9962</td>
    <td><b>0.9984</b></td>
    <td>0.9974</td>
  </tr>
</table>

    Classification Report - Without Drop List

<table>
  <tr>
    <td></td>
    <td style="text-align:center;" colspan="1">KNN</td>
    <td style="text-align:center;" colspan="1">Decision Tree</td>
    <td style="text-align:center;" colspan="1">Random Forest</td>
    <td style="text-align:center;" colspan="1">XGBoost</td>
    <td style="text-align:center;" colspan="1">LightGBM</td>
  </tr>
  <tr>
    <td>Base Model</td>
    <td style="text-align:center;">0.9147</td>
    <td style="text-align:center;">0.9884</td>
    <td style="text-align:center;">0.9943</td>
    <td style="text-align:center;"><b>0.996<</b></td>
    <td style="text-align:center;">0.9959</td>
  </tr>
  <tr>
    <td>Hyperparameter</td>
    <td style="text-align:center;">0.9242</td>
    <td style="text-align:center;">0.9883</td>
    <td style="text-align:center;">0.9932</td>
    <td style="text-align:center;">0.9968</td>
    <td style="text-align:center;"><b>0.9978<</b></td>
  </tr>
  <tr>
    <td >Test</td>
    <td style="text-align:center;">0.9306</td>
    <td style="text-align:center;">0.9866</td>
    <td style="text-align:center;">0.9921</td>
    <td style="text-align:center;">0.9962</td>
    <td style="text-align:center;"><b>0.9972<</b></td>
  </tr>
</table>

    Classification Report - With Drop List

<table>
  <tr>
    <td></td>
    <td style="text-align:center;" colspan="1">KNN</td>
    <td style="text-align:center;" colspan="1">Decision Tree</td>
    <td style="text-align:center;" colspan="1">Random Forest</td>
    <td style="text-align:center;" colspan="1">XGBoost</td>
    <td style="text-align:center;" colspan="1">LightGBM</td>
  </tr>
  <tr>
    <td>Base Model</td>
    <td style="text-align:center;">0.8687</td>
    <td style="text-align:center;">0.9895</td>
    <td style="text-align:center;">0.9899</td>
    <td style="text-align:center;"><b>0.9951<</b></td>
    <td style="text-align:center;">0.995</td>
  </tr>
  <tr>
    <td>Hyperparameter</td>
    <td style="text-align:center;">0.8716</td>
    <td style="text-align:center;">0.9898</td>
    <td style="text-align:center;">0.9906</td>
    <td style="text-align:center;">0.9952</td>
    <td style="text-align:center;"><b>0.9973<</b></td>
  </tr>
  <tr>
    <td >Test</td>
    <td style="text-align:center;">0.8764</td>
    <td style="text-align:center;">0.9893</td>
    <td style="text-align:center;">0.9891</td>
    <td style="text-align:center;">0.9962</td>
    <td style="text-align:center;"><b>0.9974<</b></td>
  </tr>
</table>

# Discussion

# Referance

# Authors
Muhammet Ali Aykanat
