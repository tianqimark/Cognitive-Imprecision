#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 23:08:37 2020

@author: hutianqi
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# np.random.seed(123)

# start_time = time.time()

###Load data
data = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/trials.csv')

ID_table = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/ID_table.csv')

### Add few columns from the ID_table
ID_table.index = ID_table['subject']
data.index = data['subject']

data['reflection'] = ID_table['reflection']
data['BNT'] = ID_table['BNT']
data['decmode'] = ID_table['decmode']

### add more variables: value difference; valus sumation
data['value_diff'] = abs(data['WTP'] - data['certainty'])
data['value_sum'] = data['WTP'] + data['certainty']

# =============================================================================
# Preprocessing data - quality control based on RT
# =============================================================================

"""Taking out invalid trials based on the RT"""

# In Khaws study, a decision maker has maximumly 10 seconds to make a decision, 
# thus we use 10s as an indicator for disruption in decision making
disrupt = 10

# from smith 2018:
# trials with very long (more than two standard deviations above the log-transformed mean) 
# or very short (􏰃300 ms) RTs were removed from analysis.
# However, we may set the floor value to 1 due to the nature of our experiment.
floor = 1

data = data[(data['RT'] > floor) & (data['RT'] < disrupt)]

# determine the effor level based on RT 
# (how many trials have RT below the floor value set above, or above 10s)
effort = data['subject'].value_counts()
effort = pd.DataFrame(effort)
effort['count'] = effort['subject']
effort['subject'] = effort.index

"""drop subjects with low effort in the following analysis"""
# when the disrupt = 10s and floor = 1s, the last 7 subjects have 
# more than a third of trials being invalid, indicating low effort.
gone = effort['subject'][-7:]
data.drop(data.index[data['subject'].isin(gone)], inplace = True)
ID_table.drop(ID_table.index[ID_table['subject'].isin(gone)], inplace = True)

# Further excluding large outliers according to Smith
ceiling = np.mean(np.log(data['RT'])) + np.std(np.log(data['RT']))*2
data = data[np.log(data['RT']) < ceiling]

# drop rows where a correct response is non-existant due to WTP = certainty
data.dropna(inplace = True)

# # drop other rows
# data = data[['treatment', 'RT', 'certainty', 'WTP', 'correct']]

# Select one treatment
treat_A = data[data['treatment'] == 'A']
treat_E = data[data['treatment'] == 'E']

# treat_A = treat_A.drop(['treatment'], axis=1)
# treat_E = treat_E.drop(['treatment'], axis=1)

treat_A = treat_A.reset_index(drop = True)
treat_E = treat_E.reset_index(drop = True)

# np.isnan(treat_A['RT'])

# =============================================================================
# A glance of data
# =============================================================================

# RT mean
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import pingouin as pg

print(treat_A['RT'].mean())
print(treat_E['RT'].mean())

ttest_ind(treat_A['RT'], treat_E['RT'])

# select lottery frequency
print(treat_A['slc_lot'].sum() / treat_A.shape[0])
print(treat_E['slc_lot'].sum() / treat_E.shape[0])

# Correct frequency
print(treat_A['correct'].sum() / treat_A.shape[0])
print(treat_E['correct'].sum() / treat_E.shape[0])

# correlation matrix for individual differences
ID_A = ID_table[ID_table['treatment'] == 'A']
ID_E = ID_table[ID_table['treatment'] == 'E']

# Full dataset
ID_corr = ID_table.iloc[:, 2:].corr()
ID_corr_pv = ID_table.iloc[:, 2:].rcorr()

# Treatment A
ID_corr_A = ID_A.iloc[:, 2:].corr()
ID_corr_pv_A = ID_A.iloc[:, 2:].rcorr()

# Teratment E
ID_corr_E = ID_E.iloc[:, 2:].corr()
ID_corr_pv_E = ID_E.iloc[:, 2:].rcorr()

### reflection correlation significance with accuracy
pearsonr(ID_table['reflection'], ID_table['accuracy'])
pearsonr(ID_A['reflection'], ID_A['accuracy'])
pearsonr(ID_E['reflection'], ID_E['accuracy'])

# correlation bettwen the value diff and sum
pearsonr(data['value_diff'], data['value_sum'] )

# Decision mode
decmode_A = ID_table[ID_table['treatment'] == 'A']['decmode']
decmode_E = ID_table[ID_table['treatment'] == 'E']['decmode']

print(decmode_A.mean())
print(decmode_E.mean())

ttest_ind(decmode_A, decmode_E)

ID_table.groupby('decmode').count()
# ID_table['decmode'].value_counts()


# =============================================================================
# Regression analysis: RT
# =============================================================================

import statsmodels.api as sm
from patsy import dmatrices

# full dataset
y, X = dmatrices('RT ~ \
                 treatment + value_diff + value_sum + reflection + decmode', data=data, return_type='dataframe')

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model

# treat_A
y, X = dmatrices('RT ~ \
                  value_diff + value_sum + reflection + decmode', data=treat_A, return_type='dataframe')

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model

# treat_E
y, X = dmatrices('RT ~ \
                  value_diff + value_sum + reflection + decmode', data=treat_E, return_type='dataframe')

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model




# =============================================================================
# Regression analysis: correct
# =============================================================================





















