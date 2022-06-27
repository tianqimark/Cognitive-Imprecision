#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 00:13:01 2020

@author: hutianqi
"""
# =============================================================================
# This fist part was to join different batches, no longer needed now.
# =============================================================================

# import numpy as np
# import pandas as pd

# # import raw dataset
# datar12 = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/Batch1n2.csv')
# datar3 = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/Batch3.csv')
# datar4 = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/Batch4.csv')

# frames = [datar12, datar3, datar4]

# datar = pd.concat(frames, ignore_index=True)

# # datar.to_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/current123.csv', index=False)

# # drop the test sessions
# # In batch 1 and 2
# datar.drop(datar.index[datar['session.code'] == '6p943jar'], inplace = True)
# datar.drop(datar.index[datar['session.code'] == '0jy7owjo'], inplace = True)
# datar.drop(datar.index[datar['session.code'] == '9q4lgmhn'], inplace = True)
# # In batch 3
# datar.drop(datar.index[datar['session.code'] == 'psgdzgio'], inplace = True)
# # In batch 4
# datar.drop(datar.index[datar['session.code'] == 'u5vh325b'], inplace = True)
# datar.drop(datar.index[datar['session.code'] == 'uqu1sde5'], inplace = True)


# # drop the invalid participants
# # from the Batch2, this person completed the experiment anyway so we decided to include him.
# # datar.drop(datar.index[datar['REItest.1.player.prolific_code'] == '5eebbbe3b069b03883fb65e0'], inplace = True)

# # Or, select certain sessions 
# # datar = datar[datar['session.code'] == 'brtqjddt']

# # drop imcomplete attempts
# datar.drop(datar.index[np.isnan(datar['cognitivenoise.216.player.pay_pound'])], inplace = True)
# datar = datar.reset_index(drop=True)

# datar.to_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/RawData.csv', index=False)

# =============================================================================
# Trial table from the Part 3
# =============================================================================

import numpy as np
import pandas as pd

# Now we can directly read the sorted raw dataset
datar = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/RawData.csv')

# Assign each subject a numerical code
datar = datar.reset_index(drop=True)
datar['subject'] = datar.index + 1

# build a trial table
column_names = ['subject', 'trial', 'treatment', 'lot_code', 
                'risk', 'reward', 'WTP', 'certainty', 
                'RT', 'choice', 'slc_lot',
                'display', 'correct']
index_length = 216 * datar.shape[0]

trials = pd.DataFrame(index = range(index_length), columns = column_names)

# assign subject and trial numbers
for i in range(trials.shape[0]):    
    counter1 = i // 216 
    trials['subject'][i] = counter1 + 1
    
    counter2 = i % 216
    trials['trial'][i] = counter2 + 1
    

# extract info from the raw dataset
datar.index = datar['subject']

for i in range(trials.shape[0]):
    subject = trials['subject'][i]
    trial = trials['trial'][i]
    
    treatment = 'cognitivenoise.' + str(trial) + '.player.treatment'
    trials['treatment'][i] = datar.loc[subject, treatment]

    risk = 'cognitivenoise.' + str(trial) + '.player.risk'
    trials['risk'][i] = datar.loc[subject, risk]

    reward = 'cognitivenoise.' + str(trial) + '.player.reward'
    trials['reward'][i] = datar.loc[subject, reward]

    certainty = 'cognitivenoise.' + str(trial) + '.player.certainty'
    trials['certainty'][i] = datar.loc[subject, certainty]

    RT = 'cognitivenoise.' + str(trial) + '.player.jsdectime'
    trials['RT'][i] = datar.loc[subject, RT]

    choice = 'cognitivenoise.' + str(trial) + '.player.choice'
    trials['choice'][i] = datar.loc[subject, choice]

    slc_lot = 'cognitivenoise.' + str(trial) + '.player.lottery'
    trials['slc_lot'][i] = datar.loc[subject, slc_lot]

    display = 'cognitivenoise.' + str(trial) + '.player.display'
    trials['display'][i] = datar.loc[subject, display]


# assign numerical code to lotteries
for i in range(trials.shape[0]):
    if trials['reward'][i] < 10 and trials['risk'][i] < 50:
        trials['lot_code'][i] = 1
    
    if trials['reward'][i] < 10 and trials['risk'][i] > 50 and trials['risk'][i] < 70:
        trials['lot_code'][i] = 2

    if trials['reward'][i] < 10 and trials['risk'][i] > 70:
        trials['lot_code'][i] = 3
   
    if trials['reward'][i] > 10 and trials['reward'][i] < 14 and trials['risk'][i] < 50:
        trials['lot_code'][i] = 4
    
    if trials['reward'][i] > 10 and trials['reward'][i] < 14 and trials['risk'][i] > 50 and trials['risk'][i] < 70:
        trials['lot_code'][i] = 5

    if trials['reward'][i] > 10 and trials['reward'][i] < 14 and trials['risk'][i] > 70:
        trials['lot_code'][i] = 6

    if trials['reward'][i] > 14 and trials['reward'][i] < 20 and trials['risk'][i] < 50:
        trials['lot_code'][i] = 7
    
    if trials['reward'][i] > 14 and trials['reward'][i] < 20 and trials['risk'][i] > 50 and trials['risk'][i] < 70:
        trials['lot_code'][i] = 8

    if trials['reward'][i] > 14 and trials['reward'][i] < 20 and trials['risk'][i] > 70:
        trials['lot_code'][i] = 9
         
    if trials['reward'][i] > 20 and trials['risk'][i] < 50:
        trials['lot_code'][i] = 10
    
    if trials['reward'][i] > 20 and trials['risk'][i] > 50 and trials['risk'][i] < 70:
        trials['lot_code'][i] = 11

    if trials['reward'][i] > 20 and trials['risk'][i] > 70:
        trials['lot_code'][i] = 12
         
    
# =============================================================================
# WTP table from the Part 2
# =============================================================================

# build a WTP table
column_names = ['subject', 
                'lot_1', 'lot_2', 'lot_3', 'lot_4', 
                'lot_5', 'lot_6', 'lot_7', 'lot_8', 
                'lot_9', 'lot_10', 'lot_11', 'lot_12']
index_length = datar.shape[0]

WTP_table = pd.DataFrame(index = range(1, index_length+ 1), columns = column_names)

WTP_table['subject'] = WTP_table.index

# extract info from the raw dataset
datar.index = datar['subject']

for i in range(WTP_table.shape[0]):
    subject = i + 1
    
    for j in range(12):
        lot_num = j + 1
        lot_in_table = 'lot_' + str(lot_num)
        lot_in_datar = 'BDMauction.' + str(lot_num) + '.player.WTP'
                
        WTP_table.loc[subject, lot_in_table] = datar.loc[subject, lot_in_datar]


WTP_table.to_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/WTP_table.csv', index=False)

# =============================================================================
# Create a trial table that is ready for DDM fitting
# =============================================================================

WTP_table.index = WTP_table['subject']

for i in range(trials.shape[0]):
    subject = trials['subject'][i]
    lot_code = trials['lot_code'][i]
    
    WTP = 'lot_' + str(lot_code)
    trials['WTP'][i] = WTP_table.loc[subject, WTP]
    
    
for i in range(trials.shape[0]):
    
    WTP = trials['WTP'][i]
    certainty = trials['certainty'][i]
    lottery = trials['slc_lot'][i]
    
    if WTP == certainty:
        trials['correct'][i] = np.nan
        # in this case the trial need to be discarded using dropna etc
    elif WTP > certainty and lottery == 1:
        trials['correct'][i] = 1
    elif WTP < certainty and lottery == 0:
        trials['correct'][i] = 1
    else:
        trials['correct'][i] = 0


trials.to_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/trials.csv', index=False)


# =============================================================================
# Create a table for the individual differences (ID)
# =============================================================================

# build a ID table
# Columns: Reflectiveness Score (CRT), Intuitiveness Score (CRT), 
# Berlin Numeracy Test, Need for Cognition, Faith in Intuition.
column_names = ['subject', 'treatment',
                'reflection', 'intuition', 'BNT', 'NFC', 'FI', 'decmode',
                'accuracy', 'lottery_freq']
index_length = datar.shape[0]

ID_table = pd.DataFrame(index = range(1, index_length+ 1), columns = column_names)

ID_table['subject'] = ID_table.index

# extract info from the raw dataset
datar.index = datar['subject']

for i in range(ID_table.shape[0]):
    subject = i + 1
    
    ID_table.loc[subject, 'treatment'] = datar.loc[subject, 'cognitivenoise.216.player.treatment']

    ID_table.loc[subject, 'reflection'] = datar.loc[subject, 'REItest.1.player.reflectiveness_score']
    
    ID_table.loc[subject, 'intuition'] = datar.loc[subject, 'REItest.1.player.intuitiveness_score']
    
    ID_table.loc[subject, 'BNT'] = datar.loc[subject, 'REItest.1.player.bnt_score']
    
    ID_table.loc[subject, 'NFC'] = datar.loc[subject, 'REItest.1.player.nfcscore']
    
    ID_table.loc[subject, 'FI'] = datar.loc[subject, 'REItest.1.player.fiscore']
    
    ID_table.loc[subject, 'decmode'] = datar.loc[subject, 'cognitivenoise.216.player.decmode']
    
    dum = trials.dropna() # drop rows with WTP == certianty, which means a correct response is non-existent.
    ID_table.loc[subject, 'accuracy'] = dum[dum['subject'] == subject]['correct'].sum() / len(dum[dum['subject'] == subject])      
    
    ID_table.loc[subject, 'lottery_freq'] = trials[trials['subject'] == subject]['slc_lot'].sum() / 216

   
ID_table.to_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z1 Dataset/ID_table.csv', index=False)




# =============================================================================
# Trestment count
# =============================================================================

A = 0
E = 0
for i in datar['cognitivenoise.4.player.treatment']:
    if i == 'A':
        A += 1
    else:
        E += 1
    









