import pdb
import argparse
import sys
import os
import pandas as pd

# pdb.set_trace()

# All test data file names
stationary_test_filename = 'outputs/test_stationary_normalized_features_filtered.csv'

multihot_test_meds_filename = 'outputs/test_medications_multihot.csv'
multihot_test_diags_filename = 'outputs/test_diagnoses_multihot.csv'
multihot_test_procs_filename = 'outputs/test_procedures_multihot.csv'
multihot_test_demogs_filename = 'outputs/test_demographics_multihot.csv'
multihot_test_metadata_filename = 'outputs/test_demographics_shuffled.csv'

singlestream_represented_test_filename='outputs/test_meds_diags_procs_demogs_represented.csv'

mupod_represented_test_filename_meds='outputs/test_meds_represented.csv'
mupod_represented_test_filename_diags='outputs/test_diags_represented.csv'
mupod_represented_test_filename_procs='outputs/test_procs_represented.csv'
mupod_represented_test_filename_demogs='outputs/test_demographics_multihot.csv'


print('Reading the stationary test data ....')
stationary_test = pd.read_csv(stationary_test_filename)
pos_enrolids = stationary_test[stationary_test['Label']==1].ENROLID
neg_enrolids = stationary_test[stationary_test['Label']==0].ENROLID
pos_1_to_2 = pos_enrolids.sample(frac=0.5)
pos_1_to_5 = pos_enrolids.sample(frac=0.2)
pos_1_to_10 = pos_enrolids.sample(frac=0.1)

# creating imb test sets for the stationary data
print('Creating 1-to-2, 1-to-5 and 1-to-10 imbalanced test sets for the stationary models ....')
stationary_test[stationary_test.ENROLID.isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(stationary_test_filename[:-4]+'_1_to_2.csv', index=False)    
stationary_test[stationary_test.ENROLID.isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(stationary_test_filename[:-4]+'_1_to_5.csv', index=False)    
stationary_test[stationary_test.ENROLID.isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(stationary_test_filename[:-4]+'_1_to_10.csv', index=False)    


# Creating imb test sets for multi hot data
print('Reading the test sets with multi-hot format (for single stream LSTM model) ....')
multihot_test_meds = pd.read_csv(multihot_test_meds_filename, skiprows=1, header=None)
multihot_test_diags = pd.read_csv(multihot_test_diags_filename, skiprows=1, header=None)
multihot_test_procs = pd.read_csv(multihot_test_procs_filename, skiprows=1, header=None)
multihot_test_demogs = pd.read_csv(multihot_test_demogs_filename, header=None)
multihot_test_metadata = pd.read_csv(multihot_test_metadata_filename)

# 1 to 2
multihot_test_meds[multihot_test_meds.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(multihot_test_meds_filename[:-4]+'_1_to_2.csv', index=False)    
multihot_test_diags[multihot_test_diags.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(multihot_test_diags_filename[:-4]+'_1_to_2.csv', index=False)    
multihot_test_procs[multihot_test_procs.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(multihot_test_procs_filename[:-4]+'_1_to_2.csv', index=False)    
multihot_test_demogs[multihot_test_demogs.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(multihot_test_demogs_filename[:-4]+'_1_to_2.csv', index=False)    
multihot_test_metadata[multihot_test_metadata.ENROLID.isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(multihot_test_metadata_filename[:-4]+'_1_to_2.csv', index=False)    

# 1 to 5
multihot_test_meds[multihot_test_meds.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(multihot_test_meds_filename[:-4]+'_1_to_5.csv', index=False)    
multihot_test_diags[multihot_test_diags.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(multihot_test_diags_filename[:-4]+'_1_to_5.csv', index=False)    
multihot_test_procs[multihot_test_procs.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(multihot_test_procs_filename[:-4]+'_1_to_5.csv', index=False)    
multihot_test_demogs[multihot_test_demogs.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(multihot_test_demogs_filename[:-4]+'_1_to_5.csv', index=False)    
multihot_test_metadata[multihot_test_metadata.ENROLID.isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(multihot_test_metadata_filename[:-4]+'_1_to_5.csv', index=False)    

# 1 to 10
multihot_test_meds[multihot_test_meds.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(multihot_test_meds_filename[:-4]+'_1_to_10.csv', index=False)    
multihot_test_diags[multihot_test_diags.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(multihot_test_diags_filename[:-4]+'_1_to_10.csv', index=False)    
multihot_test_procs[multihot_test_procs.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(multihot_test_procs_filename[:-4]+'_1_to_10.csv', index=False)    
multihot_test_demogs[multihot_test_demogs.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(multihot_test_demogs_filename[:-4]+'_1_to_10.csv', index=False)    
multihot_test_metadata[multihot_test_metadata.ENROLID.isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(multihot_test_metadata_filename[:-4]+'_1_to_10.csv', index=False)    


# Creating imb test sets for single stream
print('Reading the single stream represented tes set....')
singlestream_represented_test = pd.read_csv(singlestream_represented_test_filename, header=None)

print('Creating imbalanced test sets for the represented single stream data (for single stream transformer)....')
# 1 to 2
singlestream_represented_test[singlestream_represented_test.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(singlestream_represented_test_filename[:-4]+'_1_to_2.csv', index=False)    
# 1 to 5
singlestream_represented_test[singlestream_represented_test.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(singlestream_represented_test_filename[:-4]+'_1_to_5.csv', index=False)    
# 1 to 10
singlestream_represented_test[singlestream_represented_test.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(singlestream_represented_test_filename[:-4]+'_1_to_10.csv', index=False)    


# Creating imb test sets for MUPOD
print('Reading the MUPOD test sets ....')
mupod_represented_test_meds = pd.read_csv(mupod_represented_test_filename_meds,  header=None)
mupod_represented_test_diags = pd.read_csv(mupod_represented_test_filename_diags,  header=None)
mupod_represented_test_procs = pd.read_csv(mupod_represented_test_filename_procs,  header=None)
mupod_represented_test_demogs = pd.read_csv(mupod_represented_test_filename_demogs,  header=None)

# 1 to 2
mupod_represented_test_meds[mupod_represented_test_meds.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(mupod_represented_test_filename_meds[:-4]+'_1_to_2.csv', index=False)    
mupod_represented_test_diags[mupod_represented_test_diags.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(mupod_represented_test_filename_diags[:-4]+'_1_to_2.csv', index=False)    
mupod_represented_test_procs[mupod_represented_test_procs.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(mupod_represented_test_filename_procs[:-4]+'_1_to_2.csv', index=False)    
mupod_represented_test_demogs[mupod_represented_test_demogs.iloc[:,0].isin(pd.concat([pos_1_to_2, neg_enrolids]))].to_csv(mupod_represented_test_filename_demogs[:-4]+'_1_to_2.csv', index=False)    

# 1 to 5
mupod_represented_test_meds[mupod_represented_test_meds.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(mupod_represented_test_filename_meds[:-4]+'_1_to_5.csv', index=False)    
mupod_represented_test_diags[mupod_represented_test_diags.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(mupod_represented_test_filename_diags[:-4]+'_1_to_5.csv', index=False)    
mupod_represented_test_procs[mupod_represented_test_procs.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(mupod_represented_test_filename_procs[:-4]+'_1_to_5.csv', index=False)    
mupod_represented_test_demogs[mupod_represented_test_demogs.iloc[:,0].isin(pd.concat([pos_1_to_5, neg_enrolids]))].to_csv(mupod_represented_test_filename_demogs[:-4]+'_1_to_5.csv', index=False)    

# 1 to 10
mupod_represented_test_meds[mupod_represented_test_meds.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(mupod_represented_test_filename_meds[:-4]+'_1_to_10.csv', index=False)    
mupod_represented_test_diags[mupod_represented_test_diags.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(mupod_represented_test_filename_diags[:-4]+'_1_to_10.csv', index=False)    
mupod_represented_test_procs[mupod_represented_test_procs.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(mupod_represented_test_filename_procs[:-4]+'_1_to_10.csv', index=False)    
mupod_represented_test_demogs[mupod_represented_test_demogs.iloc[:,0].isin(pd.concat([pos_1_to_10, neg_enrolids]))].to_csv(mupod_represented_test_filename_demogs[:-4]+'_1_to_10.csv', index=False)    


