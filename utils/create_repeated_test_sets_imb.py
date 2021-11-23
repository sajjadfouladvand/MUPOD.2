import pandas as pd
import pdb
import numpy as np

# pdb.set_trace()
imb_ratio = 10

print('Stationary data ...')
test_data_stationary = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_stationary_normalized_features_filtered_1_to_'+str(imb_ratio)+'.csv')

test_data_stationary_pos = test_data_stationary[test_data_stationary['Label'] == 1]
test_data_stationary_neg = test_data_stationary[test_data_stationary['Label'] == 0]

test_data_stationary_pos = test_data_stationary_pos.sample(frac=1)
test_data_stationary_neg = test_data_stationary_neg.sample(frac=1)

test_data_stationary_pos_kfolds = np.array_split(test_data_stationary_pos, 100)  
test_data_stationary_neg_kfolds = np.array_split(test_data_stationary_neg, 100)  

fold_counter=0 
for i in range(len(test_data_stationary_pos_kfolds)):
    temp_test_fold = pd.concat([test_data_stationary_pos_kfolds[i], test_data_stationary_neg_kfolds[i]])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_stationary_normalized_features_filtered_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False)
    fold_counter +=1
    if fold_counter % 10 ==0:
        print(fold_counter)

print('Single stream data ...')
fold_counter=0 
test_data_single_stream = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_meds_diags_procs_demogs_represented_1_to_'+str(imb_ratio)+'.csv', header=None)
for i in range(len(test_data_stationary_pos_kfolds)):
    temp_test_fold_pos = test_data_single_stream[test_data_single_stream.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_single_stream[test_data_single_stream.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_meds_diags_procs_demogs_represented_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=False)
    fold_counter +=1
    if fold_counter % 10 ==0:
        print(fold_counter)

print('Multi-stream data ...')
fold_counter=0 
test_data_multi_stream_meds = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_meds_represented_1_to_'+str(imb_ratio)+'.csv', header=None)
test_data_multi_stream_diags = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_diags_represented_1_to_'+str(imb_ratio)+'.csv', header=None)
test_data_multi_stream_procs = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_procs_represented_1_to_'+str(imb_ratio)+'.csv', header=None)
test_data_multi_stream_demogs = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_demographics_multihot_1_to_'+str(imb_ratio)+'.csv', header=None)
for i in range(len(test_data_stationary_pos_kfolds)):
    temp_test_fold_pos = test_data_multi_stream_meds[test_data_multi_stream_meds.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_stream_meds[test_data_multi_stream_meds.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_meds_represented_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=False)

    temp_test_fold_pos = test_data_multi_stream_diags[test_data_multi_stream_diags.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_stream_diags[test_data_multi_stream_diags.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_diags_represented_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=False)

    temp_test_fold_pos = test_data_multi_stream_procs[test_data_multi_stream_procs.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_stream_procs[test_data_multi_stream_procs.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_procs_represented_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=False)

    temp_test_fold_pos = test_data_multi_stream_demogs[test_data_multi_stream_demogs.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_stream_demogs[test_data_multi_stream_demogs.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_demographics_multihot_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=False)
    
    fold_counter +=1
    if fold_counter % 10 ==0:
        print(fold_counter)

print('Multi-stream raw data ...')
fold_counter=0 
test_data_multi_raw_meds = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_medications_multihot_1_to_'+str(imb_ratio)+'.csv', skiprows=1, header=None)
test_data_multi_raw_diags = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_diagnoses_multihot_1_to_'+str(imb_ratio)+'.csv', skiprows=1, header=None)
test_data_multi_raw_procs = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_procedures_multihot_1_to_'+str(imb_ratio)+'.csv', skiprows=1, header=None)
test_data_multi_raw_demogs = pd.read_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_demographics_shuffled_1_to_'+str(imb_ratio)+'.csv')
for i in range(len(test_data_stationary_pos_kfolds)):
    # pdb.set_trace()
    temp_test_fold_pos = test_data_multi_raw_meds[test_data_multi_raw_meds.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_raw_meds[test_data_multi_raw_meds.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_medications_multihot_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=True)

    temp_test_fold_pos = test_data_multi_raw_diags[test_data_multi_raw_diags.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_raw_diags[test_data_multi_raw_diags.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_diagnoses_multihot_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=True)

    temp_test_fold_pos = test_data_multi_raw_procs[test_data_multi_raw_procs.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_raw_procs[test_data_multi_raw_procs.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_procedures_multihot_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header=True)

    temp_test_fold_pos = test_data_multi_raw_demogs[test_data_multi_raw_demogs.iloc[:,0].isin(test_data_stationary_pos_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold_neg = test_data_multi_raw_demogs[test_data_multi_raw_demogs.iloc[:,0].isin(test_data_stationary_neg_kfolds[i]['ENROLID'].values.tolist())]
    temp_test_fold = pd.concat([temp_test_fold_pos, temp_test_fold_neg])
    # temp_test_fold = temp_test_fold.sample(frac=1)    
    temp_test_fold.to_csv('/data/scratch/sfouladvand/RISK/comparative_study/outputs/test_demographics_shuffled_1_to_'+str(imb_ratio)+'_fold'+str(fold_counter)+'.csv', index=False, header= True)
    
    fold_counter +=1
    if fold_counter % 10 ==0:
        print(fold_counter)



