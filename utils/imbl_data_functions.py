import os
import numpy as np
import pdb
import random
import pandas as pd
import itertools
from datetime import datetime
import pickle
import time

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def remove_existing_test(pos_to_negs_ratio):
    # pdb.set_trace()
    if os.path.exists('outputs/test_medications_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv'):
        os.remove('outputs/test_medications_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv')
    if os.path.exists('outputs/test_diagnoses_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv'):
        os.remove('outputs/test_diagnoses_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv')
    if os.path.exists('outputs/test_procedures_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv'):
        os.remove('outputs/test_procedures_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv')
    if os.path.exists('outputs/test_labels_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv'):
        os.remove('outputs/test_labels_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv')
    if os.path.exists('outputs/test_demographics_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv'):
        os.remove('outputs/test_demographics_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv')  

def blind_data(line_meds_splitted
            , line_diags_splitted
            , line_procs_splitted
            , line_demogs_splitted
            , prediction_win_size
            , enrolid
            , date_idx = 5
            , label_idx = 6):
    # pdb.set_trace()
    line_meds_blinded = [enrolid]
    line_diags_blinded = [enrolid]
    line_procs_blinded = [enrolid]

    line_meds_splitted = [list(y) for x, y in itertools.groupby(line_meds_splitted[1:], lambda z: z == 'EOV') if not x]            
    line_diags_splitted = [list(y) for x, y in itertools.groupby(line_diags_splitted[1:], lambda z: z == 'EOV') if not x]            
    line_procs_splitted = [list(y) for x, y in itertools.groupby(line_procs_splitted[1:], lambda z: z == 'EOV') if not x]                

    for i in range(len(line_meds_splitted)):
        current_date = int(line_meds_splitted[i][0]) 
        if 'DIAGNOSES_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['DIAGNOSES_DATE'])
        elif 'LAST_RECORD_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['LAST_RECORD_DATE'])
        else:
            pdb.set_trace()
            print('Warning: demographics table includes unknown column.')    
        # if line_meds_splitted[i][0] 
        diff_times = diff_month(datetime(diag_or_last_data//100,diag_or_last_data%100, 1 ), datetime(current_date//100,current_date%100, 1))
        if diff_times > prediction_win_size:
            line_meds_blinded.extend(line_meds_splitted[i])
            line_meds_blinded.extend(['EOV'])
    #  diags        
    for i in range(len(line_diags_splitted)):
        current_date = int(line_diags_splitted[i][0]) 
        if 'DIAGNOSES_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['DIAGNOSES_DATE'])
        elif 'LAST_RECORD_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['LAST_RECORD_DATE'])
        else:
            pdb.set_trace()
            print('Warning: demographics table includes unknown column.')    
        diff_times = diff_month(datetime(diag_or_last_data//100,diag_or_last_data%100, 1 ), datetime(current_date//100,current_date%100, 1))
        if diff_times > prediction_win_size:
            line_diags_blinded.extend(line_diags_splitted[i])
            line_diags_blinded.extend(['EOV'])

    # procs        
    for i in range(len(line_procs_splitted)):
        current_date = int(line_procs_splitted[i][0]) 
        if 'DIAGNOSES_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['DIAGNOSES_DATE'])
        elif 'LAST_RECORD_DATE' in line_demogs_splitted.columns:
            diag_or_last_data = int(line_demogs_splitted['LAST_RECORD_DATE'])
        else:
            pdb.set_trace()
            print('Warning: demographics table includes unknown column.')    
        diff_times = diff_month(datetime(diag_or_last_data//100,diag_or_last_data%100, 1 ), datetime(current_date//100,current_date%100, 1))
        if diff_times > prediction_win_size:
            line_procs_blinded.extend(line_procs_splitted[i])
            line_procs_blinded.extend(['EOV'])

    return line_meds_blinded, line_diags_blinded, line_procs_blinded

def create_test_imbalanced(meds_oud_yes_path
                                   , diags_oud_yes_path
                                   , procs_oud_yes_path
                                   , demogs_oud_yes_path
                                   , meds_oud_no_path
                                   , diags_oud_no_path
                                   , procs_oud_no_path
                                   , demogs_oud_no_path 
                                   , sex1_matching_map_path
                                   , sex2_matching_map_path
                                   , balanced_test_filename
                                   , pos_to_negs_ratio
                                   , prediction_win_size
                                   , display_step):
    # pdb.set_trace()
    oud_yes_demographics = pd.read_csv(demogs_oud_yes_path)
    oud_no_demographics = pd.read_csv(demogs_oud_no_path)
    sex1_matching_map = pd.read_csv(sex1_matching_map_path)
    sex1_matching_map_filtered = sex1_matching_map.iloc[:,1:pos_to_negs_ratio+1]
    sex2_matching_map = pd.read_csv(sex2_matching_map_path)
    sex2_matching_map_filtered = sex2_matching_map.iloc[:,1:pos_to_negs_ratio+1]
    balanced_test_set = pd.read_csv(balanced_test_filename)

    remove_existing_test(pos_to_negs_ratio)
    enrolid_ind = 0
    
    # Positive samples   
    with open(meds_oud_yes_path) as medications_oud_yes_file, open(diags_oud_yes_path) as diagnoses_oud_yes_file, open(procs_oud_yes_path) as procedures_oud_yes_file, open(demogs_oud_yes_path) as demographics_oud_yes_file, open('outputs/test_medications_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_meds_file, open('outputs/test_diagnoses_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_diags_file, open('outputs/test_procedures_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_procs_file, open('outputs/test_demographics_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_demogs_file, open('outputs/test_labels_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv','a') as test_labels_file:
        demogs_header = next(demographics_oud_yes_file)
        # pdb.set_trace()
        test_demogs_file.write(demogs_header.replace('\n','').replace("'","") + '_OR_last_record_Date'+', Label')
        test_demogs_file.write('\n')

        line_counter = 0 
        for line_meds in medications_oud_yes_file:
            if line_counter % display_step == 0:
                print('Finished analyzing {} oud_yes patients data'.format(line_counter))
            line_counter +=1
            # print('NOTE: YOU ARE USING A TOY DATA.................')
            # if line_counter>100:
            #     break
            line_meds_splitted = line_meds.split(',')
            line_meds_splitted = [i.replace("'","") for i in line_meds_splitted]

            line_diags = diagnoses_oud_yes_file.readline()   
            line_diags_splitted = line_diags.split(',')
            line_diags_splitted = [i.replace("'","") for i in line_diags_splitted]

            line_procs = procedures_oud_yes_file.readline()   
            line_procs_splitted=line_procs.split(',')
            line_procs_splitted = [i.replace("'","") for i in line_procs_splitted]
            
            # Check if all the streams belong to the same patient
            if not(float(line_meds_splitted[enrolid_ind].replace("'",'')) == float(line_diags_splitted[enrolid_ind].replace("'",'')) == float(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")

            current_enrolid = float(line_meds_splitted[enrolid_ind].replace("'",''))            
            if current_enrolid in balanced_test_set['ENROLID']:
                continue

            current_demographics = oud_yes_demographics[oud_yes_demographics['ENROLID'] == current_enrolid]
            if current_demographics.empty == True:
                continue
            if len(current_demographics) > 1:
                print('Warning: there are more than 1 record in the demographics')
                pdb.set_trace()            

            # Blind the data before writing in into to train, validation and test
            line_meds_blinded, line_diags_blinded, line_procs_blinded = blind_data(line_meds_splitted
                                                                                    , line_diags_splitted
                                                                                    , line_procs_splitted
                                                                                    , current_demographics
                                                                                    , prediction_win_size
                                                                                    , current_enrolid)    


            if line_diags_blinded[-1] == "'EOV\\n'\n'":
                pdb.set_trace()

            test_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
            test_meds_file.write('\n')
            test_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
            test_diags_file.write('\n')  
            test_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
            test_procs_file.write('\n')  
            test_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
            test_demogs_file.write(', 1')
            test_demogs_file.write('\n')
            test_labels_file.write(str(current_enrolid))
            test_labels_file.write(',')
            test_labels_file.write('1, 0')
            test_labels_file.write('\n')

    # pdb.set_trace()
    # Negative samples   
    with open(meds_oud_no_path) as medications_oud_no_file, open(diags_oud_no_path) as diagnoses_oud_no_file, open(procs_oud_no_path) as procedures_oud_no_file, open(demogs_oud_no_path) as demographics_oud_no_file, open('outputs/test_medications_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_meds_file, open('outputs/test_diagnoses_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_diags_file, open('outputs/test_procedures_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_procs_file, open('outputs/test_demographics_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv', 'a') as test_demogs_file, open('outputs/test_labels_neg_to_pos_'+str(pos_to_negs_ratio)+'.csv','a') as test_labels_file:
        demogs_oud_no_header = next(demographics_oud_no_file)

        line_counter = 0
        for line_meds in medications_oud_no_file:
            if line_counter % display_step == 0:
                print('Finished analyzing {} oud_no patients data'.format(line_counter))
            line_counter +=1  
            # print('NOTE: YOU ARE USING A TOY DATA.................')
            # if line_counter>100:
            #     break                      
            line_meds_splitted = line_meds.split(',')
            line_meds_splitted = [i.replace("'","") for i in line_meds_splitted]

            line_diags = diagnoses_oud_no_file.readline()   
            line_diags_splitted = line_diags.split(',')
            line_diags_splitted = [i.replace("'","") for i in line_diags_splitted]

            line_procs = procedures_oud_no_file.readline()   
            line_procs_splitted=line_procs.split(',')
            line_procs_splitted = [i.replace("'","") for i in line_procs_splitted]

            if not(float(line_meds_splitted[enrolid_ind].replace("'",'')) == float(line_diags_splitted[enrolid_ind].replace("'",'')) == float(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")

            if not(float(line_meds_splitted[enrolid_ind].replace("'",'')) == float(line_diags_splitted[enrolid_ind].replace("'",'')) == float(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")
            # pdb.set_trace()
            current_enrolid = float(line_meds_splitted[enrolid_ind].replace("'",''))
            
            if (current_enrolid not in sex1_matching_map_filtered.values) and (current_enrolid not in sex2_matching_map_filtered.values):
                continue
            current_demographics = oud_no_demographics[oud_no_demographics['ENROLID'] == current_enrolid]

            if len(current_demographics) > 1:
                print('Warning: there are more than 1 record in the demographics')
                pdb.set_trace()

            line_meds_blinded, line_diags_blinded, line_procs_blinded = blind_data(line_meds_splitted
                                                                                    , line_diags_splitted
                                                                                    , line_procs_splitted
                                                                                    , current_demographics
                                                                                    , prediction_win_size
                                                                                    , current_enrolid)    


            test_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
            test_meds_file.write('\n')
            test_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
            test_diags_file.write('\n')  
            test_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
            test_procs_file.write('\n')  
            test_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
            test_demogs_file.write(', 0')
            test_demogs_file.write('\n')
            test_labels_file.write(str(current_enrolid))
            test_labels_file.write(',')
            test_labels_file.write('0, 1')
            test_labels_file.write('\n')



