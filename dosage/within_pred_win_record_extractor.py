import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import dateutil.relativedelta
import seaborn as sns
import itertools
import os
import collections
import math
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

# pdb.set_trace()
def remove_existing_train_val_test():
    if os.path.exists("outputs/test_medications_within_pred_wind.csv"):
        os.remove("outputs/test_medications_within_pred_wind.csv")
    if os.path.exists("outputs/test_diagnoses_within_pred_wind.csv"):
        os.remove("outputs/test_diagnoses_within_pred_wind.csv")
    if os.path.exists("outputs/test_procedures_within_pred_wind.csv"):
        os.remove("outputs/test_procedures_within_pred_wind.csv")
    if os.path.exists("outputs/test_labels_within_pred_wind.csv"):
        os.remove("outputs/test_labels_within_pred_wind.csv")
    if os.path.exists("outputs/test_demographics_within_pred_wind.csv"):
        os.remove("outputs/test_demographics_within_pred_wind.csv") 

def create_stationary_meds(line_med, distinct_tcgpid_2digit_dict, tcgpi_num_digits):
    # pdb.set_trace()
    #sys.float_info.epsilon is 2.220446049250313e-16
    epsil = 2.220446049250313e-16
    round_dig = 5
    line_med_splitted = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]            
    for i in range(len(line_med_splitted)):
        # if line_med_splitted[i][0]
        for j in range(1, len(line_med_splitted[i])): 
            if (line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]+'_tcgp_2digit') in distinct_tcgpid_2digit_dict:           
                distinct_tcgpid_2digit_dict[ line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]+'_tcgp_2digit'] += 1
            elif line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] == 'NO' or line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] =='EO':
                continue
            else:
                pdb.set_trace()   
                print('test') 
    distinct_tcgpid_2digit_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_2digit_dict.items())))    
    num_records = len(line_med_splitted)
    # if (num_records > 1):
    #     distinct_tcgpid_2digit_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_dig) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}
    # else:
    #     distinct_tcgpid_2digit_dict_sorted = {k: np.round(v , round_dig) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}        

    return distinct_tcgpid_2digit_dict_sorted

def create_stationary_diags(line_diag, icd_to_ccs_dict, ccs_distinct_dict):

    # pdb.set_trace()
    #sys.float_info.epsilon is 2.220446049250313e-16
    epsil = 2.220446049250313e-16
    round_dig = 5
    line_diag_splitted = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]            
    for i in range(len(line_diag_splitted)):
        for j in range(1, len(line_diag_splitted[i])): 
            if line_diag_splitted[i][j].replace("'","") == 'NOCODE' or line_diag_splitted[i][j].replace("'","")[:3] == 'EOV':
                # pdb.set_trace()
                continue
            elif line_diag_splitted[i][j].replace("'","") not in icd_to_ccs_dict:
                # pdb.set_trace()
                ccs_distinct_dict['-1000_ccs_diag'] +=1
            elif math.isnan(icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0]):
                ccs_distinct_dict['-1000_ccs_diag'] +=1                
            elif (str(icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0])+'_ccs_diag') in ccs_distinct_dict:
                ccs_distinct_dict[(str(icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0])+'_ccs_diag')] +=1
            else:
                pdb.set_trace()   
                pirnt('warning') 
    # pdb.set_trace()
    ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(ccs_distinct_dict.items())))

    num_records = len(line_diag_splitted)
    # if (num_records > 1 ):
    #     ccs_distinct_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_dig) for k, v in ccs_distinct_dict_sorted.items()}
    # else:        
    #     ccs_distinct_dict_sorted = {k: np.round(v, round_dig) for k, v in ccs_distinct_dict_sorted.items()}
   
    return ccs_distinct_dict_sorted

def create_stationary_procs(line_proc, proc_cd_to_ccs_dict, proc_ccs_distinct_dict):
    # pdb.set_trace()
    #sys.float_info.epsilon is 2.220446049250313e-16
    round_dig = 5
    epsil = 2.220446049250313e-16
    line_proc_splitted = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]            
    for i in range(len(line_proc_splitted)):
        for j in range(1, len(line_proc_splitted[i])): 
            # if line_proc_splitted[i][j].replace("'","") == '95904':
            #     pdb.set_trace()
            if line_proc_splitted[i][j].replace("'","") == 'NOCODE' or line_proc_splitted[i][j].replace("'","")[:3] == 'EOV':
                # pdb.set_trace()
                continue
            elif line_proc_splitted[i][j].replace("'","") not in proc_cd_to_ccs_dict:
                # pdb.set_trace()
                proc_ccs_distinct_dict['-1000_ccs_proc'] +=1

            elif math.isnan(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0]):
                proc_ccs_distinct_dict['-1000_ccs_proc'] +=1
            elif (str(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0])+'_ccs_proc') in proc_ccs_distinct_dict:
                proc_ccs_distinct_dict[(str(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0])+'_ccs_proc')] +=1
            else:
                pdb.set_trace()  
                print('warning')  
    # pdb.set_trace()
    proc_ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(proc_ccs_distinct_dict.items())))

    num_records = len(line_proc_splitted)
    # if (num_records > 1):
    #     proc_ccs_distinct_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_dig) for k, v in proc_ccs_distinct_dict_sorted.items()}
    # else:
    #     proc_ccs_distinct_dict_sorted = {k: np.round(v, round_dig) for k, v in proc_ccs_distinct_dict_sorted.items()}        
     
    return proc_ccs_distinct_dict_sorted   


def create_stationary(meds_file, diags_file, procs_file, demogs_file, dim_diags_file, dim_procs_ccs_file, distinct_tcgpid_file, tcgpi_num_digits, fold_name):
    # pdb.set_trace()
    dob_idx = 1
    sex_idx = 2
    label_idx = 6
    index_to_calc_age = 2019
    # Min date of birth in TRVNORM is 1889 
    # Max date of birth in TRVNORM is 2018

    dim_procs_ccs = pd.read_csv(dim_procs_ccs_file)
    proc_ccs_distinct = dim_procs_ccs['ccs'].dropna().unique()
    proc_ccs_distinct_dict = {}
    for i in range(len(proc_ccs_distinct)):
        proc_ccs_distinct_dict[str(proc_ccs_distinct[i])+'_ccs_proc'] = 0 
    proc_ccs_distinct_dict['-1000_ccs_proc'] = 0

    proc_cd_to_ccs = dim_procs_ccs[['proccd', 'ccs']]  
    proc_cd_to_ccs_dict =  proc_cd_to_ccs.set_index('proccd').T.to_dict('list') 
    
    # pdb.set_trace()
    dim_diags = pd.read_csv(dim_diags_file)
    ccs_distinct = dim_diags['CCS_CATGRY'].dropna().unique()
    ccs_distinct_dict = {}
    for i in range(len(ccs_distinct)):
        ccs_distinct_dict[str(ccs_distinct[i])+'_ccs_diag'] = 0 
    ccs_distinct_dict['-1000_ccs_diag'] = 0
    # pdb.set_trace()
    icd_to_ccs = dim_diags[['DIAG_CD', 'CCS_CATGRY']]  
    icd_to_ccs_dict =  icd_to_ccs.set_index('DIAG_CD').T.to_dict('list') 

    distinct_tcgpid = pd.read_csv(distinct_tcgpid_file)
    distinct_tcgpid_digits = distinct_tcgpid['TCGPI_ID'].str[0:tcgpi_num_digits].dropna().unique()
    distinct_tcgpid_digits_dict = {}
    for i in range(len(distinct_tcgpid_digits)):
        distinct_tcgpid_digits_dict[distinct_tcgpid_digits[i]+'_tcgp_2digit'] = 0 
    # pdb.set_trace()
    counter = 0
    with open(meds_file) as meds_filename, open(diags_file) as diags_filename, open(procs_file) as procs_filename, open(demogs_file) as demogs_filename, open('outputs/'+fold_name+'_stationary_within_pred_wind.csv','w') as stationary_file:
        # demogs_header = next(demogs_filename)

        distinct_tcgpid_digits_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_digits_dict.items())))
        ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(ccs_distinct_dict.items())))
        proc_ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(proc_ccs_distinct_dict.items())))
        
        stationary_file.write('ENROLID, '+ (','.join([*distinct_tcgpid_digits_dict_sorted.keys()])) )
        stationary_file.write(',')

        stationary_file.write(','.join([*ccs_distinct_dict_sorted.keys()]))
        stationary_file.write(',')       


        stationary_file.write(','.join([*proc_ccs_distinct_dict_sorted.keys()]))
        stationary_file.write(',')                

        stationary_file.write('Age')  
        stationary_file.write(',')              
        stationary_file.write('Sex')
        stationary_file.write(',') 
        stationary_file.write('Label') 
        stationary_file.write('\n')  

        for line_med in meds_filename:
            line_med = line_med.split(',')

            line_diag = diags_filename.readline()
            line_diag = line_diag.split(',')

            line_proc = procs_filename.readline()
            line_proc = line_proc.split(',')

            line_demog = demogs_filename.readline().rstrip('\n')
            line_demog = line_demog.split(',')            
        
            # pdb.set_trace()
            current_patient_age = index_to_calc_age - float(line_demog[dob_idx])# np.round(((2009 - float(line_demog[dob_idx])) - min_age)/(max_age-min_age), 2 )
            if current_patient_age <0 or current_patient_age>150:
                pdb.set_trace()
            # if float(line_med[0]) == 158825802:
            #     pdb.set_trace()
            current_patient_sex = float(line_demog[sex_idx])
            if not(float(line_med[0]) == float(line_diag[0]) == float(line_proc[0]) == float(line_demog[0])):
                pdb.set_trace()
                print('Warning: the streams do nott match')

            distinct_tcgpid_digits_dict = dict.fromkeys(distinct_tcgpid_digits_dict, 0)    
            ccs_distinct_dict = dict.fromkeys(ccs_distinct_dict, 0)    
            proc_ccs_distinct_dict = dict.fromkeys(proc_ccs_distinct_dict, 0)    

            # pdb.set_trace()
            multi_hot_meds_dict = create_stationary_meds(line_med, distinct_tcgpid_digits_dict, tcgpi_num_digits)
            multi_hot_diags_dict = create_stationary_diags(line_diag, icd_to_ccs_dict, ccs_distinct_dict)
            multi_hot_procs_dict = create_stationary_procs(line_proc, proc_cd_to_ccs_dict, proc_ccs_distinct_dict)
            stationary_file.write(line_med[0].replace('\n',''))
            stationary_file.write(',')
            stationary_file.write(','.join(map(repr, list(multi_hot_meds_dict.values()))))
            stationary_file.write(',')
            stationary_file.write(','.join(map(repr, list(multi_hot_diags_dict.values()))))
            stationary_file.write(',')
            stationary_file.write(','.join(map(repr, list(multi_hot_procs_dict.values()))))
            stationary_file.write(',')
            stationary_file.write(str(current_patient_age))
            stationary_file.write(',')
            stationary_file.write(str(current_patient_sex))
            stationary_file.write(',')
            stationary_file.write(str(line_demog[label_idx]))
            stationary_file.write('\n')    

def diff_month(d1, d2):

    return (d1.year - d2.year) * 12 + d1.month - d2.month

def blind_data(line_meds_splitted, line_diags_splitted, line_procs_splitted, line_demogs_splitted, prediction_win_size, enrolid, date_idx = 5, label_idx = 6):
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
        
        if diff_times <= prediction_win_size and diff_times >0:
            # pdb.set_trace()
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
        if diff_times <= prediction_win_size and diff_times >0:
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
        if diff_times <= prediction_win_size and diff_times >0:
            line_procs_blinded.extend(line_procs_splitted[i])
            line_procs_blinded.extend(['EOV'])

    return line_meds_blinded, line_diags_blinded, line_procs_blinded


print('Start....')
# remove_existing_train_val_test()
demogs_oud_yes_path = 'outputs/oud_yes_demographics_eligible.csv'
meds_oud_yes_path = 'outputs/oud_yes_medications_eligible.csv'
diags_oud_yes_path = 'outputs/oud_yes_diagnoses_eligible.csv'
procs_oud_yes_path = 'outputs/oud_yes_procedures_eligible.csv'

demogs_oud_no_path = 'outputs/oud_no_demographics_eligible.csv'
meds_oud_no_path = 'outputs/oud_no_medications_eligible.csv'
diags_oud_no_path = 'outputs/oud_no_diagnoses_eligible.csv'
procs_oud_no_path = 'outputs/oud_no_procedures_eligible.csv'

display_step = 10000

prediction_win_size = 6

oud_yes_demographics = pd.read_csv(demogs_oud_yes_path)
oud_no_demographics = pd.read_csv(demogs_oud_no_path)
test_data = pd.read_csv('outputs/test_stationary_normalized_features_filtered.csv')

enrolid_ind = 0
print('Extracting the records within prediction window')
# Positive samples   
with open(meds_oud_yes_path) as medications_oud_yes_file, open(diags_oud_yes_path) as diagnoses_oud_yes_file, open(procs_oud_yes_path) as procedures_oud_yes_file, open(demogs_oud_yes_path) as demographics_oud_yes_file, open('outputs/test_medications_within_pred_wind.csv', 'w') as test_meds_file, open('outputs/test_diagnoses_within_pred_wind.csv', 'w') as test_diags_file, open('outputs/test_procedures_within_pred_wind.csv', 'w') as test_procs_file, open('outputs/test_demographics_within_pred_wind.csv', 'w') as test_demogs_file, open('outputs/test_labels_within_pred_wind.csv','w') as test_labels_file:
    
    demogs_header = next(demographics_oud_yes_file)
    # pdb.set_trace()

    line_counter = 0 
    stopping_flag = 0
    for line_meds in medications_oud_yes_file:
        if line_counter % display_step == 0:
            print('Finished analyzing {} oud_yes patients data'.format(line_counter))
        line_counter +=1
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

        if current_enrolid not in test_data['ENROLID'].values:
            continue      
        # pdb.set_trace()
        current_demographics = oud_yes_demographics[oud_yes_demographics['ENROLID'] == current_enrolid]
        if current_demographics.empty == True:
            continue
        if len(current_demographics) > 1:
            print('Warning: there are more than 1 record in the demographics')
            pdb.set_trace()            
        # pdb.set_trace()    
        # Blind the data before writing in into to train, validation and test
        line_meds_blinded, line_diags_blinded, line_procs_blinded = blind_data(line_meds_splitted
                                                                                , line_diags_splitted
                                                                                , line_procs_splitted
                                                                                , current_demographics
                                                                                , prediction_win_size
                                                                                , current_enrolid)    
        

        if line_diags_blinded[-1] == "'EOV\\n'\n'":
            pdb.set_trace()
        # num_oud_in_val = 0
        # stopping_flag +=1
        # if stopping_flag > 400:
        #     break
        if current_enrolid in test_data['ENROLID'].values:
            test_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
            test_meds_file.write('\n')
            test_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
            test_diags_file.write('\n')  
            test_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
            test_procs_file.write('\n')  
            test_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
            # 1 means oud-yes patient
            test_demogs_file.write(', 1')
            test_demogs_file.write('\n')
            test_labels_file.write(str(current_enrolid))
            test_labels_file.write(',')
            test_labels_file.write('1, 0')
            test_labels_file.write('\n')

# Negative samples   
with open(meds_oud_no_path) as medications_oud_no_file, open(diags_oud_no_path) as diagnoses_oud_no_file, open(procs_oud_no_path) as procedures_oud_no_file, open(demogs_oud_no_path) as demographics_oud_no_file, open('outputs/test_medications_within_pred_wind.csv', 'a') as test_meds_file, open('outputs/test_diagnoses_within_pred_wind.csv', 'a') as test_diags_file, open('outputs/test_procedures_within_pred_wind.csv', 'a') as test_procs_file, open('outputs/test_demographics_within_pred_wind.csv', 'a') as test_demogs_file, open('outputs/test_labels_within_pred_wind.csv','a') as test_labels_file:
    demogs_oud_no_header = next(demographics_oud_no_file)
    # pdb.set_trace()
    # demog_read_flag = True
    line_counter = 0
    stopping_flag = 0
    for line_meds in medications_oud_no_file:
        if line_counter % display_step == 0:
            print('Finished analyzing {} oud_no patients data'.format(line_counter))
        line_counter +=1            
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
        # pdb.set_trace()
        current_enrolid = float(line_meds_splitted[enrolid_ind].replace("'",''))
      
        if current_enrolid not in test_data['ENROLID'].values:
            continue              
        current_demographics = oud_no_demographics[oud_no_demographics['ENROLID'] == current_enrolid]
        if current_demographics.empty == True:
            continue            
        if len(current_demographics) > 1:
            print('Warning: there are more than 1 record in the demographics')
            pdb.set_trace()

        line_meds_blinded, line_diags_blinded, line_procs_blinded = blind_data(line_meds_splitted
                                                                                , line_diags_splitted
                                                                                , line_procs_splitted
                                                                                , current_demographics
                                                                                , prediction_win_size
                                                                                , current_enrolid)    

        # stopping_flag +=1
        # if stopping_flag > 400:
        #     break        
        if current_enrolid in test_data['ENROLID'].values:
            test_meds_file.write((','.join(map(repr, line_meds_blinded))).replace("'","").replace('\n',''))
            test_meds_file.write('\n')
            test_diags_file.write((','.join(map(repr, line_diags_blinded))).replace("'","").replace('\n',''))
            test_diags_file.write('\n')  
            test_procs_file.write((','.join(map(repr, line_procs_blinded))).replace("'","").replace('\n',''))
            test_procs_file.write('\n')  
            test_demogs_file.write((','.join(map(repr, current_demographics.values[0]))).replace("'","").replace('\n',''))
            # 0 means oud-no patient
            test_demogs_file.write(', 0')
            test_demogs_file.write('\n')
            test_labels_file.write(str(current_enrolid))
            test_labels_file.write(',')
            test_labels_file.write('0, 1')
            test_labels_file.write('\n')                        
# pdb.set_trace()

meds_file = 'outputs/test_medications_within_pred_wind.csv'
diags_file = 'outputs/test_diagnoses_within_pred_wind.csv'
procs_file = 'outputs/test_procedures_within_pred_wind.csv'
demogs_file = 'outputs/test_demographics_within_pred_wind.csv'
dim_diags_file ='trvnorm_data/dim_diagnoses_trvnorm.csv'  
dim_procs_ccs_file ='trvnorm_data/dim_proceduresCCS_trvnorm.csv'  
distinct_tcgpid_file = 'trvnorm_data/distinct_tcgpid_trvnorm.csv'  
tcgpi_num_digits =2
   

create_stationary(meds_file
                    , diags_file
                    , procs_file
                    , demogs_file
                    , dim_diags_file
                    , dim_procs_ccs_file
                    , distinct_tcgpid_file
                    , tcgpi_num_digits
                    , 'test'
                    )

# pdb.set_trace()
print('Computing stats ...')
test_stationary_filename = 'outputs/test_stationary_within_pred_wind.csv'
test_meta_filename = 'outputs/test_demographics_shuffled.csv'

# Stats on the medications for oud-no
test_data = pd.read_csv(test_stationary_filename)
data_all = test_data#pd.concat([train_data, validation_data, test_data])

test_set_predictions = pd.read_csv('dosage/final_predictions_mupod.csv')
test_set_predictions.columns =['ENROLID', 'TP', 'TN', 'FP', 'FN']
test_set_predictions_tp = test_set_predictions[test_set_predictions['TP']==1]
test_set_predictions_tn = test_set_predictions[test_set_predictions['TN']==1]

data_all_oud_yes = data_all[data_all['ENROLID'].isin(test_set_predictions_tp['ENROLID'])]
data_all_oud_no = data_all[data_all['ENROLID'].isin(test_set_predictions_tn['ENROLID'])]


test_metadata = pd.read_csv(test_meta_filename)
data_all_meta = test_metadata#pd.concat([train_metadata, validation_metadata, test_metadata])
data_all_meta_oud_yes = data_all_meta[data_all_meta[' Label']==1]
data_all_meta_oud_no = data_all_meta[data_all_meta[' Label']==0]


# Calculate demographics
dob_oud_yes = data_all_meta_oud_yes['DOB']
age_oud_yes = 2020 - dob_oud_yes

dob_oud_no = data_all_meta_oud_no['DOB']
age_oud_no = 2020 - dob_oud_no


dob_all = data_all_meta['DOB']
age_all = 2020 - dob_all


num_oud_yes_sex_1 = data_all_meta_oud_yes[data_all_meta_oud_yes['SEX']==1].shape[0]
num_oud_yes_sex_2 = data_all_meta_oud_yes[data_all_meta_oud_yes['SEX']==2].shape[0]

num_oud_no_sex_1 = data_all_meta_oud_no[data_all_meta_oud_no['SEX']==1].shape[0]
num_oud_no_sex_2 = data_all_meta_oud_no[data_all_meta_oud_no['SEX']==2].shape[0]

num_oud_yes_patients = data_all_oud_yes.shape[0]
num_oud_no_patients = data_all_oud_no.shape[0]


with open('dosage/stat_within_pred_wind.csv','w') as stat_file:        
    
    # Average and std of age in case and cohort
    stat_file.write('Average age in TP is:\n')
    stat_file.write(str(age_oud_yes.mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of age in TP is:\n')
    stat_file.write(str(age_oud_yes.std()))        
    stat_file.write('\n')
    stat_file.write('25 percentile of age in TP is:\n')
    stat_file.write(str(np.percentile(age_oud_yes, 25)))        
    stat_file.write('\n')
    stat_file.write('50 percentile of age in TP is:\n')
    stat_file.write(str(np.percentile(age_oud_yes, 50)))        
    stat_file.write('\n')
    stat_file.write('75 percentile of age in TP is:\n')
    stat_file.write(str(np.percentile(age_oud_yes, 75)))        
    stat_file.write('\n')    


    stat_file.write('Average age in TN is:\n')
    stat_file.write(str(age_oud_no.mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of age in TN is:\n')
    stat_file.write(str(age_oud_no.std()))                
    stat_file.write('\n')
    stat_file.write('25 percentile of age in TN is:\n')
    stat_file.write(str(np.percentile(age_oud_no, 25)))        
    stat_file.write('\n')
    stat_file.write('50 percentile of age in TN is:\n')
    stat_file.write(str(np.percentile(age_oud_no, 50)))        
    stat_file.write('\n')
    stat_file.write('75 percentile of age in TN is:\n')
    stat_file.write(str(np.percentile(age_oud_no, 75)))        
    stat_file.write('\n') 

    stat_file.write('Average age in all patients:\n')
    stat_file.write(str(age_all.mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of age in all patients:\n')
    stat_file.write(str(age_all.std()))                
    stat_file.write('\n')
    stat_file.write('25 percentile of age in all patients:\n')
    stat_file.write(str(np.percentile(age_all, 25)))        
    stat_file.write('\n')
    stat_file.write('50 percentile of age in all patients:\n')
    stat_file.write(str(np.percentile(age_all, 50)))        
    stat_file.write('\n')
    stat_file.write('75 percentile of age in all patients:\n')
    stat_file.write(str(np.percentile(age_all, 75)))        
    stat_file.write('\n') 

    # Average and std of the number of month patinets hasve been on opioid medications in case and cohort       
    stat_file.write('Average number of month on opioid in TP is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of month on opioid in TP is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
    stat_file.write('\n')

    stat_file.write('Average number of month on opioid in TN is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of month on opioid in TN is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
    stat_file.write('\n')


    stat_file.write('Average number of month on opioid in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of month on opioid in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'].std()))        
    stat_file.write('\n')

    # Percentiles of on OUD
    stat_file.write('25 percentile of month on opioid in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of month on opioid in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of month on opioid in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of month on opioid in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of month on opioid in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of month on opioid in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of month on opioid in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of month on opioid in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of month on opioid in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHLY_OPIOID_PRESCS'], 75)))   
    stat_file.write('\n') 

    # Average and std of data availibility in case and cohort
    stat_file.write('Average data availibility(month) in TP is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of data availibility(month) in TP is:\n')
    stat_file.write(str(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'].std()))        
    stat_file.write('\n')

    stat_file.write('Average data availibility(month) in TN is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of data availibility(month) in TN is:\n')
    stat_file.write(str(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'].std()))        
    stat_file.write('\n')

    stat_file.write('Average data availibility(month) in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHS_IN_DATA'].mean()))
    stat_file.write('\n')
    stat_file.write('Standart deviation of data availibility(month) in all data is:\n')
    stat_file.write(str(data_all_meta['NUM_MONTHS_IN_DATA'].std()))        
    stat_file.write('\n')

    # Percentiles of data availability
    stat_file.write('25 percentile of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of data availibility(month) in oud_yes is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of data availibility(month) in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of data availibility(month) in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of data availibility(month) in oud_no is:\n')
    stat_file.write(str(np.percentile(data_all_meta_oud_no['NUM_MONTHS_IN_DATA'], 75)))   
    stat_file.write('\n')    

    stat_file.write('25 percentile of data availibility(month) in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHS_IN_DATA'], 25)))   
    stat_file.write('\n')
    stat_file.write('50 percentile of data availibility(month)d in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHS_IN_DATA'], 50)))   
    stat_file.write('\n')
    stat_file.write('75 percentile of data availibility(month) in all data is:\n')
    stat_file.write(str(np.percentile(data_all_meta['NUM_MONTHS_IN_DATA'], 75)))   
    stat_file.write('\n') 


    # Sex 
    stat_file.write('Number of people with SEX=1 in TP cohort is:\n')
    stat_file.write(str(num_oud_yes_sex_1))
    stat_file.write('\n')
    stat_file.write('Number of people with SEX=1 in TN cohort is:\n')
    stat_file.write(str(num_oud_no_sex_1))
    stat_file.write('\n')

    stat_file.write('Number of people with SEX=2 in TP cohort is:\n')
    stat_file.write(str(num_oud_yes_sex_2))
    stat_file.write('\n')       
    stat_file.write('Number of people with SEX=2 in TN cohort is:\n')
    stat_file.write(str(num_oud_no_sex_2))
    stat_file.write('\n')

    stat_file.write('Number of people with SEX=1 in all data is:\n')
    stat_file.write(str(num_oud_yes_sex_1+num_oud_no_sex_1))
    stat_file.write('\n')       
    stat_file.write('Number of people with SEX=2 in all data is:\n')
    stat_file.write(str(num_oud_yes_sex_2+num_oud_no_sex_2))
    stat_file.write('\n')

month_on_op_ustat, month_on_op_pvalue = mannwhitneyu(data_all_meta_oud_yes['NUM_MONTHLY_OPIOID_PRESCS'], data_all_meta_oud_no['NUM_MONTHLY_OPIOID_PRESCS'])
month_in_data_ustat, month_in_data_pvalue = mannwhitneyu(data_all_meta_oud_yes['NUM_MONTHS_IN_DATA'], data_all_meta_oud_no['NUM_MONTHS_IN_DATA'])

with open('dosage/p_values_u_stats_within_pred_wind.csv','w') as stat_file:  
    stat_file.write('Feature name, Num patients in TP, percentage in TP, number of patinets in our_no, percentage in TN, P-value, u-statistic, number of patinets in all data, perc in all data \n')
    
    # Month on opioid        
    stat_file.write('Months on opioid')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write('_')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write(str(month_on_op_pvalue))
    stat_file.write(',')
    stat_file.write(str(month_on_op_ustat))
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write('\n')

    stat_file.write('Data availibility(months)')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write('_')
    stat_file.write(',')
    stat_file.write('-')    
    stat_file.write(',')
    stat_file.write(str(month_on_op_pvalue))
    stat_file.write(',')
    stat_file.write(str(month_on_op_ustat))
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write(',')
    stat_file.write('-')
    stat_file.write('\n')

    # All other medications, diagnoses and procedures features
    for i in range(1, len(data_all.columns)-1):
        current_feature = data_all.columns[i]  

        stat_file.write(current_feature)
        stat_file.write(',')
        stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)))
        stat_file.write(',')
        stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)/num_oud_yes_patients))
        stat_file.write(',')
        stat_file.write(str(sum(data_all_oud_no[current_feature]>0)))
        stat_file.write(',')
        stat_file.write(str(sum(data_all_oud_no[current_feature]>0)/num_oud_no_patients))
        stat_file.write(',')            
        stat_file.write('NA')
        stat_file.write(',')
        stat_file.write('NA')
        stat_file.write(',')
        stat_file.write(str(sum(data_all[current_feature]>0)))
        stat_file.write(',')
        stat_file.write(str(sum(data_all[current_feature]>0)/(num_oud_no_patients+num_oud_yes_patients)))
        stat_file.write('\n')

        # if  (data_all_oud_yes[current_feature].sort_values().values == data_all_oud_no[current_feature].sort_values().values):
        #     stat_file.write(current_feature)
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)/num_oud_yes_patients))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_no[current_feature]>0)))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_no[current_feature]>0)/num_oud_no_patients))
        #     stat_file.write(',')            
        #     stat_file.write('identical vars')
        #     stat_file.write(',')
        #     stat_file.write('identical vars')
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all[current_feature]>0)))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all[current_feature]>0)/(num_oud_no_patients+num_oud_yes_patients)))
        #     stat_file.write('\n')
        # else:        
        #     temp_u_statistic, temp_p_value = mannwhitneyu(data_all_oud_yes[current_feature], data_all_oud_no[current_feature])   
        #     stat_file.write(current_feature)
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_yes[current_feature]>0)/num_oud_yes_patients))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_no[current_feature]>0)))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all_oud_no[current_feature]>0)/num_oud_no_patients))
        #     stat_file.write(',')               
        #     stat_file.write(str(temp_p_value))
        #     stat_file.write(',')
        #     stat_file.write(str(temp_u_statistic))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all[current_feature]>0)))
        #     stat_file.write(',')
        #     stat_file.write(str(sum(data_all[current_feature]>0)/(num_oud_no_patients+num_oud_yes_patients)))

        #     stat_file.write('\n')


print('The end')

