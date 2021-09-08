import os
import pdb
import pandas as pd
import numpy as np
import itertools
import collections
import math

def normalize_data_min_max(train_file_name
                            , validation_file_name
                            , test_file_name):
    # pdb.set_trace()
    epsil = 2.220446049250313e-16
    round_dig = 5
    train_data = pd.read_csv(train_file_name)
    # train_data=train_data.rename(columns = {'-inf':'others'})
    validation_data = pd.read_csv(validation_file_name)
    # validation_data=validation_data.rename(columns = {'-inf':'others'})
    test_data = pd.read_csv(test_file_name)
    # test_data=test_data.rename(columns = {'-inf':'others'})

    train_mins = train_data.min()
    val_mins = validation_data.min()
    test_mins = test_data.min()
    temp_mins = pd.concat([train_mins, val_mins, test_mins], axis=1, ignore_index=True)
    global_mins = temp_mins.min(axis=1)


    train_max = train_data.max()
    val_max = validation_data.max()
    test_max = test_data.max()
    temp_max = pd.concat([train_max, val_max, test_max], axis=1, ignore_index=True)
    global_max = temp_max.max(axis=1) 
    
    # global_max[global_max == global_mins]

    normalized_train=(train_data-global_mins)/((global_max-global_mins) + epsil)
    normalized_train['ENROLID'] = train_data['ENROLID']
    normalized_train['Label'] = train_data['Label']

    normalized_val=(validation_data-global_mins)/((global_max-global_mins) + epsil)
    normalized_val['ENROLID'] = validation_data['ENROLID']
    normalized_val['Label'] = validation_data['Label']

    normalized_test=(test_data-global_mins)/((global_max-global_mins) + epsil)
    normalized_test['ENROLID'] = test_data['ENROLID']
    normalized_test['Label'] = test_data['Label']
    

    normalized_train.round(round_dig).to_csv('outputs/train_stationary_normalized.csv', index=False)
    normalized_val.round(round_dig).to_csv('outputs/validation_stationary_normalized.csv', index=False)
    normalized_test.round(round_dig).to_csv('outputs/test_stationary_normalized.csv', index=False)
    # print('Test')

def create_stationary_meds(line_med, 
                            distinct_tcgpid_2digit_dict
                            , tcgpi_num_digits
                            ):
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
    # pdb.set_trace()
    distinct_tcgpid_2digit_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_2digit_dict.items())))    
    num_records = len(line_med_splitted)
    if(num_records != 0):
        distinct_tcgpid_2digit_dict_sorted = {k: np.round(v / (math.log10(num_records)+ epsil), round_dig) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}
    return distinct_tcgpid_2digit_dict_sorted

def create_stationary_diags(line_diag
                            , icd_to_ccs_dict
                            , ccs_distinct_dict
                            ):

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
    if(num_records != 0):
        ccs_distinct_dict_sorted = {k: np.round(v / (math.log10(num_records)+ epsil), round_dig) for k, v in ccs_distinct_dict_sorted.items()}
    return ccs_distinct_dict_sorted

def create_stationary_procs(line_proc
                            , proc_cd_to_ccs_dict
                            , proc_ccs_distinct_dict
                            ):
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
    if(num_records != 0):
        proc_ccs_distinct_dict_sorted = {k: np.round(v / (math.log10(num_records)+ epsil), round_dig) for k, v in proc_ccs_distinct_dict_sorted.items()}
    return proc_ccs_distinct_dict_sorted         

# def diagnoses_embeding(line_diag, ccd_dict):

# def procedures_embeding():

def create_stationary(meds_file
                    , diags_file
                    , procs_file
                    , demogs_file
                    , dim_diags_file
                    , dim_procs_ccs_file
                    , distinct_tcgpid_file    
                    , tcgpi_num_digits     
                    , fold_name                               
                    ):
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
    with open(meds_file) as meds_filename, open(diags_file) as diags_filename, open(procs_file) as procs_filename, open(demogs_file) as demogs_filename, open('outputs/'+fold_name+'_stationary.csv','w') as stationary_file:
        demogs_header = next(demogs_filename)

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
            stationary_file.write((line_med[0]))
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







