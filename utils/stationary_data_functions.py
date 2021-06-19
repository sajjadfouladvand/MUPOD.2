import os
import pdb
import pandas as pd
import numpy as np
import itertools
import collections
import math

def create_stationary_meds(line_med, 
                            distinct_tcgpid_2digit_dict
                            , tcgpi_num_digits
                            ):
    pdb.set_trace()
    line_med_splitted = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]            
    for i in range(len(line_med_splitted)):
        # if line_med_splitted[i][0]
        for j in range(1, len(line_med_splitted[i])): 
            if line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] in distinct_tcgpid_2digit_dict:           
                distinct_tcgpid_2digit_dict[line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]] += 1
            elif line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] == 'NO' or line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]=='EO':
                continue
            else:
                pdb.set_trace()   
                print('test') 
    distinct_tcgpid_2digit_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_2digit_dict.items())))
    # multi_hot_med = list(distinct_tcgpid_2digit_dict_sorted.values())
    # pdb.set_trace()
    return distinct_tcgpid_2digit_dict_sorted

def create_stationary_diags(line_diag
                            , icd_to_ccs_dict
                            , ccs_distinct_dict
                            ):
    line_diag_splitted = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]            
    for i in range(len(line_diag_splitted)):
        for j in range(1, len(line_diag_splitted[i])): 
            if line_diag_splitted[i][j].replace("'","") == 'NOCODE' or line_diag_splitted[i][j].replace("'","")[:3] == 'EOV':
                # pdb.set_trace()
                continue
            elif icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0] in ccs_distinct_dict:
                ccs_distinct_dict[icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0]] +=1
            else:
                pdb.set_trace()   
                pirnt('warning') 
    ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(ccs_distinct_dict.items())))
    # multi_hot_ccs = list(ccs_distinct_dict_sorted.values())
    # pdb.set_trace()
    return ccs_distinct_dict_sorted

def create_stationary_procs(line_proc
                            , proc_cd_to_ccs_dict
                            , proc_ccs_distinct_dict
                            ):
    # pdb.set_trace()
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
                proc_ccs_distinct_dict[-math.inf] +=1

            elif math.isnan(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0]):
                proc_ccs_distinct_dict[-math.inf] +=1
            elif proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0] in proc_ccs_distinct_dict:
                proc_ccs_distinct_dict[proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0]] +=1
            else:
                pdb.set_trace()  
                print('warning')  
    proc_ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(proc_ccs_distinct_dict.items())))
    # multi_hot_ccs = list(ccs_distinct_dict_sorted.values())
    # pdb.set_trace()
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
    # Min date of birth in TRVNORM is 1889 
    # Max date of birth in TRVNORM is 2018
    max_age = 130
    min_age = 1
    dim_procs_ccs = pd.read_csv(dim_procs_ccs_file)
    proc_ccs_distinct = dim_procs_ccs['ccs'].dropna().unique()
    proc_ccs_distinct_dict = {}
    for i in range(len(proc_ccs_distinct)):
        proc_ccs_distinct_dict[proc_ccs_distinct[i]] = 0 
    proc_ccs_distinct_dict[-math.inf] = 0

    proc_cd_to_ccs = dim_procs_ccs[['proccd', 'ccs']]  
    proc_cd_to_ccs_dict =  proc_cd_to_ccs.set_index('proccd').T.to_dict('list') 
    

    dim_diags = pd.read_csv(dim_diags_file)
    ccs_distinct = dim_diags['CCS_CATGRY'].dropna().unique()
    ccs_distinct_dict = {}
    for i in range(len(ccs_distinct)):
        ccs_distinct_dict[ccs_distinct[i]] = 0 

    icd_to_ccs = dim_diags[['DIAG_CD', 'CCS_CATGRY']]  
    icd_to_ccs_dict =  icd_to_ccs.set_index('DIAG_CD').T.to_dict('list') 

    distinct_tcgpid = pd.read_csv(distinct_tcgpid_file)
    distinct_tcgpid_digits = distinct_tcgpid['TCGPI_ID'].str[0:tcgpi_num_digits].dropna().unique()
    distinct_tcgpid_digits_dict = {}
    for i in range(len(distinct_tcgpid_digits)):
        distinct_tcgpid_digits_dict[distinct_tcgpid_digits[i]] = 0 
    counter = 0
    with open(meds_file) as meds_filename, open(diags_file) as diags_filename, open(procs_file) as procs_filename, open(demogs_file) as demogs_filename, open('outputs/'+fold_name+'_stationary.csv','w') as stationary_file:
        demogs_header = next(demogs_filename)
        stationary_file.write('ENROLID, '+(','.join(map(repr, sorted(distinct_tcgpid_digits)))))
        stationary_file.write(',')

        stationary_file.write(','.join(map(repr, sorted(ccs_distinct))))
        stationary_file.write(',')   

        stationary_file.write(','.join(map(repr, sorted(proc_ccs_distinct))))
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

            line_demog = demogs_filename.readline()
            line_demog = line_demog.split(',')            
        
            # pdb.set_trace()
            current_patient_age = np.round(((2009 - float(line_demog[dob_idx])) - min_age)/(max_age-min_age), 2 )
            current_patient_sex = float(line_demog[sex_idx])
            if not(int(line_med[0]) == int(line_diag[0]) == int(line_proc[0]) == int(line_demog[0])):
                pdb.set_trace()
                print('Warning: the streams do nott match')
            pdb.set_trace()
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
            stationary_file.write('\n')            







