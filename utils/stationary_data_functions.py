import os
import pdb
import pandas as pd
import numpy as np
import itertools
import collections

def create_stationary_meds(line_med, distinct_tcgpid_2digit_dict, tcgpi_num_digits):
    line_med_splitted = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]            
    for i in range(len(line_med_splitted)):
        for j in range(1, len(line_med_splitted[i])): 
            if line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] in distinct_tcgpid_2digit_dict:           
                distinct_tcgpid_2digit_dict[line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]] += 1
            elif line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] == 'NO' or line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]=='EO':
                continue
            else:
                pdb.set_trace()    
    distinct_tcgpid_2digit_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_2digit_dict.items())))
    # multi_hot_med = list(distinct_tcgpid_2digit_dict_sorted.values())
    # pdb.set_trace()
    return distinct_tcgpid_2digit_dict_sorted

def create_stationary_diags(line_diag, icd_to_ccs_dict, ccs_distinct_dict):
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
    ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(ccs_distinct_dict.items())))
    # multi_hot_ccs = list(ccs_distinct_dict_sorted.values())
    # pdb.set_trace()
    return ccs_distinct_dict_sorted
            

# def diagnoses_embeding(line_diag, ccd_dict):

# def procedures_embeding():

def create_stationary(meds_file
                    , diags_file
                    , procs_file
                    , demogs_file
                    , dim_diags_file
                    , distinct_tcgpid_file    
                    , tcgpi_num_digits     
                    , fold_name           
                    ):
    dim_diags = pd.read_csv(dim_diags_file)
    ccs_distinct = dim_diags['CCS_CATGRY'].unique()
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

        stationary_file.write(','.join(map(repr, sorted(dim_diags['CCS_CATGRY'].unique()))))
        stationary_file.write(',')   

        stationary_file.write('Age')  
        stationary_file.write(',')              
        stationary_file.write('Sex')
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
            if not(int(line_med[0]) == int(line_diag[0]) == int(line_proc[0]) == int(line_demog[0])):
                pdb.set_trace()
                print('Warning: the streams do nott match')
        
            multi_hot_meds_dict = create_stationary_meds(line_med, distinct_tcgpid_digits_dict, tcgpi_num_digits)
            multi_hot_diags_dict = create_stationary_diags(line_diag, icd_to_ccs_dict, ccs_distinct_dict)

            stationary_file.write((line_med[0]))
            stationary_file.write(',')
            stationary_file.write(','.join(map(repr, list(multi_hot_meds_dict.values()))))
            stationary_file.write(',')
            stationary_file.write(','.join(map(repr, list(multi_hot_diags_dict.values()))))
            stationary_file.write('\n')            







