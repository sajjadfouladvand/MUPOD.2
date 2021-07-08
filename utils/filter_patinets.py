import operator
import os
import pdb 
import numpy as np
import pandas as pd
import itertools
from functools import reduce
from datetime import datetime

# pdb.set_trace()

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def check_data_availibility(line_med
                            , line_diag
                            , line_proc
                            , diagnoses_date
                            , prediction_win_size):
    num_opioid_prescs = 0
    num_non_opioid_prescs = 0    
    first_vdate = min(int(line_med[0][0]), int(line_diag[0][0]), int(line_proc[0][0]))
    num_month_available = diff_month(datetime(int(diagnoses_date)//100,int(diagnoses_date)%100, 1 ), datetime(first_vdate//100,first_vdate%100, 1))
    for i in range(len(line_med)):
        sublist = line_med[i]    
        date_diff_to_diag = diff_month(datetime(int(diagnoses_date)//100,int(diagnoses_date)%100, 1 ), datetime(int(sublist[0])//100,int(sublist[0])%100, 1))
        if any((x[:2] == '65' and x[:8] != '65200010' and x[:8] != '65100050' and x[:8] != '96448248') for x in sublist[1:]) == True and date_diff_to_diag >= prediction_win_size:
            num_opioid_prescs +=1
        #if any((x[:2] != '65' and x[:8] != '96448248') for x in sublist[1:]) == True and date_diff_to_diag >= prediction_win_size:    
        #    num_non_opioid_prescs +=1

    return num_month_available, num_opioid_prescs

def find_diagnoses_date(meds
    , diags
    , oud_codes_meds
    , oud_codes_icd
    #, temp
    ):
    diagnoses_date_icd = '2050'
    diagnoses_date_tcgpi = '2050'
    diagnoses_type = '0'
    # if temp == 145267101:
    #     pdb.set_trace()
    for i in range(len(meds)):
        sublist = meds[i]
        if any(x[:8] in oud_codes_meds for x in sublist[1:]) == True:
            # if temp == 145267101:
            #     pdb.set_trace()            
            diagnoses_date_tcgpi =  sublist[0]
            diagnoses_type = 'TCGPI' 
            break       
    for i in range(len(diags)):
        sublist = diags[i]
        if any(x[:4] in oud_codes_icd or x[:3] in oud_codes_icd for x in sublist[1:]) == True:
            # if temp == 145267101:
            #     pdb.set_trace()           
            diagnoses_date_icd =  sublist[0]
            diagnoses_type = 'ICD'   
            break        
    return min(diagnoses_date_tcgpi, diagnoses_date_icd), diagnoses_type

def extract_enrolids(meds_path
                    , diags_path
                    , procs_path
                    , demogs_path):
    enrolids_meds = []
    enrolids_diags = []
    enrolids_procs = []
    enrolids_demogs = []
    with open(meds_path) as medications_file:
        for line in medications_file:            
            enrolids_meds.append(int(line.split(',')[0]))
    with open(diags_path) as diagnoses_file:    
        for line in diagnoses_file:            
            enrolids_diags.append(int(line.split(',')[0]))        
    with open(procs_path) as procedures_file:  
        for line in procedures_file:            
            enrolids_procs.append(int(line.split(',')[0]))                    
    with open(demogs_path) as demographics_file:
        demogs_header = next(demographics_file)
        # pdb.set_trace()
        for line in demographics_file:  
            if (line.split(',')[0]) == '':
                continue
            enrolids_demogs.append(int(line.split(',')[0]))                                
    enrolids_meds_df = pd.DataFrame(enrolids_meds, columns=['ENROLIDS'])
    enrolids_diags_df = pd.DataFrame(enrolids_diags, columns=['ENROLIDS'])
    enrolids_procs_df = pd.DataFrame(enrolids_procs, columns=['ENROLIDS'])
    enrolids_demogs_df = pd.DataFrame(enrolids_demogs, columns=['ENROLIDS'])
    all_enrolids_unique = reduce(np.intersect1d, (enrolids_meds, enrolids_diags, enrolids_procs, enrolids_demogs))        
    all_enrolids_unique_sorted = np.sort(all_enrolids_unique)
    np.savetxt('outputs/enrolids_all_uniques_oud_yes.csv', all_enrolids_unique_sorted)
    return all_enrolids_unique_sorted

def filter_patients_positives(meds_path
                            , diags_path
                            , procs_path
                            , demogs_path
                            # , metadata_path
                            , cohort
                            , min_month_available
                            , min_num_opioid
                            , prediction_win_size
                            , display_step):
    enrolid_idx = 0
    med_dim = 3
    end_of_visit_tocken = 'EOV'
    enrolids_all_unique = extract_enrolids(meds_path, diags_path, procs_path, demogs_path)
    demographics_data = pd.read_csv(demogs_path)
    # metadata = pd.read_csv(metadata_path)
    demographics_data_dict = demographics_data.set_index('ENROLID').T.to_dict('list')
    # metadata_dict = metadata.set_index('ENROLID').T.to_dict('list')
    with open(meds_path) as medications_file, open(diags_path) as diagnoses_file, open(procs_path) as procedures_file, open(demogs_path) as demographics_file, open(meds_path[:-4]+'_eligible.csv','w') as elig_medications_file, open(diags_path[:-4]+'_eligible.csv','w') as elig_diagnoses_file, open(procs_path[:-4]+'_eligible.csv','w') as elig_procedures_file, open('outputs/'+cohort+'_demographics_eligible.csv','w') as elig_demographics_file, open('outputs/'+cohort+'_un_eligibles.csv','w') as uneligs_file:
        demogs_header = next(demographics_file)
        elig_demographics_file.write(','.join(map(repr,['ENROLID', 'DOB', 'SEX', 'NUM_MONTHLY_OPIOID_PRESCS', 'NUM_MONTHS_IN_DATA', 'DIAGNOSES_DATE'])))    
        elig_demographics_file.write('\n')
        uneligs_file.write(','.join(map(repr,['ENROLID', 'num_opioid_prescriptions', 'num_month_data_available'])))    
        uneligs_file.write('\n')
        # pdb.set_trace()
        for i in range(len(enrolids_all_unique)):
            current_enrolid = enrolids_all_unique[i]
            line_med = medications_file.readline()
            line_med = line_med.split(',')
            line_diag = diagnoses_file.readline()
            line_diag = line_diag.split(',')
            line_proc = procedures_file.readline()
            line_proc = line_proc.split(',')


            if int(line_med[enrolid_idx]) < current_enrolid:
                while current_enrolid > int(line_med[enrolid_idx]):
                    line_med = medications_file.readline()
                    line_med = line_med.split(',')  
            elif int(line_med[enrolid_idx]) > current_enrolid:                       
                pdb.set_trace()
                print('Warning')
            if int(line_diag[enrolid_idx]) < current_enrolid:
                while current_enrolid > int(line_diag[enrolid_idx]):
                    line_diag = diagnoses_file.readline()
                    line_diag = line_diag.split(',') 
            elif int(line_diag[enrolid_idx]) > current_enrolid:
                pdb.set_trace()
                print('Warning')
            if int(line_proc[enrolid_idx]) < current_enrolid:
                while current_enrolid > int(line_proc[enrolid_idx]):
                    line_proc = procedures_file.readline()
                    line_proc = line_proc.split(',') 
            elif int(line_proc[enrolid_idx]) > current_enrolid:
                pdb.set_trace()
                print('Warning')                

            current_patinet_demogs = demographics_data_dict[current_enrolid]
            # current_patinet_metadata = metadata_dict[current_enrolid]
            line_med = [x.replace("'", '') for x in line_med]
            line_med[-1] = line_med[-1].replace('\n', '')
            line_med_splitted = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]            
            line_diag = [x.replace("'", '') for x in line_diag]
            line_diag[-1] = line_diag[-1].replace('\n', '') 
            line_diag_splitted = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]                        
            line_proc = [x.replace("'", '') for x in line_proc]
            line_proc[-1] = line_proc[-1].replace('\n', '')                         
            line_proc_splitted = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]                        
                      
            current_patinet_diagnoses_date, current_patinet_diagnoses_type = find_diagnoses_date(line_med_splitted
                                                                                                , line_diag_splitted
                                                                                                , ['65200010', '65100050', '96448248']                                                                                                                                            
                                                                                                , ['F11', '3040', '3055'])
            if current_patinet_diagnoses_date == '2050':
                # pdb.set_trace()
                print('Warning: no diagnoses date was found for the current patinet with enrolid {}.'.format(current_enrolid))
                continue
            current_patinet_num_month_available, current_patinet_num_opioid_prescs = check_data_availibility(line_med_splitted
                                                                                                            , line_diag_splitted
                                                                                                            , line_proc_splitted
                                                                                                            , current_patinet_diagnoses_date
                                                                                                            , prediction_win_size)
            # if  current_patinet_num_opioid_prescs > current_patinet_metadata[0]+1 or current_patinet_num_opioid_prescs < current_patinet_metadata[0]-1:
            # pdb.set_trace()
            if current_patinet_num_opioid_prescs >= min_num_opioid and  current_patinet_num_month_available >= min_month_available:
                elig_medications_file.write(','.join(map(repr,line_med)))   
                elig_medications_file.write('\n')
                elig_diagnoses_file.write(','.join(map(repr,line_diag)))   
                elig_diagnoses_file.write('\n')
                elig_procedures_file.write(','.join(map(repr,line_proc)))   
                elig_procedures_file.write('\n')                
                elig_demographics_file.write(str(current_enrolid))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_demogs[0]))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_demogs[1]))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_num_opioid_prescs))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_num_month_available))
                elig_demographics_file.write(',')
                elig_demographics_file.write(current_patinet_diagnoses_date)
                elig_demographics_file.write('\n')
            else:
                uneligs_file.write(str(current_enrolid))
                uneligs_file.write(',')
                uneligs_file.write(str(current_patinet_num_opioid_prescs))
                uneligs_file.write(',')
                uneligs_file.write(str(current_patinet_num_month_available))
                uneligs_file.write('\n')
                    
    return 0    



def filter_patients_negatives(meds_path
                            , diags_path
                            , procs_path
                            , demogs_path
                            # , metadata_path
                            , cohort
                            , min_month_available
                            , min_num_opioid
                            , prediction_win_size
                            , display_step):
    enrolid_idx = 0
    med_dim = 3
    end_of_visit_tocken = 'EOV'
    enrolids_all_unique = extract_enrolids(meds_path, diags_path, procs_path, demogs_path)
    demographics_data = pd.read_csv(demogs_path)
    # metadata = pd.read_csv(metadata_path)
    demographics_data_dict = demographics_data.set_index('ENROLID').T.to_dict('list')
    # metadata_dict = metadata.set_index('ENROLID').T.to_dict('list')
    with open(meds_path) as medications_file, open(diags_path) as diagnoses_file, open(procs_path) as procedures_file, open(demogs_path) as demographics_file, open(meds_path[:-4]+'_eligible.csv','w') as elig_medications_file, open(diags_path[:-4]+'_eligible.csv','w') as elig_diagnoses_file, open(procs_path[:-4]+'_eligible.csv','w') as elig_procedures_file, open('outputs/'+cohort+'_demographics_eligible.csv','w') as elig_demographics_file, open('outputs/'+cohort+'_un_eligibles.csv','w') as uneligs_file:
        demogs_header = next(demographics_file)
        elig_demographics_file.write(','.join(map(repr,['ENROLID', 'DOB', 'SEX', 'NUM_MONTHLY_OPIOID_PRESCS','NUM_MONTHS_IN_DATA','LAST_RECORD_DATE'])))    
        elig_demographics_file.write('\n')
        uneligs_file.write(','.join(map(repr,['ENROLID', 'num_opioid_prescriptions', 'num_month_data_available'])))    
        uneligs_file.write('\n')        
        # pdb.set_trace()
        for i in range(len(enrolids_all_unique)):
            current_enrolid = enrolids_all_unique[i]
            if current_enrolid == 145267101:
               pdb.set_trace()
            line_med = medications_file.readline()
            line_med = line_med.split(',')
            line_diag = diagnoses_file.readline()
            line_diag = line_diag.split(',')
            line_proc = procedures_file.readline()
            line_proc = line_proc.split(',')


            if int(line_med[enrolid_idx]) < current_enrolid:
                while current_enrolid > int(line_med[enrolid_idx]):
                    line_med = medications_file.readline()
                    line_med = line_med.split(',')  
            elif int(line_med[enrolid_idx]) > current_enrolid:                       
                pdb.set_trace()
                print('Warning')
            if int(line_diag[enrolid_idx]) < current_enrolid:
                while current_enrolid > int(line_diag[enrolid_idx]):
                    line_diag = diagnoses_file.readline()
                    line_diag = line_diag.split(',') 
            elif int(line_diag[enrolid_idx]) > current_enrolid:
                pdb.set_trace()
                print('Warning')
            if int(line_proc[enrolid_idx]) < current_enrolid:
                while current_enrolid > int(line_proc[enrolid_idx]):
                    line_proc = procedures_file.readline()
                    line_proc = line_proc.split(',') 
            elif int(line_proc[enrolid_idx]) > current_enrolid:
                pdb.set_trace()
                print('Warning')                
            if current_enrolid == 145267101:
               pdb.set_trace()
            current_patinet_demogs = demographics_data_dict[current_enrolid]
            # current_patinet_metadata = metadata_dict[current_enrolid]
            line_med = [x.replace("'", '') for x in line_med]
            line_med[-1] = line_med[-1].replace('\n', '')
            line_med_splitted = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]            
            line_diag = [x.replace("'", '') for x in line_diag]
            line_diag[-1] = line_diag[-1].replace('\n', '') 
            line_diag_splitted = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]                        
            line_proc = [x.replace("'", '') for x in line_proc]
            line_proc[-1] = line_proc[-1].replace('\n', '')                         
            line_proc_splitted = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]                        
            current_patinet_diagnoses_date, current_patinet_diagnoses_type = find_diagnoses_date(line_med_splitted
                                                                                                , line_diag_splitted
                                                                                                , ['65200010', '65100050', '96448248']
                                                                                                , ['F11', '3040', '3055']
                                                                                                )
            if  current_patinet_diagnoses_date != '2050':
                pdb.set_trace()
                print('Warning: current patinet with enrolid {} is OUD-yes, but is labeled as OUD-no.'.format(current_enrolid))            
                continue                            
            current_patinet_last_recorrd_date = max(int(line_med_splitted[-1][0]), int(line_diag_splitted[-1][0]), int(line_proc_splitted[-1][0]))
            current_patinet_num_month_available, current_patinet_num_opioid_prescs = check_data_availibility(line_med_splitted
                                                                                                            , line_diag_splitted
                                                                                                            , line_proc_splitted
                                                                                                            , current_patinet_last_recorrd_date
                                                                                                            , prediction_win_size)
            # if  current_patinet_num_opioid_prescs > current_patinet_metadata[0]+1 or current_patinet_num_opioid_prescs < current_patinet_metadata[0]-1:
                # pdb.set_trace()
            if current_patinet_num_opioid_prescs >= min_num_opioid and  current_patinet_num_month_available >= min_month_available:
                elig_medications_file.write(','.join(map(repr,line_med)))   
                elig_medications_file.write('\n')
                elig_diagnoses_file.write(','.join(map(repr,line_diag)))   
                elig_diagnoses_file.write('\n')
                elig_procedures_file.write(','.join(map(repr,line_proc)))   
                elig_procedures_file.write('\n')                
                elig_demographics_file.write(str(current_enrolid))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_demogs[0]))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_demogs[1]))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_num_opioid_prescs))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_num_month_available))
                elig_demographics_file.write(',')
                elig_demographics_file.write(str(current_patinet_last_recorrd_date))
                elig_demographics_file.write('\n')
            else:
                uneligs_file.write(str(current_enrolid))
                uneligs_file.write(',')
                uneligs_file.write(str(current_patinet_num_opioid_prescs))
                uneligs_file.write(',')
                uneligs_file.write(str(current_patinet_num_month_available))
                uneligs_file.write('\n')                
    return 0        
