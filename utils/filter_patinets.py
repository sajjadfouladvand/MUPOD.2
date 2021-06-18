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
    with open(meds_path) as medications_file, open(diags_path) as diagnoses_file, open(procs_path) as procedures_file, open(demogs_path) as demographics_file, open(meds_path[:-4]+'_eligible.csv','w') as elig_medications_file, open(diags_path[:-4]+'_eligible.csv','w') as elig_diagnoses_file, open(procs_path[:-4]+'_eligible.csv','w') as elig_procedures_file, open('outputs/'+cohort+'_demographics_eligible.csv','w') as elig_demographics_file:
        demogs_header = next(demographics_file)
        elig_demographics_file.write(','.join(map(repr,['ENROLID', 'DOB', 'SEX', 'NUM_MONTHLY_OPIOID_PRESCS', 'NUM_MONTHS_IN_DATA', 'DIAGNOSES_DATE'])))    
        elig_demographics_file.write('\n')
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
    with open(meds_path) as medications_file, open(diags_path) as diagnoses_file, open(procs_path) as procedures_file, open(demogs_path) as demographics_file, open(meds_path[:-4]+'_eligible.csv','w') as elig_medications_file, open(diags_path[:-4]+'_eligible.csv','w') as elig_diagnoses_file, open(procs_path[:-4]+'_eligible.csv','w') as elig_procedures_file, open('outputs/'+cohort+'_demographics_eligible.csv','w') as elig_demographics_file:
        demogs_header = next(demographics_file)
        elig_demographics_file.write(','.join(map(repr,['ENROLID', 'DOB', 'SEX', 'NUM_MONTHLY_OPIOID_PRESCS','NUM_MONTHS_IN_DATA','LAST_RECORD_DATE'])))    
        elig_demographics_file.write('\n')
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
                                                                                                , ['F11', '3040', '3055']
                                                                                                )
            if  current_patinet_diagnoses_date != '2050':
                #pdb.set_trace()
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
    return 0        
'''
#==========================
max_len_meds_oud_yes = 0
max_len_meds_oud_no = 0
max_len_diags_oud_yes = 0
max_len_diags_oud_no = 0
end_of_visit_tocken = 'EOV'
num_skipped_patients_oud_yes = 0
num_skipped_patients_oud_no = 0
num_oud_yes_patients = 0 
num_oud_no_patients = 0
min_visit_threshold = 3

num_desired_meds = 10
num_desired_diags = 10
med_65_idx=7
# med_65_cut_of = 0.3
# pdb.set_trace()
# frequent_meds_oud = pd.read_csv('frequent_meds_OUD.csv')
# frequent_diags_oud = pd.read_csv('frequent_diags_OUD.csv')
# frequent_meds_oud_no = pd.read_csv('frequent_meds_nonOUD.csv')
# frequent_diags_oud_no = pd.read_csv('frequent_diags_nonOUD.csv')


# meds_codes_list_oud = frequent_meds_oud.columns.tolist()
# meds_codes_list_oud = [x.replace("'","") for x in meds_codes_list_oud]
# meds_codes_dict_oud = {}
# for i in range(len(meds_codes_list_oud)):
#     meds_codes_dict_oud[meds_codes_list_oud[i]] = 0 
# meds_codes_dict_oud_sorted=sorted(meds_codes_dict_oud.items(), key=operator.itemgetter(0))    
# meds_oud_header = np.array(meds_codes_dict_oud_sorted)[:,0].tolist()
# # pdb.set_trace()

# diags_codes_list_oud = frequent_diags_oud.columns.tolist()
# diags_codes_list_oud = [x.replace("'","") for x in diags_codes_list_oud]
# diags_codes_dict_oud = {}
# for i in range(len(diags_codes_list_oud)):
#     diags_codes_dict_oud[diags_codes_list_oud[i]] = 0 
# diags_codes_dict_oud_sorted=sorted(diags_codes_dict_oud.items(), key=operator.itemgetter(0))    
# diags_oud_header = np.array(diags_codes_dict_oud_sorted)[:,0].tolist()
# # pdb.set_trace()


# meds_codes_list_oud_no = frequent_meds_oud_no.columns.tolist()
# meds_codes_list_oud_no = [x.replace("'","") for x in meds_codes_list_oud_no]
# meds_codes_dict_oud_no = {}
# for i in range(len(meds_codes_list_oud_no)):
#     meds_codes_dict_oud_no[meds_codes_list_oud_no[i]] = 0 
# meds_codes_dict_oud_no_sorted=sorted(meds_codes_dict_oud_no.items(), key=operator.itemgetter(0))    
# meds_oud_no_header = np.array(meds_codes_dict_oud_no_sorted)[:,0].tolist()
# # pdb.set_trace()

# diags_codes_list_oud_no = frequent_diags_oud_no.columns.tolist()
# diags_codes_list_oud_no = [x.replace("'","") for x in diags_codes_list_oud_no]
# diags_codes_dict_oud_no = {}
# for i in range(len(diags_codes_list_oud_no)):
#     diags_codes_dict_oud_no[diags_codes_list_oud_no[i]] = 0 
# diags_codes_dict_oud_no_sorted=sorted(diags_codes_dict_oud_no.items(), key=operator.itemgetter(0))    
# diags_oud_no_header = np.array(diags_codes_dict_oud_no_sorted)[:,0].tolist()
# pdb.set_trace()


# diags_codes_dict = {}
# diags_codes_dict.update(diags_codes_dict_oud)
# diags_codes_dict.update(diags_codes_dict_oud_no)
# diags_codes_dict_sorted=sorted(diags_codes_dict.items(), key=operator.itemgetter(0))    
# diags_header = np.array(diags_codes_dict_sorted)[:,0].tolist()

# meds_codes_dict = {}
# meds_codes_dict.update(meds_codes_dict_oud)
# meds_codes_dict.update(meds_codes_dict_oud_no)
# meds_codes_dict_sorted=sorted(meds_codes_dict.items(), key=operator.itemgetter(0))    
# meds_header = np.array(meds_codes_dict_sorted)[:,0].tolist()


frequent_diags = pd.read_csv('ccs_visit_based_stat_oud_yes_disease_yes.csv')
frequent_meds = pd.read_csv('tcgpi_2digits_visit_based_stat_oud_yes_disease_yes.csv')
# pdb.set_trace()
#===== Selecting top k meds and diags
# pdb.set_trace()
#=== Removing CCS=205, because everone in the cohort has it
frequent_diags = frequent_diags.drop(frequent_diags.index[[0]])
#=== Removing 
frequent_meds = frequent_meds.drop(frequent_meds.index[[4]])

# frequent_diags_sorted = frequent_diags.sort_values(0, axis=1, ascending=False)
# frequent_diags_sorted_filtered = frequent_diags_sorted.iloc[:,0:num_desired_diags]
# frequent_meds_sorted = frequent_meds.sort_values(0, axis=1, ascending=False)
# frequent_meds_sorted_filtered = frequent_meds_sorted.iloc[:,0:num_desired_meds]
# frequent_diags_dict = frequent_diags.to_dict()
# frequent_meds_dict = frequent_meds.to_dict()
# Frequent diagnoses dictionary. To be used to form diagnoses vector for each visit       
diags_codes_list = frequent_diags['CCS_CATGRY'][:num_desired_diags].tolist()
# diags_codes_list = frequent_diags_sorted_filtered.columns.tolist()
# diags_codes_list = [x.replace("'","") for x in diags_codes_list]
diags_codes_dict = {}
for i in range(len(diags_codes_list)):
    diags_codes_dict[diags_codes_list[i]] = 0
# pdb.set_trace()
diags_codes_dict_sorted=sorted(diags_codes_dict.items(), key=operator.itemgetter(0))    
diags_header = np.array(diags_codes_dict_sorted)[:,0].tolist()

# meds_codes_list = frequent_meds_sorted_filtered.columns.tolist()
meds_codes_list = frequent_meds['medtype'][:num_desired_meds].tolist()
# meds_codes_list = [int(x) for x in meds_codes_list]
# meds_codes_list = [x.replace("'","") for x in meds_codes_list]
meds_codes_dict = {}
for i in range(len(meds_codes_list)):
    meds_codes_dict[meds_codes_list[i]] = 0 
# pdb.set_trace()
meds_codes_dict_sorted=sorted(meds_codes_dict.items(), key=operator.itemgetter(0))    
meds_header = np.array(meds_codes_dict_sorted)[:,0].tolist()

# gaps_vector = np.zeros(len(meds_codes_dict) + len(diags_codes_dict) + 2, dtype=float)
gaps_vector = np.zeros(len(meds_codes_dict) + 2, dtype=float)
# pdb.set_trace()

# Converting lists of medications and diagnoses to a structured format in which rows are visits and columns are frequent meds and diags
def visits_to_vectors(meds_codes_dict, diags_codes_dict, visits_list_meds, visits_list_diags):
    # pdb.set_trace()
    # Forming the medication vectors 
    patient_medications = np.zeros((len(visits_list_meds), len(meds_codes_dict)+1), dtype=int)
    for i in range(len(visits_list_meds)):
        meds_codes_dict = dict.fromkeys(meds_codes_dict, 0)
        if visits_list_meds[i][1] != 'NOCODE':
            # pdb.set_trace()
            for j in range(1, len(visits_list_meds[i])):
                if visits_list_meds[i][j] in meds_codes_dict.keys():
                    # pdb.set_trace()
                    meds_codes_dict[visits_list_meds[i][j]] = meds_codes_dict[visits_list_meds[i][j]] + 1           
            # pdb.set_trace()
            # print("THERE IS CODE")
            meds_codes_dict_sorted=sorted(meds_codes_dict.items(), key=operator.itemgetter(0))    
            dict_values_temp = np.array(meds_codes_dict_sorted)[:,1].tolist()
            dict_values_temp = [int(x) for x in dict_values_temp]
            patient_medications[i][0] = int(visits_list_meds[i][0])
            patient_medications[i][1:] = dict_values_temp
            
        else: 
            # pdb.set_trace()
            # print("NOCODE")
            patient_medications[i][0] = int(visits_list_meds[i][0])
            patient_medications[i][1:] = 0            

    # Forming the diagnoses vectors 
    # pdb.set_trace()
    patient_diagnoses = np.zeros((len(visits_list_diags), len(diags_codes_dict)+1), dtype=int)
    for i in range(len(visits_list_diags)):
        diags_codes_dict = dict.fromkeys(diags_codes_dict, 0)
        if visits_list_diags[i][1] != 'NOCODE':
            # pdb.set_trace()
            for j in range(1, len(visits_list_diags[i])):
                if int(visits_list_diags[i][j]) in diags_codes_dict.keys():
                    # pdb.set_trace()
                    diags_codes_dict[int(visits_list_diags[i][j])] = diags_codes_dict[int(visits_list_diags[i][j])] + 1           
            # pdb.set_trace()
            # print("THERE IS CODE")
            diags_codes_dict_sorted=sorted(diags_codes_dict.items(), key=operator.itemgetter(0))    
            dict_values_temp = np.array(diags_codes_dict_sorted)[:,1].tolist()
            dict_values_temp = [int(x) for x in dict_values_temp]
            patient_diagnoses[i][0] = int(visits_list_diags[i][0])
            patient_diagnoses[i][1:] = dict_values_temp
            
        else: 
            # pdb.set_trace()
            # print("NOCODE")
            patient_diagnoses[i][0] = int(visits_list_diags[i][0])
            patient_diagnoses[i][1:] = 0            
    # pdb.set_trace()
    return patient_medications, patient_diagnoses  
def demog_creator(dob, sex, vdates):
    # pdb.set_trace()
    patient_demogs = np.zeros((len(vdates), 3))
    for i in range(len(patient_demogs)):
        patient_demogs[i,0] = vdates[i]
        patient_demogs[i,1] = int(((vdates[i]//100 - dob) * 12 + vdates[i]%100 - 6)/12)
        patient_demogs[i,2] = sex
    return patient_demogs

med_65_ratio_pos = []
with open ('oud_yes_disease_yes_meds.csv') as oud_yes_meds_file, open('oud_yes_disease_yes_diags.csv') as oud_yes_diags_file, open('oud_yes_disease_yes_demogs.csv') as oud_yes_demogs_file, open('oud_yes_disease_yes_meds_formatted.csv', 'w') as oud_meds_fmt_file, open('oud_yes_disease_yes_diags_formatted.csv', 'w') as oud_diags_fmt_file, open('oud_yes_disease_yes_demogs_formatted.csv', 'w') as oud_demogs_fmt_file:    
    # pdb.set_trace()
    oud_meds_fmt_file.write('Enrolid')
    oud_meds_fmt_file.write(',')
    oud_meds_fmt_file.write('Visit Date')
    oud_meds_fmt_file.write(',')    
    oud_meds_fmt_file.write(','.join(map(repr,meds_header)))    
    # (",".join(["".join(x) for x in meds_header]))  
    oud_meds_fmt_file.write('\n')

    oud_diags_fmt_file.write('Enrolid')
    oud_diags_fmt_file.write(',')
    oud_diags_fmt_file.write('Visit Date')
    oud_diags_fmt_file.write(',')    
    oud_diags_fmt_file.write(','.join(map(repr,diags_header)))    
    # (",".join(["".join(str(x)) for x in diags_header]))  
    oud_diags_fmt_file.write('\n')    
    
    oud_demogs_fmt_file.write('Enrolid')
    oud_demogs_fmt_file.write(',')
    oud_demogs_fmt_file.write('Visit Date')
    oud_demogs_fmt_file.write(',')    
    oud_demogs_fmt_file.write('Age')
    oud_demogs_fmt_file.write(',')  
    oud_demogs_fmt_file.write('Sex')    
    oud_demogs_fmt_file.write('\n')          

    # next(oud_yes_demogs_file)
    patient_counter_oud_yes = 0
    for line in oud_yes_meds_file:
        patient_counter_oud_yes +=1
        if patient_counter_oud_yes%10000==0:
           print(patient_counter_oud_yes)
           # break
        # pdb.set_trace()
        line = line.split(',')
        line = [x.replace("'", '') for x in line]
        line[-1] = line[-1].replace('\n', '')
        current_enrolid_meds = float(line[0])
        visits_list_meds = [list(y) for x, y in itertools.groupby(line[1:], lambda z: z == end_of_visit_tocken) if not x]
        
        # Reading diagnoses visits
        line = oud_yes_diags_file.readline()
        line = line.split(',')
        line = [x.replace("'", '') for x in line]
        line[-1] = line[-1].replace('\n', '')
        current_enrolid_diags = float(line[0])
        visits_list_diags = [list(y) for x, y in itertools.groupby(line[1:], lambda z: z == end_of_visit_tocken) if not x]        
        # pdb.set_trace()
        patient_meds, patient_diags = visits_to_vectors(meds_codes_dict, diags_codes_dict, visits_list_meds, visits_list_diags)

        # Reading demographic information
        # pdb.set_trace()
        line = oud_yes_demogs_file.readline()
        line = line.split(',')
        line = [x.replace("'", '') for x in line]
        line[-1] = line[-1].replace('\n', '')
        current_enrolid_demogs = float(line[0])
        current_dob = int(line[1])
        current_sex = int(line[2])
        current_diagnoses_date = int(line[3].split('-')[0] + line[3].split('-')[1])
        
        if current_enrolid_meds != current_enrolid_diags or current_enrolid_meds != current_enrolid_demogs:
            print("Error: current streams enrolids don't match!")
            pdb.set_trace()

        patient_meds_valid = []
        patient_diags_valid = []
        # Descarding the visits after diagnoses date
        # pdb.set_trace()
        for j in range(len(patient_meds)):
            if  patient_meds[j,0] !=  patient_diags[j,0]:
                pdb.set_trace()
                print("Warning: visit date mistmatch") 
            if  patient_meds[j,0] < current_diagnoses_date:
                patient_meds_valid.append(patient_meds[j])
                patient_diags_valid.append(patient_diags[j])                         
        #====== Only looking at medications
        # pdb.set_trace()
        if len(patient_meds_valid)==0:# or len(patient_diags_valid)==0:
            num_skipped_patients_oud_yes += 1
            continue         
        if np.count_nonzero(np.sum(np.array(patient_meds_valid)[:,1:], axis=1)) < min_visit_threshold:# or np.count_nonzero(np.sum(np.array(patient_diags_valid)[:,1:], axis=1))<min_visit_threshold:
            num_skipped_patients_oud_yes += 1
            continue   
        
        med_65_ratio_pos.append(sum(np.array(patient_meds_valid)[:,med_65_idx])/len(np.array(patient_meds_valid)[:,med_65_idx]))
        
        # if (sum(np.array(patient_meds_valid)[:,med_65_idx])/len(np.array(patient_meds_valid)[:,med_65_idx])) < med_65_cut_of:
        #     num_skipped_patients_oud_yes += 1
        #     continue                    
        patient_meds_reshaped = np.reshape(np.array(patient_meds_valid), (1, -1)) 
        if np.shape(patient_meds_valid)[0] * (np.shape(patient_meds_valid)[1] -1) >  max_len_meds_oud_yes:
            max_len_meds_oud_yes = np.shape(patient_meds_valid)[0] * (np.shape(patient_meds_valid)[1] -1)
        
        patient_diags_reshaped = np.reshape(np.array(patient_diags_valid),(1, -1))   
        if np.shape(patient_diags_valid)[0] * (np.shape(patient_diags_valid)[1]-1) >  max_len_diags_oud_yes:
            max_len_diags_oud_yes = np.shape(patient_diags_valid)[0] * (np.shape(patient_diags_valid)[1]-1)
        
        current_patient_demogs = demog_creator(current_dob, current_sex, np.array(patient_meds_valid)[:,0])
        current_patient_demogs_reshaped = np.reshape(current_patient_demogs,(1,-1))

        num_oud_yes_patients += 1    
        oud_meds_fmt_file.write(str(current_enrolid_meds))
        oud_meds_fmt_file.write(',')
        oud_meds_fmt_file.write(','.join(map(repr,patient_meds_reshaped[0])))    
        oud_meds_fmt_file.write('\n')

        oud_diags_fmt_file.write(str(current_enrolid_diags))
        oud_diags_fmt_file.write(',')
        oud_diags_fmt_file.write(','.join(map(repr, patient_diags_reshaped[0])))    
        oud_diags_fmt_file.write('\n')        


        oud_demogs_fmt_file.write(str(current_enrolid_demogs))
        oud_demogs_fmt_file.write(',')
        oud_demogs_fmt_file.write(','.join(map(repr,current_patient_demogs_reshaped[0])))    
        oud_demogs_fmt_file.write('\n')        
np.savetxt("med_65_ratio_pos.csv", np.array(med_65_ratio_pos))

#=============== OUD negatives
print("==========End of processing OUD patients=========")
med_65_ratio_negs=[]
with open ('oud_no_disease_yes_meds.csv') as oud_no_meds_file, open('oud_no_disease_yes_diags.csv') as oud_no_diags_file, open('oud_no_disease_yes_demogs.csv') as oud_no_demogs_file, open('oud_no_disease_yes_meds_formatted.csv', 'w') as oud_no_meds_fmt_file, open('oud_no_disease_yes_diags_formatted.csv', 'w') as oud_no_diags_fmt_file, open('oud_no_disease_yes_demogs_formatted.csv', 'w') as oud_no_demogs_fmt_file:    
    oud_no_meds_fmt_file.write('Enrolid')
    oud_no_meds_fmt_file.write(',')
    oud_no_meds_fmt_file.write('Visit Date')
    oud_no_meds_fmt_file.write(',')    
    oud_no_meds_fmt_file.write(','.join(map(repr,meds_header)))   
    oud_no_meds_fmt_file.write('\n')

    oud_no_diags_fmt_file.write('Enrolid')
    oud_no_diags_fmt_file.write(',')
    oud_no_diags_fmt_file.write('Visit Date')
    oud_no_diags_fmt_file.write(',')    
    oud_no_diags_fmt_file.write(','.join(map(repr,diags_header))) 
    oud_no_diags_fmt_file.write('\n')    
    
    oud_no_demogs_fmt_file.write('Enrolid')
    oud_no_demogs_fmt_file.write(',')
    oud_no_demogs_fmt_file.write('Visit Date')
    oud_no_demogs_fmt_file.write(',')    
    oud_no_demogs_fmt_file.write('Age')
    oud_no_demogs_fmt_file.write(',')  
    oud_no_demogs_fmt_file.write('Sex')    
    oud_no_demogs_fmt_file.write('\n')          

    # next(oud_no_demogs_file)
    patient_counter_oud_no = 0
    for line in oud_no_meds_file:
        patient_counter_oud_no +=1
        if patient_counter_oud_no%10000==0:
           print(patient_counter_oud_no)
           # break
        line = line.split(',')
        line = [x.replace("'", '') for x in line]
        line[-1] = line[-1].replace('\n', '')
        current_enrolid_meds = float(line[0])
        visits_list_meds = [list(y) for x, y in itertools.groupby(line[1:], lambda z: z == end_of_visit_tocken) if not x]
        if current_enrolid_meds == 247501:
            pdb.set_trace()
        # Reading diagnoses visits
        line = oud_no_diags_file.readline()
        line = line.split(',')
        line = [x.replace("'", '') for x in line]
        line[-1] = line[-1].replace('\n', '')
        current_enrolid_diags = float(line[0])
        visits_list_diags = [list(y) for x, y in itertools.groupby(line[1:], lambda z: z == end_of_visit_tocken) if not x]        
        # pdb.set_trace()
        patient_meds, patient_diags = visits_to_vectors(meds_codes_dict, diags_codes_dict, visits_list_meds, visits_list_diags)

        # Reading demographic information
        # pdb.set_trace()
        line = oud_no_demogs_file.readline()
        line = line.split(',')
        line = [x.replace("'", '') for x in line]
        line[-1] = line[-1].replace('\n', '')
        current_enrolid_demogs = float(line[0])
        current_dob = int(float(line[1]))
        current_sex = int(float(line[2]))
        # current_diagnoses_date = int(line[3].split('-')[0] + line[3].split('-')[1])
        
        if current_enrolid_meds != current_enrolid_diags or current_enrolid_meds != current_enrolid_demogs:
            print("Error: current streams enrolids don't match!")
            pdb.set_trace()

        patient_meds_valid = []
        patient_diags_valid = []
        # Descarding the visits after diagnoses date
        for j in range(len(patient_meds)):
            if  patient_meds[j,0] !=  patient_diags[j,0]:
                pdb.set_trace()
                print("Warning: visit date mistmatch") 
            # if  patient_meds[j,0] < current_diagnoses_date:
            #     patient_meds_valid.append(patient_meds[j])
            #     patient_diags_valid.append(patient_diags[j])                         
        patient_meds_valid = patient_meds
        patient_diags_valid = patient_diags
        #====== Only looking at medications
        # pdb.set_trace()
        if len(patient_meds_valid)==0:# or len(patient_diags_valid)==0:
            num_skipped_patients_oud_no += 1
            continue         
        if np.count_nonzero(np.sum(np.array(patient_meds_valid)[:,1:], axis=1)) < min_visit_threshold:# or np.count_nonzero(np.sum(np.array(patient_diags_valid)[:,1:], axis=1))<min_visit_threshold:
            num_skipped_patients_oud_no += 1
            continue        
        patient_meds_reshaped = np.reshape(np.array(patient_meds_valid), (1, -1)) 
        if np.shape(patient_meds_valid)[0] * (np.shape(patient_meds_valid)[1] -1) >  max_len_meds_oud_no:
            max_len_meds_oud_no = np.shape(patient_meds_valid)[0] * (np.shape(patient_meds_valid)[1] -1)
        
        med_65_ratio_negs.append(sum(np.array(patient_meds_valid)[:,med_65_idx])/len(np.array(patient_meds_valid)[:,med_65_idx]))        
        
        # if (sum(np.array(patient_meds_valid)[:,med_65_idx])/len(np.array(patient_meds_valid)[:,med_65_idx])) < med_65_cut_of:
        #     num_skipped_patients_oud_no += 1
        #     continue           
        
        patient_diags_reshaped = np.reshape(np.array(patient_diags_valid),(1, -1))   
        if np.shape(patient_diags_valid)[0] * (np.shape(patient_diags_valid)[1]-1) >  max_len_diags_oud_no:
            max_len_diags_oud_no = np.shape(patient_diags_valid)[0] * (np.shape(patient_diags_valid)[1]-1)
        
        current_patient_demogs = demog_creator(current_dob, current_sex, np.array(patient_meds_valid)[:,0])
        current_patient_demogs_reshaped = np.reshape(current_patient_demogs,(1,-1))

        num_oud_no_patients += 1    
        oud_no_meds_fmt_file.write(str(current_enrolid_meds))
        oud_no_meds_fmt_file.write(',')
        oud_no_meds_fmt_file.write(','.join(map(repr,patient_meds_reshaped[0])))    
        oud_no_meds_fmt_file.write('\n')

        oud_no_diags_fmt_file.write(str(current_enrolid_diags))
        oud_no_diags_fmt_file.write(',')
        oud_no_diags_fmt_file.write(','.join(map(repr, patient_diags_reshaped[0])))    
        oud_no_diags_fmt_file.write('\n')        


        oud_no_demogs_fmt_file.write(str(current_enrolid_demogs))
        oud_no_demogs_fmt_file.write(',')
        oud_no_demogs_fmt_file.write(','.join(map(repr,current_patient_demogs_reshaped[0])))    
        oud_no_demogs_fmt_file.write('\n')    

np.savetxt("med_65_ratio_negs.csv", np.array(med_65_ratio_negs))

        # print("the end")
with open('stat_data_formatting_forLSTM.csv', 'w') as stat_file:
    stat_file.write('Number of oud-yes patients is:')
    stat_file.write(str(patient_counter_oud_yes))
    stat_file.write('\n')
    stat_file.write('Number of oud-no patients is:')
    stat_file.write(str(patient_counter_oud_no))
    stat_file.write('\n')    
    stat_file.write('Number of skipped oud patients is:')
    stat_file.write(str(num_skipped_patients_oud_yes))
    stat_file.write('\n')
    stat_file.write('Number of skipped oud-no patients is:')
    stat_file.write(str(num_skipped_patients_oud_no))
    stat_file.write('\n')     
    stat_file.write("The threshold to skip oud patients is:")
    stat_file.write(str(min_visit_threshold))
    stat_file.write('\n')  
    stat_file.write("Maximum meds sequence length for oud-yes patients is:")
    stat_file.write(str(max_len_meds_oud_yes))
    stat_file.write('\n')  
    stat_file.write("Maximum diags sequence length for oud-yes patients is:")
    stat_file.write(str(max_len_diags_oud_yes))
    stat_file.write('\n')      
    stat_file.write("Maximum meds sequence length for oud-no patients is:")
    stat_file.write(str(max_len_meds_oud_no))
    stat_file.write('\n')          
    stat_file.write("Maximum diags sequence length for oud-no patients is:")
    stat_file.write(str(max_len_diags_oud_no))
    stat_file.write('\n')      
    stat_file.write("Number of meds features:")
    stat_file.write(str(len(meds_header)))
    stat_file.write('\n')      
    stat_file.write("Number of diags features:")
    stat_file.write(str(len(diags_header)))
    stat_file.write('\n')      


'''