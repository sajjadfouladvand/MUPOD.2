import os
import numpy as np
import pdb
import random
import pandas as pd
from scipy.spatial.distance import cdist

def blind_data(line_meds_splitted, line_diags_splitted, line_procs_splitted, diagnoses_date):

    return line_meds_blinded, line_diags_blinded, line_procs_blinded

def remove_existing_train_val_test():
    if os.path.exists("outputs/train_medications.csv"):
        os.remove("outputs/train_medications.csv")
    if os.path.exists("outputs/train_diagnoses.csv"):    
        os.remove("outputs/train_diagnoses.csv")
    if os.path.exists("outputs/train_procedures.csv"):    
        os.remove("outputs/train_procedures.csv")
    if os.path.exists("outputs/train_labels.csv"):
        os.remove("outputs/train_labels.csv")
    if os.path.exists("outputs/train_demographics.csv"):
        os.remove("outputs/train_demographics.csv")    

    if os.path.exists("outputs/validation_medications.csv"):
        os.remove("outputs/validation_medications.csv")
    if os.path.exists("outputs/validation_diagnoses.csv"):
        os.remove("outputs/validation_diagnoses.csv")
    if os.path.exists("outputs/validation_procedures.csv"):
        os.remove("outputs/validation_procedures.csv")
    if os.path.exists("outputs/validation_labels.csv"):
        os.remove("outputs/validation_labels.csv")
    if os.path.exists("outputs/validation_demographics.csv"):
        os.remove("outputs/validation_demographics.csv")

    if os.path.exists("outputs/test_medications.csv"):
        os.remove("outputs/test_medications.csv")
    if os.path.exists("outputs/test_diagnoses.csv"):
        os.remove("outputs/test_diagnoses.csv")
    if os.path.exists("outputs/test_procedures.csv"):
        os.remove("outputs/test_procedures.csv")
    if os.path.exists("outputs/test_labels.csv"):
        os.remove("outputs/test_labels.csv")
    if os.path.exists("outputs/test_demographics.csv"):
        os.remove("outputs/test_demographics.csv")    


def cohort_matching( demogs_oud_yes_path
                    , demogs_oud_no_path
                    , pos_to_negs_ratio
                    ):

    match_based_on = ['DOB', 'NUM_MONTHLY_OPIOID_PRESCS', 'NUM_MONTHS_IN_DATA']
    demogs_oud_yes = pd.read_csv(demogs_oud_yes_path)
    demogs_oud_yes.columns = demogs_oud_yes.columns.str.replace("'",'')
    demogs_oud_no = pd.read_csv(demogs_oud_no_path)
    demogs_oud_no.columns = demogs_oud_no.columns.str.replace("'",'')
    demogs_oud_no['MATCHED'] = 0
    matched_negatives_enrolids = list()
    for index, row in demogs_oud_yes.iterrows():
        oud_no_data_filtered_sex = demogs_oud_no[(demogs_oud_no['SEX'] == row['SEX']) & (demogs_oud_no['MATCHED'] == 0)]
        if len(oud_no_data_filtered_sex) == 0:
            # if len(demogs_oud_no[demogs_oud_no['MATCHED'] == 0]) == 0:
                # pdb.set_trace()
            # print('Warning: could not find a match based on patients sex') 
            similarities = cdist(np.reshape(row[match_based_on].values, (1,len(match_based_on))), np.reshape(demogs_oud_no[demogs_oud_no['MATCHED'] == 0][match_based_on].values, (-1,len(match_based_on))), metric='cosine')
            matched_negs = demogs_oud_no[demogs_oud_no['MATCHED'] == 0].iloc[similarities[0].argsort()[:pos_to_negs_ratio]]
            demogs_oud_no.loc[demogs_oud_no['ENROLID'].isin(matched_negs['ENROLID'].values),'MATCHED'] = 1
        else:
            similarities = cdist(np.reshape(row[match_based_on].values, (1,len(match_based_on))), np.reshape(oud_no_data_filtered_sex[match_based_on].values, (-1,len(match_based_on))), metric='cosine')
            matched_negs = oud_no_data_filtered_sex.iloc[similarities[0].argsort()[:pos_to_negs_ratio]]
            demogs_oud_no.loc[demogs_oud_no['ENROLID'].isin(matched_negs['ENROLID'].values),'MATCHED'] = 1
    demogs_oud_no[demogs_oud_no['MATCHED']==1].to_csv(demogs_oud_no_path[:-4]+'_matched.csv', index=False)
    return 0

def split_train_validation_test(meds_oud_yes_path
                                   , diags_oud_yes_path
                                   , procs_oud_yes_path
                                   , demogs_oud_yes_path
                                   , meds_oud_no_path
                                   , diags_oud_no_path
                                   , procs_oud_no_path
                                   , demogs_oud_no_path
                                   , train_ratio
                                   , validation_ratio
                                   ):
    # Remove existing files
    remove_existing_train_val_test()
    enrolid_ind = 0
    
    # Positive samples   
    with open(meds_oud_yes_path) as medications_oud_yes_file, open(diags_oud_yes_path) as diagnoses_oud_yes_file, open(procs_oud_yes_path) as procedures_oud_yes_file, open(demogs_oud_yes_path) as demographics_oud_yes_file, open('outputs/train_medications.csv', 'a') as train_meds_file, open('outputs/train_diagnoses.csv', 'a') as train_diags_file, open('outputs/train_procedures.csv', 'a') as train_procs_file, open('outputs/train_demographics.csv', 'a') as train_demogs_file, open('outputs/train_labels.csv','a') as train_labels_file, open('outputs/validation_medications.csv', 'a') as valid_meds_file, open('outputs/validation_diagnoses.csv', 'a') as valid_diags_file, open('outputs/validation_procedures.csv', 'a') as valid_procs_file, open('outputs/validation_demographics.csv', 'a') as valid_demogs_file, open('outputs/validation_labels.csv','a') as valid_labels_file, open('outputs/test_medications.csv', 'a') as test_meds_file, open('outputs/test_diagnoses.csv', 'a') as test_diags_file, open('outputs/test_procedures.csv', 'a') as test_procs_file, open('outputs/test_demographics.csv', 'a') as test_demogs_file, open('outputs/test_labels.csv','a') as test_labels_file:
        demogs_header = next(demographics_oud_yes_file)
        train_demogs_file.write(demogs_header)
        valid_demogs_file.write(demogs_header)
        test_demogs_file.write(demogs_header)

        for line_meds in medications_oud_yes_file:
            line_meds_splitted = line_meds.split(',')
            line_meds_splitted = [i.replace("'","") for i in line_meds_splitted]

            line_diags = diagnoses_oud_yes_file.readline()   
            line_diags_splitted = line_diags.split(',')
            line_diags_splitted = [i.replace("'","") for i in line_diags_splitted]

            line_procs = procedures_oud_yes_file.readline()   
            line_procs_splitted=line_procs.split(',')
            line_procs_splitted = [i.replace("'","") for i in line_procs_splitted]

            line_demogs = demographics_oud_yes_file.readline()   
            line_demogs_splitted = line_demogs.split(',')
            line_demogs_splitted = [i.replace("'","") for i in line_demogs_splitted]

            if not(int(line_demogs_splitted[enrolid_ind]) == int(line_meds_splitted[enrolid_ind].replace("'",'')) == int(line_diags_splitted[enrolid_ind].replace("'",'')) == int(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")
            current_enrolid = int(line_meds_splitted[enrolid_ind].replace("'",''))
            rand_temp=random.randint(1,100)
            # print(line_diags_splitted[enrolid_ind].replace("'",''))
            # if line_diags_splitted[enrolid_ind].replace("'",'') == '15331702':
            # pdb.set_trace()
            if line_diags_splitted[-1] == "'EOV\\n'\n'":
                pdb.set_trace()
            # num_oud_in_val = 0
            if (rand_temp > train_ratio *100) and (rand_temp <= (train_ratio+validation_ratio) * 100):     
                # num_oud_in_val+=1
                valid_meds_file.write((','.join(map(repr, line_meds_splitted))).replace("'","").replace('\n',''))
                valid_meds_file.write('\n')
                valid_diags_file.write((','.join(map(repr, line_diags_splitted))).replace("'","").replace('\n',''))
                valid_diags_file.write('\n')  
                valid_procs_file.write((','.join(map(repr, line_procs_splitted))).replace("'","").replace('\n',''))
                valid_procs_file.write('\n')  
                valid_demogs_file.write((','.join(map(repr, line_demogs_splitted))).replace("'","").replace('\n',''))
                valid_demogs_file.write('\n')
                valid_labels_file.write(str(current_enrolid))
                valid_labels_file.write(',')
                valid_labels_file.write('1, 0')
                valid_labels_file.write('\n')
            elif rand_temp > (train_ratio+validation_ratio) * 100:
                #======== Testing set
                # num_oud_in_test+=1
                test_meds_file.write((','.join(map(repr, line_meds_splitted))).replace("'","").replace('\n',''))
                test_meds_file.write('\n')
                test_diags_file.write((','.join(map(repr, line_diags_splitted))).replace("'","").replace('\n',''))
                test_diags_file.write('\n')  
                test_procs_file.write((','.join(map(repr, line_procs_splitted))).replace("'","").replace('\n',''))
                test_procs_file.write('\n')  
                test_demogs_file.write((','.join(map(repr, line_demogs_splitted))).replace("'","").replace('\n',''))
                test_demogs_file.write('\n')
                test_labels_file.write(str(current_enrolid))
                test_labels_file.write(',')
                test_labels_file.write('1, 0')
                test_labels_file.write('\n')
            else:
                #======== train set
                # num_oud_in_train+=1
                train_meds_file.write((','.join(map(repr, line_meds_splitted))).replace("'","").replace('\n',''))
                train_meds_file.write('\n')
                train_diags_file.write((','.join(map(repr, line_diags_splitted))).replace("'","").replace('\n',''))
                train_diags_file.write('\n')  
                train_procs_file.write((','.join(map(repr, line_procs_splitted))).replace("'","").replace('\n',''))
                train_procs_file.write('\n')  
                train_demogs_file.write((','.join(map(repr, line_demogs_splitted))).replace("'","").replace('\n',''))
                train_demogs_file.write('\n')
                train_labels_file.write(str(current_enrolid))
                train_labels_file.write(',')
                train_labels_file.write('1, 0')
                train_labels_file.write('\n')


    # Negative samples   
    with open(meds_oud_no_path) as medications_oud_no_file, open(diags_oud_no_path) as diagnoses_oud_no_file, open(procs_oud_no_path) as procedures_oud_no_file, open(demogs_oud_no_path) as demographics_oud_no_file, open('outputs/train_medications.csv', 'a') as train_meds_file, open('outputs/train_diagnoses.csv', 'a') as train_diags_file, open('outputs/train_procedures.csv', 'a') as train_procs_file, open('outputs/train_demographics.csv', 'a') as train_demogs_file, open('outputs/train_labels.csv','a') as train_labels_file, open('outputs/validation_medications.csv', 'a') as valid_meds_file, open('outputs/validation_diagnoses.csv', 'a') as valid_diags_file, open('outputs/validation_procedures.csv', 'a') as valid_procs_file, open('outputs/validation_demographics.csv', 'a') as valid_demogs_file, open('outputs/validation_labels.csv','a') as valid_labels_file, open('outputs/test_medications.csv', 'a') as test_meds_file, open('outputs/test_diagnoses.csv', 'a') as test_diags_file, open('outputs/test_procedures.csv', 'a') as test_procs_file, open('outputs/test_demographics.csv', 'a') as test_demogs_file, open('outputs/test_labels.csv','a') as test_labels_file:
        demogs_oud_no_header = next(demographics_oud_no_file)
        for line_meds in medications_oud_no_file:
            line_meds_splitted = line_meds.split(',')
            line_meds_splitted = [i.replace("'","") for i in line_meds_splitted]

            line_diags = diagnoses_oud_no_file.readline()   
            line_diags_splitted = line_diags.split(',')
            line_diags_splitted = [i.replace("'","") for i in line_diags_splitted]

            line_procs = procedures_oud_no_file.readline()   
            line_procs_splitted=line_procs.split(',')
            line_procs_splitted = [i.replace("'","") for i in line_procs_splitted]

            line_demogs = demographics_oud_no_file.readline()   
            line_demogs_splitted = line_demogs.split(',')
            line_demogs_splitted = [i.replace("'","") for i in line_demogs_splitted]

            if not(int(line_demogs_splitted[enrolid_ind]) == int(line_meds_splitted[enrolid_ind].replace("'",'')) == int(line_diags_splitted[enrolid_ind].replace("'",'')) == int(line_procs_splitted[enrolid_ind].replace("'",''))):
               pdb.set_trace()
               print("Warning: current streams don't match!")
            # pdb.set_trace()
            current_enrolid = int(line_meds_splitted[enrolid_ind].replace("'",''))
            rand_temp=random.randint(1,100)
            # num_oud_in_val = 0
            if (rand_temp > train_ratio *100) and (rand_temp <= (train_ratio+validation_ratio) * 100):     
                # num_oud_in_val+=1
                valid_meds_file.write((','.join(map(repr, line_meds_splitted))).replace("'","").replace('\n',''))
                valid_meds_file.write('\n')
                valid_diags_file.write((','.join(map(repr, line_diags_splitted))).replace("'","").replace('\n',''))
                valid_diags_file.write('\n')  
                valid_procs_file.write((','.join(map(repr, line_procs_splitted))).replace("'","").replace('\n',''))
                valid_procs_file.write('\n')  
                valid_demogs_file.write((','.join(map(repr, line_demogs_splitted))).replace("'","").replace('\n',''))
                valid_demogs_file.write('\n')
                valid_labels_file.write(str(current_enrolid))
                valid_labels_file.write(',')
                valid_labels_file.write('1, 0')
                valid_labels_file.write('\n')
            elif rand_temp > (train_ratio+validation_ratio) * 100:
                #======== Testing set
                # num_oud_in_test+=1
                test_meds_file.write((','.join(map(repr, line_meds_splitted))).replace("'","").replace('\n',''))
                test_meds_file.write('\n')
                test_diags_file.write((','.join(map(repr, line_diags_splitted))).replace("'","").replace('\n',''))
                test_diags_file.write('\n')  
                test_procs_file.write((','.join(map(repr, line_procs_splitted))).replace("'","").replace('\n',''))
                test_procs_file.write('\n')  
                test_demogs_file.write((','.join(map(repr, line_demogs_splitted))).replace("'","").replace('\n',''))
                test_demogs_file.write('\n')
                test_labels_file.write(str(current_enrolid))
                test_labels_file.write(',')
                test_labels_file.write('1, 0')
                test_labels_file.write('\n')
            else:
                #======== train set
                # num_oud_in_train+=1
                train_meds_file.write((','.join(map(repr, line_meds_splitted))).replace("'","").replace('\n',''))
                train_meds_file.write('\n')
                train_diags_file.write((','.join(map(repr, line_diags_splitted))).replace("'","").replace('\n',''))
                train_diags_file.write('\n')  
                train_procs_file.write((','.join(map(repr, line_procs_splitted))).replace("'","").replace('\n',''))
                train_procs_file.write('\n')  
                train_demogs_file.write((','.join(map(repr, line_demogs_splitted))).replace("'","").replace('\n',''))
                train_demogs_file.write('\n')
                train_labels_file.write(str(current_enrolid))
                train_labels_file.write(',')
                train_labels_file.write('1, 0')
                train_labels_file.write('\n')




