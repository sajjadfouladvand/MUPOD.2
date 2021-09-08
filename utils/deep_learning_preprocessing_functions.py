import os
import pdb
import pandas as pd
import numpy as np
import itertools
import collections
import math
from datetime import datetime
from dateutil.relativedelta import *

def demog_creator(dob, sex, vdates):
    # pdb.set_trace()
    patient_demogs = np.zeros((len(vdates), 3))
    for i in range(len(patient_demogs)):
        patient_demogs[i,0] = vdates[i]
        patient_demogs[i,1] = int(((vdates[i]//100 - dob) * 12 + vdates[i]%100 - 6)/12)
        patient_demogs[i,2] = sex
    return patient_demogs

def normalize_demogs(demogs_data):
    # train_demogs = pd.read_csv('outputs/train_demographics_shuffled.csv')
    # validation_demogs = pd.read_csv('outputs/validation_demographics_shuffled.csv')
    # test_demogs = pd.read_csv('outputs/test_demographics_shuffled.csv')
    # demogs_all = pd.concat([train_demogs, validation_demogs, test_demogs])
    # min_dob = demogs_all['DOB'].min()
    # max_dob = demogs_all['DOB'].max()
    max_age = 2020 - 1914
    min_age = 2020 - 2003
    min_sex=1
    max_sex=2
    demogs_data[:,1] = np.round((demogs_data[:,1]- min_age)/(max_age - min_age),5)
    demogs_data[:,2] = (demogs_data[:,2]- min_sex)/(max_sex - min_sex)
    return demogs_data

def shuffle_data(train_meds_file
                , train_diags_file
                , train_procs_file
                , train_demogs_file
                
                , validation_meds_file
                , validation_diags_file
                , validation_procs_file
                , validation_demogs_file                                   
                
                , test_meds_file
                , test_diags_file
                , test_procs_file
                , test_demogs_file                                  
                ):
    # pdb.set_trace()
    
    train_meds_data = pd.read_csv(train_meds_file, header=None, sep='\n')
    train_diags_data = pd.read_csv(train_diags_file, header=None, sep='\n')
    train_procs_data = pd.read_csv(train_procs_file, header=None, sep='\n')
    train_demogs_data = pd.read_csv(train_demogs_file, header=None, sep='\n')
    demographics_header = train_demogs_data.iloc[0].values.tolist()
    train_demogs_data = train_demogs_data.iloc[1:].reset_index(drop=True)

    train_meds_data = train_meds_data.sample(frac=1)
    train_diags_data = train_diags_data.reindex(train_meds_data.index)
    train_procs_data = train_procs_data.reindex(train_meds_data.index)
    train_demogs_data = train_demogs_data.reindex(train_meds_data.index)
    

    train_meds_data.to_csv(train_meds_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    train_diags_data.to_csv(train_diags_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    train_procs_data.to_csv(train_procs_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    train_demogs_data.to_csv(train_demogs_file[:-4]+'_shuffled.csv', index=False, header=demographics_header, sep='\n')

    # train_all = pd.concat([train_meds_data, train_diags_data, train_procs_data, train_demogs_data])

    validation_meds_data = pd.read_csv(validation_meds_file, header=None, sep='\n')
    validation_diags_data = pd.read_csv(validation_diags_file, header=None, sep='\n')
    validation_procs_data = pd.read_csv(validation_procs_file, header=None, sep='\n')
    validation_demogs_data = pd.read_csv(validation_demogs_file, header=None, sep='\n', skiprows=[0])

    validation_meds_data = validation_meds_data.sample(frac=1)
    validation_diags_data = validation_diags_data.reindex(validation_meds_data.index)
    validation_procs_data = validation_procs_data.reindex(validation_meds_data.index)
    validation_demogs_data = validation_demogs_data.reindex(validation_meds_data.index)

    validation_meds_data.to_csv(validation_meds_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    validation_diags_data.to_csv(validation_diags_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    validation_procs_data.to_csv(validation_procs_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    validation_demogs_data.to_csv(validation_demogs_file[:-4]+'_shuffled.csv', index=False, header=demographics_header, sep='\n')


    test_meds_data = pd.read_csv(test_meds_file, header=None, sep='\n')
    test_diags_data = pd.read_csv(test_diags_file, header=None, sep='\n')
    test_procs_data = pd.read_csv(test_procs_file, header=None, sep='\n')
    test_demogs_data = pd.read_csv(test_demogs_file, header=None, sep='\n', skiprows=[0])

    test_meds_data = test_meds_data.sample(frac=1)
    test_diags_data = test_diags_data.reindex(test_meds_data.index)
    test_procs_data = test_procs_data.reindex(test_meds_data.index)
    test_demogs_data = test_demogs_data.reindex(test_meds_data.index)
    
    test_meds_data.to_csv(test_meds_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    test_diags_data.to_csv(test_diags_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    test_procs_data.to_csv(test_procs_file[:-4]+'_shuffled.csv', index=False, header=False, sep='\n')
    test_demogs_data.to_csv(test_demogs_file[:-4]+'_shuffled.csv', index=False, header=demographics_header, sep='\n')



def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def create_multihot_procs(line_proc
                            , proc_cd_to_ccs_dict
                            , proc_ccs_distinct_dict
                            ): 
    start_date = 200901
    end_date = 202006  
    start_less_than_2009 = 0  
    line_proc_splitted = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]            
    
    patient_multi_hot_vectors = []
    patient_multi_hot_vectors.append(float(line_proc[0]))    
    if len(line_proc_splitted) == 0:    
        current_record_date = start_date
        while current_record_date <= end_date:            
            patient_multi_hot_vectors.append(current_record_date)
            patient_multi_hot_vectors.extend([0]*len(proc_ccs_distinct_dict))
            current_record_date = int((datetime(current_record_date//100, current_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
        return patient_multi_hot_vectors, start_less_than_2009

    first_record_date = int(line_proc_splitted[0][0])
    if first_record_date < start_date:
        start_less_than_2009 = 1     
    if first_record_date > start_date:
        time_dif = diff_month(datetime(first_record_date//100,first_record_date%100, 1), datetime(start_date//100,start_date%100, 1))
        current_record_date = start_date
        for i in range(time_dif):            
            patient_multi_hot_vectors.append(current_record_date)
            patient_multi_hot_vectors.extend([0]*len(proc_ccs_distinct_dict))
            current_record_date = int((datetime(current_record_date//100, current_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m")) 
    for i in range(len(line_proc_splitted)):
        if int(line_proc_splitted[i][0]) < start_date:
            continue
        for j in range(1, len(line_proc_splitted[i])): 
            if line_proc_splitted[i][j].replace("'","") == 'NOCODE' or line_proc_splitted[i][j].replace("'","")[:3] == 'EOV' or line_proc_splitted[i][j].replace("'","").replace(' ','')=='':
                # pdb.set_trace()
                #patient_multi_hot_vectors.append(int(line_proc_splitted[i][0]))
                #patient_multi_hot_vectors.extend([0]*len(proc_ccs_distinct_dict))                                
                continue
            elif line_proc_splitted[i][j].replace("'","") not in proc_cd_to_ccs_dict:
                # pdb.set_trace()
                if '-1000_ccs_proc' in proc_ccs_distinct_dict:
                    proc_ccs_distinct_dict['-1000_ccs_proc'] =1
                else:
                    continue    
            elif math.isnan(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0]):
                if '-1000_ccs_proc' in proc_ccs_distinct_dict:
                    proc_ccs_distinct_dict['-1000_ccs_proc'] =1
                else:
                    continue                
            elif (str(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0]) +'_ccs_proc') in proc_ccs_distinct_dict:
                proc_ccs_distinct_dict[(str(proc_cd_to_ccs_dict[line_proc_splitted[i][j].replace("'","")][0]) +'_ccs_proc')] =1
            # else:
            #     pdb.set_trace()  
            #     print('warning')  
        current_multi_hot_vec = dict(collections.OrderedDict(sorted(proc_ccs_distinct_dict.items()))).values()                   
        patient_multi_hot_vectors.append(int(line_proc_splitted[i][0]))
        patient_multi_hot_vectors.extend(current_multi_hot_vec)
        proc_ccs_distinct_dict = dict.fromkeys(proc_ccs_distinct_dict, 0)
    # pdb.set_trace()    
    last_record_date = int(line_proc_splitted[-1][0])     
    if last_record_date < end_date:
        time_dif = diff_month(datetime(end_date//100,end_date%100, 1), datetime(last_record_date//100,last_record_date%100, 1))  
        for i in range(time_dif):
            last_record_date = int((datetime(last_record_date//100, last_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
            patient_multi_hot_vectors.append(last_record_date)
            patient_multi_hot_vectors.extend([0]*len(proc_ccs_distinct_dict))
    # pdb.set_trace()        
    return patient_multi_hot_vectors, start_less_than_2009


def create_multihot_diags(line_diag
                            , icd_to_ccs_dict
                            , ccs_distinct_dict
                            ):
    # pdb.set_trace()    
    start_date = 200901
    end_date = 202006
    start_less_than_2009 = 0
    line_diag_splitted = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]            
    patient_multi_hot_vectors = []
    patient_multi_hot_vectors.append(float(line_diag[0]))
    if len(line_diag_splitted) == 0:    
        current_record_date = start_date
        while current_record_date <= end_date:            
            patient_multi_hot_vectors.append(current_record_date)
            patient_multi_hot_vectors.extend([0]*len(ccs_distinct_dict))
            current_record_date = int((datetime(current_record_date//100, current_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
        # pdb.set_trace()    
        return patient_multi_hot_vectors, start_less_than_2009
    first_record_date = int(line_diag_splitted[0][0])
    if first_record_date < start_date:
        start_less_than_2009 = 1    
    if first_record_date > start_date:
        time_dif = diff_month(datetime(first_record_date//100,first_record_date%100, 1), datetime(2009,1, 1 ))
        current_record_date = start_date
        for i in range(time_dif):            
            patient_multi_hot_vectors.append(current_record_date)
            patient_multi_hot_vectors.extend([0]*len(ccs_distinct_dict))
            current_record_date = int((datetime(current_record_date//100, current_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
    
    for i in range(len(line_diag_splitted)):
        if int(line_diag_splitted[i][0]) < start_date:
            continue        
        for j in range(1, len(line_diag_splitted[i])): 
            if line_diag_splitted[i][j].replace("'","") == 'NOCODE' or line_diag_splitted[i][j].replace("'","")[:3] == 'EOV':
                #patient_multi_hot_vectors.append(int(line_diag_splitted[i][0]))
                #patient_multi_hot_vectors.extend([0]*len(ccs_distinct_dict))                
                # pdb.set_trace()
                continue
            elif line_diag_splitted[i][j].replace("'","") not in icd_to_ccs_dict:
                # pdb.set_trace()
                if '-1000_ccs_diag' in ccs_distinct_dict:
                    ccs_distinct_dict['-1000_ccs_diag'] =1
                else:
                    continue    
            elif math.isnan(icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0]):
                if '-1000_ccs_diag' in ccs_distinct_dict:
                    ccs_distinct_dict['-1000_ccs_diag'] =1
                else:
                    continue                
            elif (str(icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0])+'_ccs_diag') in ccs_distinct_dict:
                ccs_distinct_dict[(str(icd_to_ccs_dict[line_diag_splitted[i][j].replace("'","")][0])+'_ccs_diag')] =1
            # else:
            #     pdb.set_trace()   
            #     print('warning') 
        # pdb.set_trace()
        current_multi_hot_vec = dict(collections.OrderedDict(sorted(ccs_distinct_dict.items()))).values()    
        patient_multi_hot_vectors.append(int(line_diag_splitted[i][0]))
        patient_multi_hot_vectors.extend(current_multi_hot_vec)
        ccs_distinct_dict = dict.fromkeys(ccs_distinct_dict, 0)

    # pdb.set_trace()    
    last_record_date = int(line_diag_splitted[-1][0])     
    if last_record_date < end_date:
        time_dif = diff_month(datetime(end_date//100,end_date%100, 1), datetime(last_record_date//100,last_record_date%100, 1))  
        for i in range(time_dif):
            last_record_date = int((datetime(last_record_date//100, last_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
            patient_multi_hot_vectors.append(last_record_date)
            patient_multi_hot_vectors.extend([0]*len(ccs_distinct_dict))
    # pdb.set_trace()        
    return patient_multi_hot_vectors, start_less_than_2009

def create_multihot_meds(line_med
                        , distinct_tcgpid_2digit_dict
                        , tcgpi_num_digits
                            ):
    # pdb.set_trace()
    start_date = 200901
    end_date = 202006    
    start_less_than_2009 = 0
    line_med_splitted = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]            
    patient_multi_hot_vectors = []
    patient_multi_hot_vectors.append(float(line_med[0]))
    if len(line_med_splitted) == 0:    
        current_record_date = start_date
        while current_record_date <= end_date:            
            patient_multi_hot_vectors.append(current_record_date)
            patient_multi_hot_vectors.extend([0]*len(distinct_tcgpid_2digit_dict))
            current_record_date = int((datetime(current_record_date//100, current_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
        # pdb.set_trace()    
        return patient_multi_hot_vectors, start_less_than_2009
    # pdb.set_trace()    
    first_record_date = int(line_med_splitted[0][0])
    if first_record_date < start_date:
        start_less_than_2009 = 1     
    if first_record_date > start_date:
        # pdb.set_trace()
        time_dif = diff_month(datetime(first_record_date//100,first_record_date%100, 1), datetime(2009,1, 1 ))
        current_record_date = start_date
        for i in range(time_dif):            
            patient_multi_hot_vectors.append(current_record_date)
            patient_multi_hot_vectors.extend([0]*len(distinct_tcgpid_2digit_dict))
            current_record_date = int((datetime(current_record_date//100, current_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
    # pdb.set_trace()
    for i in range(len(line_med_splitted)):
        if int(line_med_splitted[i][0]) < start_date:
            continue 
        for j in range(1, len(line_med_splitted[i])): 
            if (line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]+'_tcgp_2digit') in distinct_tcgpid_2digit_dict:           
                distinct_tcgpid_2digit_dict[(line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]+'_tcgp_2digit')] = 1
            elif line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] == 'NO' or line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]=='EO':
                continue
            # else:
            #     pdb.set_trace()   
            #     print('test')         
        current_multi_hot_vec = dict(collections.OrderedDict(sorted(distinct_tcgpid_2digit_dict.items()))).values()    
        patient_multi_hot_vectors.append(int(line_med_splitted[i][0]))
        # if int(line_med_splitted[i][0]) >= 201205:
        #     pdb.set_trace()
        patient_multi_hot_vectors.extend(current_multi_hot_vec)
        distinct_tcgpid_2digit_dict = dict.fromkeys(distinct_tcgpid_2digit_dict, 0)
        # pdb.set_trace()

        # print('test')  
    # pdb.set_trace()    
    last_record_date = int(line_med_splitted[-1][0])     
    if last_record_date < end_date:
        time_dif = diff_month(datetime(end_date//100,end_date%100, 1), datetime(last_record_date//100,last_record_date%100, 1))  
        for i in range(time_dif):
            last_record_date = int((datetime(last_record_date//100, last_record_date%100,1) + relativedelta(months=+1)).strftime("%Y%m"))
            patient_multi_hot_vectors.append(last_record_date)
            patient_multi_hot_vectors.extend([0]*len(distinct_tcgpid_2digit_dict))
    # pdb.set_trace()        
    return patient_multi_hot_vectors, start_less_than_2009

def reformat_to_multihot(meds_file
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
    index_to_calc_age = 2020
    num_patinet_before_2009 = 0
    num_time_steps = 138 # From 200901 to 202006 there are 138 months

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
    ccs_distinct_dict[-math.inf] = 0
    
    icd_to_ccs = dim_diags[['DIAG_CD', 'CCS_CATGRY']]  
    icd_to_ccs_dict =  icd_to_ccs.set_index('DIAG_CD').T.to_dict('list') 

    distinct_tcgpid = pd.read_csv(distinct_tcgpid_file)
    distinct_tcgpid_digits = distinct_tcgpid['TCGPI_ID'].str[0:tcgpi_num_digits].dropna().unique()
    distinct_tcgpid_digits_dict = {}
    for i in range(len(distinct_tcgpid_digits)):
        distinct_tcgpid_digits_dict[distinct_tcgpid_digits[i]] = 0 
    counter = 0
    # pdb.set22_trace()
    with open(meds_file) as meds_filename, open(diags_file) as diags_filename, open(procs_file) as procs_filename, open(demogs_file) as demogs_filename, open('outputs/'+fold_name+'_medications_multihot.csv','w') as meds_multihot_file, open('outputs/'+fold_name+'_diagnoses_multihot.csv','w') as diags_multihot_file, open('outputs/'+fold_name+'_procedures_multihot.csv','w') as procs_multihot_file, open('outputs/'+fold_name+'_demographics_multihot.csv','w') as demogs_multihot_file:
        demogs_header = next(demogs_filename)
        counter = 0
        for line_med in meds_filename:
            if counter %10000 ==0:
                print('Finished analyzing data of {} patients.'.format(counter))
            counter += 1    
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
            # med_dim = 94
            # diag_dim = 284
            # proc_dim = 243
            pdb.set_trace()
            multi_hot_meds_dict, start_less_than_2009_m = create_multihot_meds(line_med, distinct_tcgpid_digits_dict, tcgpi_num_digits)      
            multi_hot_diags_dict, start_less_than_2009_d = create_multihot_diags(line_diag, icd_to_ccs_dict, ccs_distinct_dict)
            multi_hot_procs_dict, start_less_than_2009_p = create_multihot_procs(line_proc, proc_cd_to_ccs_dict, proc_ccs_distinct_dict)
            
            if start_less_than_2009_m==1 or start_less_than_2009_d==1 or start_less_than_2009_p==1:
                num_patinet_before_2009+=1
            # pdb.set_trace()
            current_patient_demog = demog_creator(float(line_demog[dob_idx]), float(line_demog[sex_idx]), np.reshape(multi_hot_meds_dict[1:], (num_time_steps,-1))[:,0])
            
            current_patient_demog_normalized = normalize_demogs(current_patient_demog)
            current_patient_demogs_reshaped = np.reshape(current_patient_demog,(1,-1))

            
            if not(multi_hot_meds_dict[0] == multi_hot_diags_dict[0] == multi_hot_procs_dict[0] == float(line_demog[0])):
                pdb.set_trace()
                print('Error: enrolid mistmatch')
            # meds_multihot_file.write(str(float(line_med[0]) ))
            # meds_multihot_file.write('\n')
            if len(multi_hot_meds_dict) != 13111 or len(multi_hot_diags_dict) != 39331 or len(multi_hot_procs_dict) != 33673 or len(current_patient_demogs_reshaped[0]) != 414:
                pdb.set_trace()
            # pdb.set_trace()

            meds_multihot_file.write(','.join(map(repr, multi_hot_meds_dict)))
            meds_multihot_file.write('\n')

            diags_multihot_file.write(','.join(map(repr, multi_hot_diags_dict)))
            diags_multihot_file.write('\n')

            procs_multihot_file.write(','.join(map(repr, multi_hot_procs_dict)))
            procs_multihot_file.write('\n')

            demogs_multihot_file.write(str(float(line_demog[0])))
            demogs_multihot_file.write(',')
            demogs_multihot_file.write(','.join(map(repr, current_patient_demogs_reshaped[0])))
            demogs_multihot_file.write('\n')
    print('Number of patinets with records before 2009 is: {}'.format(num_patinet_before_2009))
    with open('outputs/num_patinet_before_2009.csv','w') as f:
        f.write('Number of patinets with records before 2009 is\n')
        f.write(str(num_patinet_before_2009))

def reformat_to_multihot_with_feature_selection(meds_file
                , diags_file
                , procs_file
                , demogs_file
                , dim_diags_file
                , dim_procs_ccs_file
                , distinct_tcgpid_file    
                , tcgpi_num_digits     
                , fold_name     
                , selected_features_meds
                , selected_features_diags
                , selected_features_procs
                ):
    # pdb.set_trace()
    # Read selected features
    selected_meds = pd.read_csv(selected_features_meds)
    selected_diags = pd.read_csv(selected_features_diags)
    selected_procs = pd.read_csv(selected_features_procs)


    ccs_distinct_dict = {}
    for i in range(len(selected_diags)):
        ccs_distinct_dict[selected_diags['Feature'][i]] = 0 
    # ccs_distinct_dict[-math.inf] = 0

    # pdb.set_trace()
    distinct_tcgpid_digits_dict = {}
    for i in range(len(selected_meds)):
        distinct_tcgpid_digits_dict[selected_meds['Feature'][i]] = 0

    proc_ccs_distinct_dict = {}
    for i in range(len(selected_procs)):
        proc_ccs_distinct_dict[selected_procs['Feature'][i]] = 0 
    # proc_ccs_distinct_dict[-math.inf] = 0
    # pdb.set_trace()
    dob_idx = 1
    sex_idx = 2
    label_idx = 6
    index_to_calc_age = 2020
    num_patinet_before_2009 = 0
    num_time_steps = 138 # From 200901 to 202006 there are 138 months

    # 
    dim_procs_ccs = pd.read_csv(dim_procs_ccs_file)
    # proc_ccs_distinct = dim_procs_ccs['ccs'].dropna().unique()
    # proc_ccs_distinct_dict = {}
    # for i in range(len(proc_ccs_distinct)):
    #     proc_ccs_distinct_dict[proc_ccs_distinct[i]] = 0 
    # proc_ccs_distinct_dict[-math.inf] = 0
    # pdb.set_trace()
    proc_cd_to_ccs = dim_procs_ccs[['proccd', 'ccs']]  
    proc_cd_to_ccs_dict =  proc_cd_to_ccs.set_index('proccd').T.to_dict('list') 


    dim_diags = pd.read_csv(dim_diags_file)
    # ccs_distinct = dim_diags['CCS_CATGRY'].dropna().unique()
    # ccs_distinct_dict = {}
    # for i in range(len(ccs_distinct)):
    #     ccs_distinct_dict[ccs_distinct[i]] = 0 
    # ccs_distinct_dict[-math.inf] = 0
    # pdb.set_trace()
    icd_to_ccs = dim_diags[['DIAG_CD', 'CCS_CATGRY']]  
    icd_to_ccs_dict =  icd_to_ccs.set_index('DIAG_CD').T.to_dict('list') 

    # distinct_tcgpid = pd.read_csv(distinct_tcgpid_file)
    # distinct_tcgpid_digits = distinct_tcgpid['TCGPI_ID'].str[0:tcgpi_num_digits].dropna().unique()
    # distinct_tcgpid_digits_dict = {}
    # for i in range(len(distinct_tcgpid_digits)):
    #     distinct_tcgpid_digits_dict[distinct_tcgpid_digits[i]] = 0 
    # counter = 0
    # pdb.set_trace()
    with open(meds_file) as meds_filename, open(diags_file) as diags_filename, open(procs_file) as procs_filename, open(demogs_file) as demogs_filename, open('outputs/'+fold_name+'_medications_multihot.csv','w') as meds_multihot_file, open('outputs/'+fold_name+'_diagnoses_multihot.csv','w') as diags_multihot_file, open('outputs/'+fold_name+'_procedures_multihot.csv','w') as procs_multihot_file, open('outputs/'+fold_name+'_demographics_multihot.csv','w') as demogs_multihot_file:
        demogs_header = next(demogs_filename)

        distinct_tcgpid_digits_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_digits_dict.items())))
        ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(ccs_distinct_dict.items())))
        proc_ccs_distinct_dict_sorted = dict(collections.OrderedDict(sorted(proc_ccs_distinct_dict.items())))
        
        meds_multihot_file.write('ENROLID, '+ 'date, '+ (','.join([*distinct_tcgpid_digits_dict_sorted.keys()])))
        meds_multihot_file.write('\n')

        diags_multihot_file.write('ENROLID, '+ 'date, '+ (','.join([*ccs_distinct_dict_sorted.keys()])))
        diags_multihot_file.write('\n')       


        procs_multihot_file.write('ENROLID, '+ 'date, '+ (','.join([*proc_ccs_distinct_dict_sorted.keys()])))
        procs_multihot_file.write('\n')                
 


        counter = 0
        for line_med in meds_filename:
            if counter %10000 ==0:
                print('Finished analyzing data of {} patients.'.format(counter))
            counter += 1    
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
            # med_dim = 94
            # diag_dim = 284
            # proc_dim = 243
            # pdb.set_trace()
            multi_hot_meds_dict, start_less_than_2009_m = create_multihot_meds(line_med, distinct_tcgpid_digits_dict, tcgpi_num_digits)      
            multi_hot_diags_dict, start_less_than_2009_d = create_multihot_diags(line_diag, icd_to_ccs_dict, ccs_distinct_dict)
            multi_hot_procs_dict, start_less_than_2009_p = create_multihot_procs(line_proc, proc_cd_to_ccs_dict, proc_ccs_distinct_dict)
            
            if start_less_than_2009_m==1 or start_less_than_2009_d==1 or start_less_than_2009_p==1:
                num_patinet_before_2009+=1
            # pdb.set_trace()
            current_patient_demog = demog_creator(float(line_demog[dob_idx]), float(line_demog[sex_idx]), np.reshape(multi_hot_meds_dict[1:], (num_time_steps,-1))[:,0])
            
            current_patient_demog_normalized = normalize_demogs(current_patient_demog)
            current_patient_demogs_reshaped = np.reshape(current_patient_demog,(1,-1))

            
            if not(multi_hot_meds_dict[0] == multi_hot_diags_dict[0] == multi_hot_procs_dict[0] == float(line_demog[0])):
                pdb.set_trace()
                print('Error: enrolid mistmatch')
            # meds_multihot_file.write(str(float(line_med[0]) ))
            # meds_multihot_file.write('\n')
            if len(multi_hot_meds_dict) !=  ((len(distinct_tcgpid_digits_dict)+1) * num_time_steps)+1 or len(multi_hot_diags_dict) != ((len(ccs_distinct_dict)+1) * num_time_steps)+1 or len(multi_hot_procs_dict) != ((len(proc_ccs_distinct_dict)+1) * num_time_steps)+1 or len(current_patient_demogs_reshaped[0]) != 414:
                pdb.set_trace()
            # pdb.set_trace()
            
            meds_multihot_file.write(','.join(map(repr, multi_hot_meds_dict)))
            meds_multihot_file.write('\n')

            diags_multihot_file.write(','.join(map(repr, multi_hot_diags_dict)))
            diags_multihot_file.write('\n')

            procs_multihot_file.write(','.join(map(repr, multi_hot_procs_dict)))
            procs_multihot_file.write('\n')

            demogs_multihot_file.write(str(float(line_demog[0])))
            demogs_multihot_file.write(',')
            demogs_multihot_file.write(','.join(map(repr, current_patient_demogs_reshaped[0])))
            demogs_multihot_file.write('\n')
    print('Number of patinets with records before 2009 is: {}'.format(num_patinet_before_2009))
    with open('outputs/num_patinet_before_2009.csv','w') as f:
        f.write('Number of patinets with records before 2009 is\n')
        f.write(str(num_patinet_before_2009))

