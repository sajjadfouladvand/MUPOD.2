import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import *
import pdb
import os
import time
import logging

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_extract_streams_for_proc.log', filemode = 'a')


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def vdates_syncher(current_patient_vdates):
    # pdb.set_trace()
    current_patient_vdates_synched = []
    current_patient_vdates_synched.append(int(current_patient_vdates[0]))
    for i in range(1, len(current_patient_vdates)):
        current_vdate= int(current_patient_vdates[i])
        prev_vdate= int(current_patient_vdates[i-1])
        diff_times = diff_month(datetime(current_vdate//100,current_vdate%100, 1 ), datetime(prev_vdate//100,prev_vdate%100, 1))
        if diff_times <2:
            current_patient_vdates_synched.append(current_vdate)
        else:
            for j in range(diff_times-1):   
                # pdb.set_trace()
                current_patient_vdates_synched.append(int((datetime(prev_vdate//100, prev_vdate%100,1) + relativedelta(months=+1)).strftime("%Y%m")))
                prev_vdate=current_patient_vdates_synched[-1]
            current_patient_vdates_synched.append(current_vdate)
    return  current_patient_vdates_synched 



def extract_procedures(procs_rawdata_filename
                    , cohort
                    , display_step
                    , logging_milestone):
    start_time = time.time()    
    # ==== Some constant containing field indexes
    enrolid_idx = 0
    svcdate_idx = 1
    proccd_idx = 2
    end_of_proc_file = False
    #========================

    start_time = time.time()
    with open(procs_rawdata_filename) as f:
        total_num_records = sum(1 for line in f)

    with open(procs_rawdata_filename) as procs_raw_file, open('outputs/'+cohort+'_procedures_for_proc.csv', 'w') as procedures_file:
        procs_header = next(procs_raw_file)        
        #=== Reading first line of procedures 
        line_proc = procs_raw_file.readline().replace('\n','').replace("'","").rstrip('\n\\n\r\\r')
        while line_proc.split(',')[enrolid_idx] ==  'NULL' or line_proc.split(',')[enrolid_idx] ==  '':
            line_proc = procs_raw_file.readline().replace('\n','').replace("'","").rstrip('\n\\n\r\\r')
        line_proc = line_proc.split(',')

        #==== While not end of the procedures file
        line_counter = 0
        patient_num = 0
        while end_of_proc_file == False:
            if line_counter%display_step==0:
                print('Finished processing {} procedures records out of {} records.'.format(line_counter,total_num_records))
            #==== Reading procedures visits for current patients
            if line_proc[enrolid_idx] == 'NULL':
                pdb.set_trace()            
            current_enrolid_proc = float(line_proc[enrolid_idx])
            current_patient_procs = []
            while current_enrolid_proc == float(line_proc[enrolid_idx]):
                current_patient_procs.append(line_proc)
                line_proc = procs_raw_file.readline().replace('\n','').replace("'","").rstrip('\n\\n\r\\r')
                line_counter+=1
                if line_counter%display_step==0:
                    print('Finished processing {} procedures records out of {} records.'.format(line_counter,total_num_records))                
                if line_proc == '' or line_proc == ['']:
                    end_of_proc_file = True
                    break
                line_proc = line_proc.split(',')
                while line_proc[enrolid_idx] == 'NULL' or line_proc[enrolid_idx] == '':
                    line_proc = procs_raw_file.readline().replace('\n','').replace("'","").rstrip('\n\\n\r\\r')
                    line_counter+=1
                    if line_counter%display_step==0:
                        print('Finished processing {} procedures records out of {} records.'.format(line_counter,total_num_records))                    
                    if line_proc == '' or line_proc == ['']:
                        end_of_proc_file = True
                        break                    
                    line_proc = line_proc.split(',')  
                if line_proc == '' or line_proc == ['']:                   
                    end_of_proc_file = True
                    break                                            
            current_patient_procs_ar = np.array(current_patient_procs, dtype='U')       
            #==== Visit dates in procedures stream
            if len(current_patient_procs_ar) >0:
                current_vdate_procs = current_patient_procs_ar[:,svcdate_idx]
            else:
                current_vdate_procs = []   

            current_patient_vdates = current_vdate_procs
            current_patient_vdates_str_nodash = np.char.replace(current_patient_vdates, '-','')
            current_patient_vdates = np.char.ljust(current_patient_vdates_str_nodash, width=6)        
            current_patient_vdates = np.sort(np.unique(current_patient_vdates))
            
            #==== Fill the gaps between visit dates 
            current_patient_vdates_synched = vdates_syncher(current_patient_vdates)
                        
            current_patient_procs_stream = [current_enrolid_proc]
            for j in range(len(current_patient_vdates_synched)):
                current_vdate = current_patient_vdates_synched[j]
                current_vdate_formatted = str(current_vdate)[:4] + '-' + str(current_vdate)[4:]

                if len(current_patient_procs_ar) >0:
                    current_vdate_procs_codes = current_patient_procs_ar[np.where(np.logical_and(np.char.ljust(current_patient_procs_ar[:,svcdate_idx], width = 7) == current_vdate_formatted, current_patient_procs_ar[:,proccd_idx] != ''))][:,proccd_idx]
                else:
                    current_vdate_procs_codes = []
           
                current_patient_procs_stream.append(current_vdate)
                
                #==== Storing procedures of the current month 
                if len(current_vdate_procs_codes) == 0:
                    current_patient_procs_stream.append('NOCODE')
                else:    
                    current_patient_procs_stream.extend(current_vdate_procs_codes)                
                current_patient_procs_stream.append('EOV')
            procedures_file.write(','.join(map(repr, current_patient_procs_stream)))
            procedures_file.write('\n')
            patient_num += 1
            if patient_num%logging_milestone==0:
                logging.info('Completed extracting and writing the procedures stream for {} patient with ENROLID = {}.'.format(cohort, current_enrolid_proc))            
    print('Finished processing all {} of procedures records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
    logging.info('Finished processing all {} of procedures records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
    logging.info('================================')    
    return 0  


