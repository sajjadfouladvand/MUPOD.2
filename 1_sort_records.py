import pdb
import numpy as np
from operator import itemgetter


def sort_medications(meds_path='data/oud_yes_icd_presc_based_prescriptions_view.csv'):
    enrolid_idx = 0
    dim=4
    num_inval=0
    data=[]
    line_counter = 0
    with open(meds_path) as data_file:
      header=next(data_file)  
      for line in data_file:
        line_counter+=1
        line=line.split(',')
        if line[enrolid_idx] == 'NULL' or line[enrolid_idx] == '' or line[enrolid_idx] == 'ENROLID':
            num_inval+=1
            continue
        line[0] = int(line[0])
        data.append(line[:dim])
    data_sorted = sorted(data,key=itemgetter(0), reverse=False)
    with open(meds_path[:-4]+'_sorted'+'.csv', 'w') as data_file:
        data_file.write(','.join(header.split(',')[:dim]).replace('\n',''))
        data_file.write('\n')
        for i in range(len(data_sorted)):
            data_file.write(','.join(map(repr, data_sorted[i])))
            data_file.write('\n')

    print('End of sorting for {}'.format(meds_path))
    print('The data included {} records with no ENROLID'.format(num_inval))
    return 0

def sort_diagnoses(diags_path='data/oud_yes_icd_presc_based_diagnoses_view.csv'):
    enrolid_idx = 0
    dim=4
    num_inval=0
    data=[]
    line_counter = 0
    with open(diags_path) as data_file:
      header=next(data_file)  
      for line in data_file:
        line_counter+=1
        line = line.rstrip().replace('"','').replace(' ','')
        line=line.split(',')
        if line[enrolid_idx] == 'NULL' or line[enrolid_idx] == '':
            num_inval+=1
            continue
        line[0] = int(line[0])
        data.append(line[:dim])
    data_sorted = sorted(data,key=itemgetter(0))
    with open(diags_path[:-4]+'_sorted'+'.csv', 'w') as data_file:
        data_file.write(','.join(header.split(',')[:dim]).replace('\n',''))
        data_file.write('\n')
        for i in range(len(data_sorted)):
            data_file.write(','.join(map(repr, data_sorted[i])))
            data_file.write('\n')
    print('End of sorting for {}'.format(diags_path))
    print('The data included {} records with no ENROLID'.format(num_inval))
    return 0

def sort_procedures(procs_path='data/oud_yes_icd_presc_based_procedures_view.csv'):
    enrolid_idx = 0
    dim=3
    num_inval=0
    data=[]
    line_counter = 0
    with open(procs_path) as data_file:
      header=next(data_file)  
      for line in data_file:
        line_counter+=1
        line = line.rstrip().replace('"','').replace(' ','')
        line=line.split(',')
        if line[enrolid_idx] == 'NULL' or line[enrolid_idx] == '':
            num_inval+=1
            continue
        line[0] = int(line[0])
        data.append(line[:dim])
    data_sorted = sorted(data,key=itemgetter(0))

    with open(procs_path[:-4]+'_sorted'+'.csv', 'w') as data_file:
        data_file.write(','.join(header.split(',')[:dim]).replace('\n',''))
        data_file.write('\n')
        for i in range(len(data_sorted)):
            data_file.write(','.join(map(repr, data_sorted[i])))
            data_file.write('\n')

    print('End of sorting for {}'.format(procs_path))
    print('The data included {} records with no ENROLID'.format(num_inval))
    return 0
# pdb.set_trace()
# sort_medications(meds_path='data/oud_yes_icd_presc_based_prescriptions_view.csv')
# sort_medications(meds_path='data/oud_no_icd_presc_based_prescriptions_view.csv')
# sort_diagnoses(diags_path='data/oud_yes_icd_presc_based_diagnoses_view.csv')
# sort_diagnoses(diags_path='data/oud_no_icd_presc_based_diagnoses_view.csv')
# sort_procedures(procs_path='data/oud_yes_icd_presc_based_procedures_view.csv')
sort_procedures(procs_path='data/oud_no_icd_presc_based_procedures_view.csv')
print('All medications, diagnoses and procedures records for both OUD-yes and OUD-no has been sucessfully sorted. ')
