import pandas as pd
import pdb


def sort_medications(meds_path='data/oud_yes_icd_presc_based_prescriptions_view.csv'):
    data = pd.read_csv(meds_path)
    data = data.sort_values(by=['ENROLID'])
    data.to_csv(meds_path[:-4]+'_sorted'+'.csv', index= False)

def sort_diagnoses(diags_path='data/oud_yes_icd_presc_based_diagnoses_view.csv'):
    data = pd.read_csv(diags_path)
    data = data.sort_values(by=['ENROLID'])
    data.to_csv(diags_path[:-4]+'_sorted'+'.csv', index= False)

def sort_procedures(procs_path='data/oud_yes_icd_presc_based_procedures_view.csv'):
    # pdb.set_trace()
    data = pd.read_csv(procs_path)
    data = data.sort_values(by=['ENROLID'])
    data.to_csv(procs_path[:-4]+'_sorted'+'.csv', index= False)
# pdb.set_trace()
# sort_medications(meds_path='data/oud_yes_icd_presc_based_prescriptions_view.csv')
# sort_medications(meds_path='data/oud_no_icd_presc_based_prescriptions_view.csv')
# sort_diagnoses(diags_path='data/oud_yes_icd_presc_based_diagnoses_view.csv')
sort_diagnoses(diags_path='data/oud_no_icd_presc_based_diagnoses_view.csv')
# sort_procedures(procs_path='data/oud_yes_icd_presc_based_procedures_view.csv')
# sort_procedures(procs_path='data/oud_no_icd_presc_based_procedures_view.csv')
print('All medications, diagnoses and procedures records for both OUD-yes and OUD-no has been sucessfully sorted. ')

