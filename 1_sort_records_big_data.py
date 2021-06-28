import os
from csvsort import csvsort


csvsort('data/sampled_oud_yes_icd_presc_based_prescriptions_view.csv', [0], output_filename='data/sampled_oud_yes_icd_presc_based_prescriptions_view_sorted.csv', has_header=True) 
csvsort('data/sampled_oud_yes_icd_presc_based_diagnoses_view.csv', [0], output_filename='data/sampled_oud_yes_icd_presc_based_diagnoses_view_sorted.csv', has_header= True) 
csvsort('data/sampled_oud_yes_icd_presc_based_procedures_view.csv', [0], output_filename='data/sampled_oud_yes_icd_presc_based_procedures_view_sorted.csv', has_header= True) 