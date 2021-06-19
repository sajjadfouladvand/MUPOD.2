import pdb
import argparse
import sys
import os
import utils.extract_streams as ext_strs


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort", type=str, default='oud_yes', choices = ['oud_yes','oud_no'])    
parser.add_argument("--display_step", type=int, default=100000)    

if  parser.parse_args().cohort == "oud_yes":
    parser.add_argument("--meds_rawdata_filename", type=str, default='data/sampled_oud_yes_icd_presc_based_prescriptions_view_sorted.csv')    
    parser.add_argument("--diags_rawdata_filename", type=str, default='data/sampled_oud_yes_icd_presc_based_diagnoses_view_sorted.csv')
    parser.add_argument("--procs_rawdata_filename", type=str, default='data/sampled_oud_yes_icd_presc_based_procedures_view_sorted.csv')    
elif parser.parse_args().cohort == "oud_no":    
    parser.add_argument("--meds_rawdata_filename", type=str, default='data/sampled_oud_no_icd_presc_based_prescriptions_view_sorted.csv')    
    parser.add_argument("--diags_rawdata_filename", type=str, default='data/sampled_oud_no_icd_presc_based_diagnoses_view_sorted.csv')
    parser.add_argument("--procs_rawdata_filename", type=str, default='data/sampled_oud_no_icd_presc_based_procedures_view_sorted.csv')    

args = parser.parse_args()

ext_strs.extract_medications(args.meds_rawdata_filename 
                                    , args.cohort
                                    , args.display_step)

ext_strs.extract_diagnoses(args.diags_rawdata_filename 
                                    , args.cohort
                                    , args.display_step)

ext_strs.extract_procedures(args.procs_rawdata_filename 
                                    , args.cohort
                                    , args.display_step)

