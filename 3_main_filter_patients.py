import pdb
import argparse
import sys
import os
import utils.filter_patinets as flt_pts


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort", type=str, default='oud_yes', choices = ['oud_yes','oud_no'])    
parser.add_argument("--display_step", type=int, default=100000)    
parser.add_argument("--min_month_available", type=int, default=12)    
parser.add_argument("--min_num_opioid", type=int, default=3)    
parser.add_argument("--prediction_win_size", type=int, default=6)    


if  parser.parse_args().cohort == "oud_yes":
    parser.add_argument("--meds_rawdata_filename", type=str, default='outputs/oud_yes_medications.csv')    
    parser.add_argument("--diags_rawdata_filename", type=str, default='outputs/oud_yes_diagnoses.csv')
    parser.add_argument("--procs_rawdata_filename", type=str, default='outputs/oud_yes_procedures.csv')    
    parser.add_argument("--demogs_rawdata_filename", type=str, default='data/sampled_oud_yes_icd_presc_based_demographics_view.csv')       
    # parser.add_argument("--metadata_rawdata_filename", type=str, default='data/oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3_sampled.csv')    
    args = parser.parse_args()
    flt_pts.filter_patients_positives(args.meds_rawdata_filename 
                                        , args.diags_rawdata_filename     
                                        , args.procs_rawdata_filename     
                                        , args.demogs_rawdata_filename     
                                        # , args.metadata_rawdata_filename
                                        , args.cohort
                                        , args.min_month_available
                                        , args.min_num_opioid 
                                        , args.prediction_win_size
                                        , args.display_step)
elif parser.parse_args().cohort == "oud_no":    
    parser.add_argument("--meds_rawdata_filename", type=str, default='outputs/oud_no_medications.csv')    
    parser.add_argument("--diags_rawdata_filename", type=str, default='outputs/oud_no_diagnoses.csv')
    parser.add_argument("--procs_rawdata_filename", type=str, default='outputs/oud_no_procedures.csv')    
    parser.add_argument("--demogs_rawdata_filename", type=str, default='data/sampled_oud_no_icd_presc_based_demographics_view.csv') 
    # parser.add_argument("--metadata_rawdata_filename", type=str, default='data/oud_no_icd_presc_based_opioid_ratio_demogs_filtered3_sampled.csv')    
    args = parser.parse_args()
    flt_pts.filter_patients_negatives(args.meds_rawdata_filename 
                                        , args.diags_rawdata_filename     
                                        , args.procs_rawdata_filename     
                                        , args.demogs_rawdata_filename     
                                        # , args.metadata_rawdata_filename
                                        , args.cohort
                                        , args.min_month_available
                                        , args.min_num_opioid 
                                        , args.prediction_win_size
                                        , args.display_step)








