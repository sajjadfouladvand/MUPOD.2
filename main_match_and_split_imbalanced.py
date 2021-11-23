import pdb
import argparse
import sys
import os
import utils.imbl_data_functions as imb_funcs 


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--pos_to_negs_ratio", type=int, default=2)    
parser.add_argument("--display_step", type=int, default=10000)    
parser.add_argument("--prediction_win_size", type=int, default=6)  

parser.add_argument("--meds_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_medications_eligible.csv')    
parser.add_argument("--diags_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_diagnoses_eligible.csv')
parser.add_argument("--procs_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_procedures_eligible.csv')    
parser.add_argument("--demogs_oud_yes_filename", type=str, default='outputs/oud_yes_demographics_eligible.csv') 

parser.add_argument("--meds_oud_no_rawdata_filename", type=str, default='outputs/oud_no_medications_eligible.csv')    
parser.add_argument("--diags_oud_no_rawdata_filename", type=str, default='outputs/oud_no_diagnoses_eligible.csv')
parser.add_argument("--procs_oud_no_rawdata_filename", type=str, default='outputs/oud_no_procedures_eligible.csv')    
parser.add_argument("--demogs_oud_no_filename", type=str, default='outputs/oud_no_demographics_eligible.csv')              

parser.add_argument("--sex1_matching_map_path", type=str, default='imb_patients_map/enrolids_1_to_n_sex1.csv') 
parser.add_argument("--sex2_matching_map_path", type=str, default='imb_patients_map/enrolids_1_to_n_sex2.csv') 
parser.add_argument("--balanced_test_filename", type=str, default='outputs/test_stationary_normalized_features_filtered.csv') 


args = parser.parse_args()

# pdb.set_trace()

imb_funcs.create_test_imbalanced(args.meds_oud_yes_rawdata_filename
                                    , args.diags_oud_yes_rawdata_filename
                                    , args.procs_oud_yes_rawdata_filename
                                    , args.demogs_oud_yes_filename
                                    , args.meds_oud_no_rawdata_filename
                                    , args.diags_oud_no_rawdata_filename
                                    , args.procs_oud_no_rawdata_filename                                    
                                    , args.demogs_oud_no_filename
                                    , args.sex1_matching_map_path
                                    , args.sex2_matching_map_path
                                    , args.balanced_test_filename
                                    , args.pos_to_negs_ratio
                                    , args.prediction_win_size
                                    , args.display_step)
