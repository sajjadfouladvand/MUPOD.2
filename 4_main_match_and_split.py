import pdb
import argparse
import sys
import os
import utils.match_and_split as mch_splt 


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
matched = 0
parser.add_argument("--pos_to_negs_ratio", type=int, default=1)    
parser.add_argument("--train_ratio", type=int, default=0.8)
parser.add_argument("--validation_ratio", type=int, default=0.1)    
parser.add_argument("--display_step", type=int, default=100000)    
parser.add_argument("--matched", type=int, default=1)    
parser.add_argument("--prediction_win_size", type=int, default=3)  

parser.add_argument("--meds_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_medications_eligible.csv')    
parser.add_argument("--diags_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_diagnoses_eligible.csv')
parser.add_argument("--procs_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_procedures_eligible.csv')    
parser.add_argument("--demogs_oud_yes_filename", type=str, default='outputs/oud_yes_demographics_eligible.csv') 

parser.add_argument("--meds_oud_no_rawdata_filename", type=str, default='outputs/oud_no_medications_eligible.csv')    
parser.add_argument("--diags_oud_no_rawdata_filename", type=str, default='outputs/oud_no_diagnoses_eligible.csv')
parser.add_argument("--procs_oud_no_rawdata_filename", type=str, default='outputs/oud_no_procedures_eligible.csv')    
parser.add_argument("--demogs_oud_no_filename", type=str, default='outputs/oud_no_demographics_eligible.csv')              

args = parser.parse_args()
matched = mch_splt.cohort_matching( args.demogs_oud_yes_filename
                    , args.demogs_oud_no_filename
                    , args.pos_to_negs_ratio
                    )
mch_splt.split_train_validation_test(args.meds_oud_yes_rawdata_filename
                                    , args.diags_oud_yes_rawdata_filename
                                    , args.procs_oud_yes_rawdata_filename
                                    , args.demogs_oud_yes_filename
                                    , args.meds_oud_no_rawdata_filename
                                    , args.diags_oud_no_rawdata_filename
                                    , args.procs_oud_no_rawdata_filename                                    
                                    , args.demogs_oud_no_filename   
                                    , args.train_ratio
                                    , args.validation_ratio 
                                    , matched
                                    , args.prediction_win_size
                                    )

