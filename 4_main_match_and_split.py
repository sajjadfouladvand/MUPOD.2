import pdb
import argparse
import sys
import os
import utils.match_and_split as mch_splt 


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--pos_to_negs_ratio", type=int, default=1)    
parser.add_argument("--train_ratio", type=int, default=0.8)
parser.add_argument("--validation_ratio", type=int, default=0.1)    
parser.add_argument("--display_step", type=int, default=10000)    
parser.add_argument("--matched", type=int, default=1, choices =[0, 1])    
parser.add_argument("--prediction_win_size", type=int, default=6)  
parser.add_argument("--num_sample_pos", type=int, default=5000) 
parser.add_argument("--sampled_pos", type=int, default=0 ,choices =[0, 1])  
parser.add_argument("--k", type=int, default=100) 


parser.add_argument("--meds_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_medications_eligible.csv')    
parser.add_argument("--diags_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_diagnoses_eligible.csv')
parser.add_argument("--procs_oud_yes_rawdata_filename", type=str, default='outputs/oud_yes_procedures_eligible.csv')    
parser.add_argument("--demogs_oud_yes_filename", type=str, default='outputs/oud_yes_demographics_eligible.csv') 

parser.add_argument("--meds_oud_no_rawdata_filename", type=str, default='outputs/oud_no_medications_eligible.csv')    
parser.add_argument("--diags_oud_no_rawdata_filename", type=str, default='outputs/oud_no_diagnoses_eligible.csv')
parser.add_argument("--procs_oud_no_rawdata_filename", type=str, default='outputs/oud_no_procedures_eligible.csv')    
parser.add_argument("--demogs_oud_no_filename", type=str, default='outputs/oud_no_demographics_eligible.csv')              

args = parser.parse_args()

# pdb.set_trace()

# =======================
if args.sampled_pos == 1:
    demogs_oud_yes_filename_sampled = mch_splt.sampling_oud_yes_cohort(args.demogs_oud_yes_filename
                                    , args.num_sample_pos)

if args.matched == 1 and args.sampled_pos == 1:
    demogs_oud_no_filename_matched = mch_splt.cohort_matching_big_data(demogs_oud_yes_filename_sampled
                        , args.demogs_oud_no_filename
                        , args.pos_to_negs_ratio
                        , args.k
                        )
    mch_splt.split_train_validation_test(args.meds_oud_yes_rawdata_filename
                                        , args.diags_oud_yes_rawdata_filename
                                        , args.procs_oud_yes_rawdata_filename
                                        , demogs_oud_yes_filename_sampled
                                        , args.meds_oud_no_rawdata_filename
                                        , args.diags_oud_no_rawdata_filename
                                        , args.procs_oud_no_rawdata_filename                                    
                                        , demogs_oud_no_filename_matched   
                                        , args.train_ratio
                                        , args.validation_ratio 
                                        , args.prediction_win_size
                                        , args.display_step)
elif args.matched == 1 and args.sampled_pos == 0:
    demogs_oud_no_filename_matched = mch_splt.cohort_matching_big_data(args.demogs_oud_yes_filename
                        , args.demogs_oud_no_filename
                        , args.pos_to_negs_ratio
                        , args.k
                        )    
    mch_splt.split_train_validation_test(args.meds_oud_yes_rawdata_filename
                                        , args.diags_oud_yes_rawdata_filename
                                        , args.procs_oud_yes_rawdata_filename
                                        , args.demogs_oud_yes_filename
                                        , args.meds_oud_no_rawdata_filename
                                        , args.diags_oud_no_rawdata_filename
                                        , args.procs_oud_no_rawdata_filename                                    
                                        , demogs_oud_no_filename_matched   
                                        , args.train_ratio
                                        , args.validation_ratio 
                                        , args.prediction_win_size
                                        , args.display_step)
elif args.matched == 0 and args.sampled_pos == 1:
   
    mch_splt.split_train_validation_test(args.meds_oud_yes_rawdata_filename
                                        , args.diags_oud_yes_rawdata_filename
                                        , args.procs_oud_yes_rawdata_filename
                                        , demogs_oud_yes_filename_sampled
                                        , args.meds_oud_no_rawdata_filename
                                        , args.diags_oud_no_rawdata_filename
                                        , args.procs_oud_no_rawdata_filename                                    
                                        , args.demogs_oud_no_filename  
                                        , args.train_ratio
                                        , args.validation_ratio 
                                        , args.prediction_win_size
                                        , args.display_step)
elif args.matched == 0 and args.sampled_pos == 0:
   
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
                                        , args.prediction_win_size
                                        , args.display_step)

