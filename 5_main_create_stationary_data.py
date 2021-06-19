import pdb
import argparse
import sys
import os
import utils.stationary_data_functions as sta_funcs 


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()  


# TRVNORM files
parser.add_argument("--dim_diags_file", type=str, default='trvnorm_data/dim_diagnoses_trvnorm.csv')    
parser.add_argument("--dim_procs_ccs_file", type=str, default='trvnorm_data/dim_proceduresCCS_trvnorm.csv')    

parser.add_argument("--distinct_diagcd_file", type=str, default='trvnorm_data/distinct_diagcd_trvnorm.csv')
parser.add_argument("--distinct_tcgpid_file", type=str, default='trvnorm_data/distinct_tcgpid_trvnorm.csv')    

parser.add_argument("--tcgpi_num_digits", type=int, default=2)    
parser.add_argument("--fold_name", type=str, default='train')    


  

if  parser.parse_args().fold_name == "train":
       parser.add_argument("--meds_file", type=str, default='outputs/train_medications.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/train_diagnoses.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/train_procedures.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/train_demographics.csv') 
       args = parser.parse_args()
       sta_funcs.create_stationary(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   )
elif parser.parse_args().fold_name == "validation":
       parser.add_argument("--meds_file", type=str, default='outputs/validation_medications.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/validation_diagnoses.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/validation_procedures.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/validation_demographics.csv') 
       args = parser.parse_args()
       sta_funcs.create_stationary(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   )
elif parser.parse_args().fold_name == "test":
       parser.add_argument("--meds_file", type=str, default='outputs/test_medications.csv')    
       parser.add_argument("--diags_file", type=str, default='outputs/test_diagnoses.csv')
       parser.add_argument("--procs_file", type=str, default='outputs/test_procedures.csv')    
       parser.add_argument("--demogs_file", type=str, default='outputs/test_demographics.csv') 
       args = parser.parse_args()
       sta_funcs.create_stationary(args.meds_file
                                   , args.diags_file
                                   , args.procs_file
                                   , args.demogs_file
                                   , args.dim_diags_file
                                   , args.dim_procs_ccs_file
                                   , args.distinct_tcgpid_file
                                   , args.tcgpi_num_digits
                                   , args.fold_name
                                   )