import pdb
import argparse
import sys
import os
import utils.stationary_data_functions as sta_funcs 

'''
First use the fold_name parameter to covert train, validation and test sets seperately to stationary features.
Note, you have to run this script three times each time with different fold_name (train, validation, test).

After you finished converting ALL train, validation and test sets to stationary, run this script with 
the parameter normalization of min_max to normalize all train, validation and test stationary data.
'''

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()  


# TRVNORM files
parser.add_argument("--dim_diags_file", type=str, default='trvnorm_data/dim_diagnoses_trvnorm.csv')    
parser.add_argument("--dim_procs_ccs_file", type=str, default='trvnorm_data/dim_proceduresCCS_trvnorm.csv')    

parser.add_argument("--distinct_diagcd_file", type=str, default='trvnorm_data/distinct_diagcd_trvnorm.csv')
parser.add_argument("--distinct_tcgpid_file", type=str, default='trvnorm_data/distinct_tcgpid_trvnorm.csv')    

parser.add_argument("--tcgpi_num_digits", type=int, default=2)    
parser.add_argument("--fold_name", type=str, default='None')    
parser.add_argument("--normalization", type=str, default='None')    


  

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
elif   parser.parse_args().fold_name == "None": 
       print('WARNING: None of the train, validation or test sets has been selected to convert to stationary.')        

if     parser.parse_args().normalization == "min_max":
              sta_funcs.normalize_data_min_max( 'outputs/train_stationary.csv'
                                          , 'outputs/validation_stationary.csv'
                                          , 'outputs/test_stationary.csv'
                                          )    
elif   parser.parse_args().normalization == "None":
       print('Wanring: No method has been selected for normalization.')



