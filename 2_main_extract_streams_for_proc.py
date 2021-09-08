import pdb
import argparse
import sys
import os
import utils.extract_streams_for_proc as ext_strs
import logging

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort", type=str, default='oud_yes', choices = ['oud_yes','oud_no'])    
parser.add_argument("--display_step", type=int, default=1000000)    
parser.add_argument("--logging_milestone", type=int, default=1000)    

if  parser.parse_args().cohort == "oud_yes":
    parser.add_argument("--procs_rawdata_filename", type=str, default='data/oud_yes_icd_presc_based_procedures_view_sorted.csv')    
elif parser.parse_args().cohort == "oud_no":    
    parser.add_argument("--procs_rawdata_filename", type=str, default='data/oud_no_icd_presc_based_procedures_view_sorted.csv')    

args = parser.parse_args()

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_extract_streams_for_proc.log', filemode = 'a')




ext_strs.extract_procedures(args.procs_rawdata_filename 
                                    , args.cohort
                                    , args.display_step
                                    , args.logging_milestone)

logging.info('Extracting medications, diagnoses and procedures streams successfully completed!')
logging.info('================================')
