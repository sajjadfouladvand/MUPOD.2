Place the prescriptions, diagnoses, procedures and demographics data for both OUD-positive and OUD-negative cohorts under the "data" folder. Run the following script to sort the patients records (prescriptions, diagnoses and procedures) based on patient's ID:
```
python 1_sort_records.py
```
Run the follwoing script for both OUD-positive and OUD-negative to convert the data into an enrollee-time matrix X(P,T,F), where P is the complete set of enrollees in the data, T is the set of time steps between Jan 2009 and Dec 2018 (by month) and F is the feature set, each vector x_ij records the medications/diagnoses/procedures an enrollee p_i took at time t_j. Arguments ```cohort``` can be used to apply the script on OUD-positive (```---cohort oud_yes```) and OUD-negative (```--cohort oud_no```) cohorts.
```
python 2_main_extract_streams.py --cohort oud_yes
python 2_main_extract_streams.py --cohort oud_no
```
Run the follwoing scripts to filter patients based on: 1) Minimum number of months that the patinet has been in the data. The argument to tune this is ```min_month_available``` and the default value is is 12, 2) Minimum number of months that the patinet has been prescribed with at least one Opioid medication (other than Buprenorphine or Methadone). The argument to set this is ```min_num_opioid``` and the default value is 3, 3)Prediction window size. The argument value to tune this feature is prediction_win_size and the default value is 6.
```
3_main_filter_patients.py --cohort oud_no
3_main_filter_patients.py --cohort oud_yes
```
