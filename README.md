Place the prescriptions, diagnoses, procedures and demographics data for both OUD-positive and OUD-negative cohorts under the "data" folder. Run the following script to sort the patients records (prescriptions, diagnoses and procedures) based on patient's ID:
```
python 1_sort_records.py
```
Run the follwoing script for both OUD-positive and OUD-negative to convert the data into an enrollee-time matrix X(P,T,F), where P is the complete set of enrollees in the data, T is the set of time steps between Jan 2009 and Dec 2018 (by month) and F is the feature set, each vector x_ij records the medications/diagnoses/procedures an enrollee p_i took at time t_j. 
```
python 2_main_extract_streams.py --cohort oud_yes
python 2_main_extract_streams.py --cohort oud_no
```
