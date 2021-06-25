<h1 style="font-size:60px;">Preprocessing</h1>

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
Follwoing command performs multiple tasks: 1) It first apply a cohort matching. This function matches OUD-negatives with OUD-positives based on patinets sex, date of birth, number of months they have been prescribed with at least one Opioid (other than Buprenorphine or Methadone) and the total number of months they have been in the data. Argument ```pos_to_negs_ratio``` can be utilized to set how many OUD-negative samples should be selected for each OUD-positive sample. Note, this code uses cosine similarity to find best matches. 
```
python 4_main_match_and_split.py 
```
Here is a list of argument that can be used to apply different constrains:

``` train_ratio```: This can be used to define what portion of the data you prefer to use in the train set. The default value is 80%.

```matched```: If  ```matched=1``` then the matched negative cohort will be used to create train, validation and test. 

```prediction_win_size```: set the prediction window size. The default value is 6 months. For the OUD-positive patinets, all the data within a window of 6 month prior to the diagnoses date is erased. For the OUD-negative cohort, all the data within a window of 6 month prior to the patient's last record in the data will eb erased.
<h1 style="font-size:60px;">Classical Machine Learning Models</h1>
Run the following commands to create stationary train, validation and test data. Argument ```fold_name``` can be used to produce train, validation and test data seperately:
```
python 5_main_create_stationary_data.py --fold_name train
python 5_main_create_stationary_data.py --fold_name validation
python 5_main_create_stationary_data.py --fold_name test
```
After creating the stationary data, you can run the follwoing command to train, validate and test classical machine learning models on the stationary data. Argument ```ml_model``` can be used to choose which ML model be used to do the predictions. The default value is ```--ml_model rf``` which applyes a random forest on the stationary data. 
```
6_main_classical_ml_models.py 
```
The results will be stored under ```/results/classical_ml_models```. 
