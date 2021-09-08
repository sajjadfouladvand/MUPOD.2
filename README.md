<h1 style="font-size:60px;">1. Preprocessing</h1>

Place the prescriptions, diagnoses, procedures and demographics data for both OUD-positive and OUD-negative cohorts under the "data" folder. Run the following script to sort the patients records (prescriptions, diagnoses and procedures) based on patient's ID:
```
python 1_sort_records.py
```
As a result, sorted files are created under the "data/" directory. All sorted files are indicated by "_sorted" at the end of the file name. Run the follwoing script for both OUD-positive and OUD-negative to convert the data into an enrollee-time matrix X(P,T,F), where P is the complete set of enrollees in the data, T is the set of time steps between Jan 2009 and Dec 2018 (by month) and F is the feature set, each vector x_ij records the medications/diagnoses/procedures an enrollee p_i took at time t_j. Arguments ```cohort``` can be used to apply the script on OUD-positive (```---cohort oud_yes```) and OUD-negative (```--cohort oud_no```) cohorts.
```
python 2_main_extract_streams.py --cohort oud_yes
python 2_main_extract_streams.py --cohort oud_no
```
As a result medications, diagnoses and procedures streams are extracted and saved under the directory "outputs/". Run the follwoing scripts to filter patients based on: 1) Minimum number of months that the patinet has been in the data. The argument to tune this is ```min_month_available``` and the default value is is 12, 2) Minimum number of months that the patinet has been prescribed with at least one Opioid medication (other than Buprenorphine or Methadone). The argument to set this is ```min_num_opioid``` and the default value is 3, 3)Prediction window size. The argument value to tune this feature is prediction_win_size and the default value is 6.
```
3_main_filter_patients.py --cohort oud_no
3_main_filter_patients.py --cohort oud_yes
```
As a result, medications, diagnoses, procedures and demographics streams are filtered and only patients who met the above criteria will be included. The resulted files will be save under the directory "outputs/" and the new files are indicated with "_eligible" at the end of the file names. 

The Follwoing command is the next command that should be executed and it performs multiple tasks: 1) It first apply a cohort matching. This function matches OUD-negatives with OUD-positives based on patinets sex, date of birth, number of months they have been prescribed with at least one Opioid (other than Buprenorphine or Methadone) and the total number of months they have been in the data. Argument ```pos_to_negs_ratio``` can be utilized to set how many OUD-negative samples should be selected for each OUD-positive sample. Note, this code uses K-means clustering and anchor based methods to find best matches. 
```
python 4_main_match_and_split.py 
```
Here is a list of argument that can be used to apply different constrains:

``` train_ratio```: This can be used to define what portion of the data you prefer to use in the train set. The default value is 80%.

```matched```: If  ```matched=1``` then the matched negative cohort will be used to create train, validation and test. 

```prediction_win_size```: set the prediction window size. The default value is 6 months. For the OUD-positive patinets, all the data within a window of 6 month prior to the diagnoses date is erased. For the OUD-negative cohort, all the data within a window of 6 month prior to the patient's last record in the data will eb erased.

<h1 style="font-size:60px;">2. Classical Machine Learning Models</h1>
Run the following commands to create stationary train, validation and test data. Argument ```fold_name``` can be used to produce train, validation and test data seperately:

```
python 5_main_create_stationary_data.py --fold_name train
python 5_main_create_stationary_data.py --fold_name validation
python 5_main_create_stationary_data.py --fold_name test
```

After creating the stationary data, you can use the same script to normalize the train, validation and test data using min-max normalization:

```
python 5_main_create_stationary_data.py --normalization min_max
```
As a result, train, validation and test sets are normalized and saved under the "outputs/" directory. These files are indicated by "_stationary_normalized.csv" at the end of train, validation and test sets file names. At this point you can use the fillowing scripts to compute some basic statistics on the data:
```
python viz/visualize_stationary_data.py --compute_stat 1
```
For visualizing the data using tSNE method (pca is also available):
```
python viz/visualize_stationary_data.py --viz_method tsne
```
If you wish to see the frequency of the features you can run this comamnd. This script will produce bar diagrams of feature frequencies for medication, diagnoses and procedures seperately under the "results/visualization_results/":
```
python viz/visualize_stationary_data.py --plot_feature_dist_flag 1
```
The following command can then be used to perform feature selection using the frequencies calculated in the previous step. This will produce three files indicated by "__features_filtered.csv" under the "results/visualization_results/" directory for medications, diagnoses and procedures. Each file includes selected features. Furtheremore, this command will generate train, validation and test files after performing feature selection and the resulted files can be found under the "outputs/" directory and are indicated by "_stationary_normalized_features_filtered.csv" in their file names.  

```
python viz/visualize_stationary_data.py --feature_selection 1
```

Now you can run the follwoing command to train, validate and test classical machine learning models on the stationary data. Argument ```ml_model``` can be used to choose which ML model be used to do the predictions. The default value is ```--ml_model rf``` which applyes a random forest on the stationary data. Furtheremore, the argument "feature_selection" can be used to train and test the models using the data generated after performing feature selection in the previous step.
```
6_main_classical_ml_models.py --ml_model rf --feature_selection 1
```
The results will be stored under ```/results/classical_ml_models```. 

<h1 style="font-size:60px;">3. Long Short Term Memory</h1>

The first step is to shuffle the train, validation and test sets:

```
python main_multi_hot_shuffle.py --shuffle 1
```

The shuffled train, validation and test data are stored under the "outputs/" directory and indicated by "_shuffled.csv". Then, run the following command to format the train, validation and test data into multi-hot vectors and prepare them for training a Long Short Term Memory model:


```
python main_multi_hot_shuffle.py --fold_name train --feature_selection 1
python main_multi_hot_shuffle.py --fold_name validation --feature_selection 1
python main_multi_hot_shuffle.py --fold_name test --feature_selection 1
```

<h1 style="font-size:60px;">4. Transformer</h1>

<h1 style="font-size:60px;">4. MUPOD</h1>
