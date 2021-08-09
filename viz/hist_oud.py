import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib as mpl 
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

feature = '65'
sample_size = 1000
file_name = 'outputs/test_stationary.csv'
test_data=pd.read_csv(file_name, index_col='ENROLID')
file_name = 'outputs/validation_stationary.csv'
validation_data=pd.read_csv(file_name, index_col='ENROLID')
file_name = 'outputs/train_stationary.csv'
train_data=pd.read_csv(file_name, index_col='ENROLID')
file_name = 'sampled_' + str(sample_size) +'smpl'
data_all = pd.concat([train_data, test_data,validation_data])
data_pos = data_all[data_all['Label'] == 1]
data_neg = data_all[data_all['Label'] == 0]
# pdb.set_trace()
data_pos_sampled = data_pos.sample(n=sample_size, replace=False)
data_neg_sampled = data_neg.sample(n=sample_size, replace=False)
data = pd.concat([data_pos_sampled, data_neg_sampled])

data=data.sample(frac=1)

pdb.set_trace()
data_pos = data[data['Label']==1]
data_pos_filtered = data_pos["'65'"].to_numpy()
data_negs = data[data['Label']==0]
data_negs_filtered = data_negs["'65'"].to_numpy()


data_filtered_all = np.zeros((len(data_pos_filtered), 2))
data_filtered_all[:,0] = data_pos_filtered
data_filtered_all[:,1] = data_negs_filtered

#===== method 2
# fig, axes = plt.subplots(nrows=2, ncols=2)
# ax0, ax1, ax2, ax3 = axes.flatten()

# colors = ['red', 'blue']
# ax0.hist(data_filtered_all, n_bins, density=True, histtype='bar', color=colors, label=colors)
# ax0.legend(prop={'size': 10})
# ax0.set_title('bars with legend')
#========

bins = 30
# data = np.random.randn(1000, 2)
# pdb.set_trace()
colors = ['red','blue']
plt.hist(data_filtered_all, bins, histtype='bar', color=colors, stacked=False, fill=True, label=['OUD-positive patients','OUD-negative patients'])
plt.legend()
plt.xlabel('Opioid medications', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig("results/visualization_results/histogram.png", dpi=600)