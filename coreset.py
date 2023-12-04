import numpy as np
from apricot import FacilityLocationSelection
from apricot import FeatureBasedSelection
from apricot import MaxCoverageSelection
from apricot import SaturatedCoverageSelection
import os
import random

# return the coreset indices from apricot
def apricot_coreset(X, n):
    selector = FeatureBasedSelection(n,concave_func='sigmoid',optimizer='naive',verbose=True)
    selector.fit(X)
    return selector.ranking


# datasets = ['hhar', 'motion', 'shoaib', 'uci']
datasets = ['hhar']
# set the coreset factor 1%, 5%, 10%, 25%, 50%, 75%
core_factors = [0.1]


for dataset in datasets:
    dir_path = f'coreset_data/{dataset}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

count = 1
        
for dataset in datasets:
    for core_factor in core_factors:
        data = np.load(f'dataset/{dataset}/data_20_120.npy')
        label = np.load(f'dataset/{dataset}/label_20_120.npy')
        core_num = int(data.shape[0] * core_factor)
        data_dim2 =  data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        print(data_dim2)
        print(f"start generation for {core_factor*100}% {dataset}")

        # generation for coreset data
        idx = apricot_coreset(data_dim2, core_num)
        data_coreset = data[idx]
        label_coreset = label[idx]
        
        np.save(f'coreset_data/{dataset}/data_20_120_{core_num}_{100*core_factor}%_SCS{count}.npy',data_coreset)
        np.save(f'coreset_data/{dataset}/label_20_120_{core_num}_{100*core_factor}%_SCS{count}.npy',label_coreset)
        
        
        #generation for random subset data
        idx_random = random.sample(range(0,data.shape[0]),core_num)
        data_ram = data[idx_random]
        label_ram = label[idx_random]

        np.save(f'coreset_data/{dataset}/data_20_120_{core_num}_{100*core_factor}%_random{count}.npy',data_ram)
        np.save(f'coreset_data/{dataset}/label_20_120_{core_num}_{100*core_factor}%_random{count}.npy',label_ram)
        count = count +1
        
