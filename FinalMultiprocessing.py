# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:13:48 2024

@author: marjan
"""


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *

#-- Pytorch specific libraries import -----#
import torch
import torch.nn as nn
#device = torch.device("cuda:0")
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import tarfile
import os, requests
import time

class ChurnModel(nn.Module):
    def __init__(self, n_input_dim):
        super(ChurnModel, self).__init__()
        self.n_hidden1 = 720
        self.n_hidden2 = 720
        self.n_output = 1
        self.layer_1 = nn.Linear(n_input_dim, self.n_hidden1)
        self.layer_2 = nn.Linear(self.n_hidden1, self.n_hidden2)
        self.layer_out = nn.Linear(self.n_hidden2, self.n_output)
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(self.n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(self.n_hidden2)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        return x

def evaluate_model(model, test_loader):
     model.eval()  # Set the model to evaluation mode
     correct = 0
     total = 0
 
     with torch.no_grad():  # Disable gradient calculation for evaluation
         for xb, yb in test_loader:
             y_pred = model(xb)  # Get model predictions
             y_pred_tag = torch.round(y_pred)  # Round predictions to get binary outputs
             correct += (y_pred_tag.eq(yb).sum().item())  # Count correct predictions
             total += yb.size(0)  # Count total samples
 
     accuracy = correct / total  # Calculate accuracy
     return accuracy
 

def train_worker(epochs, model, loss_func, optimizer, evaluate_model, train, test, q, cuda_device):
    import torch
    
    print(f"process running on CUDA:{cuda_device}")
    device = torch.device(f"cuda:{cuda_device}")
    
    val_acc = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
    
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for xb, yb in test:
                xb, yb = xb.to(device), yb.to(device) #transferring the data to GPU
                y_pred = model(xb)  # Get model predictions
                y_pred_tag = torch.round(y_pred)  # Round predictions to get binary outputs
                correct += (y_pred_tag.eq(yb).sum().item())  # Count correct predictions
                total += yb.size(0)  # Count total samples
    
        v_acc = correct / total  # Calculate accuracy
        val_acc.append(v_acc)
    
    accuracy = max(val_acc)
    q.put(accuracy)
    return

if __name__ == '__main__':
    # The data shared for NMA projects is a subset of the full HCP dataset
    N_SUBJECTS = 100
    
    # The data have already been aggregated into ROIs from the Glasser parcellation
    N_PARCELS = 360
    
    # The acquisition parameters for all tasks were identical
    TR = 0.72  # Time resolution, in seconds
    
    # The parcels are matched across hemispheres with the same order
    HEMIS = ["Right", "Left"]
    
    # Each experiment was repeated twice in each subject
    RUNS   = ['LR','RL']
    N_RUNS = 2
    
    # There are 7 tasks. Each has a number of 'conditions'
    # TIP: look inside the data folders for more fine-graned conditions
    
    EXPERIMENTS = {
        'MOTOR'      : {'cond':['lf','rf','lh','rh','t','cue']},
        'WM'         : {'cond':['0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools']},
        'EMOTION'    : {'cond':['fear','neut']},
        'GAMBLING'   : {'cond':['loss','win']},
        'LANGUAGE'   : {'cond':['math','story']},
        'RELATIONAL' : {'cond':['match','relation']},
        'SOCIAL'     : {'cond':['ment','rnd']}
    }
    
    fname = "hcp_task.tgz"
    url = "https://osf.io/2y3fw/download"
    
    if not os.path.isfile(fname):
      try:
        r = requests.get(url)
      except requests.ConnectionError:
        print("!!! Failed to download data !!!")
      else:
        if r.status_code != requests.codes.ok:
          print("!!! Failed to download data !!!")
        else:
          with open(fname, "wb") as fid:
            fid.write(r.content)
            
    # The download cells will store the data in nested directories starting here:
    HCP_DIR = "./hcp_task"
    fname = "hcp_task.tgz"
    
    # open file
    with tarfile.open(fname) as tfile:
      # extracting file
      tfile.extractall('.')
    
    subjects = np.loadtxt(os.path.join(HCP_DIR, 'subjects_list.txt'), dtype='str')
    
    regions = np.load(f"{HCP_DIR}/regions.npy").T
    region_info = dict(
        name=regions[0].tolist(),
        network=regions[1],
        hemi=['Right']*int(N_PARCELS/2) + ['Left']*int(N_PARCELS/2),
    )
    
    all_ROI_names = region_info['name'][1:]
    err_txt = "Marji-joon trained without the 1st column"
    assert len(all_ROI_names) == 359, err_txt
    # print(all_ROI_names)
    
    def load_single_timeseries(subject, experiment, run, remove_mean=True):
      """Load timeseries data for a single subject and single run.
    
      Args:
        subject (str):      subject ID to load
        experiment (str):   Name of experiment
        run (int):          (0 or 1)
        remove_mean (bool): If True, subtract the parcel-wise mean (typically the mean BOLD signal is not of interest)
        # WHY?
    
      Returns
        ts (n_parcel x n_timepoint array): Array of BOLD data values
    
      """
      bold_run  = RUNS[run]
      bold_path = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{bold_run}"
      bold_file = "data.npy"
      ts = np.load(f"{bold_path}/{bold_file}")
      if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
      return ts
    
    
    # print(EXPERIMENTS)
    # start computes start time in terms of frames, divides onset times by repitition time TR, round to integer
    # duration computes length of trial in terms of frames by dividing duration time by TR, round up to integer
    # frames = for each trial, generate range of frames corresponding to trial duration and start time..
    def load_evs(subject, experiment, run):
      """Load EVs (explanatory variables) data for one task experiment.
    
      Args:
        subject (str): subject ID to load
        experiment (str) : Name of experiment
        run (int): 0 or 1
    
      Returns
        evs (list of lists): A list of frames associated with each condition
    
      """
      frames_list = []
      task_key = f'tfMRI_{experiment}_{RUNS[run]}'
      for cond in EXPERIMENTS[experiment]['cond']:
        ev_file  = f"{HCP_DIR}/subjects/{subject}/{experiment}/{task_key}/EVs/{cond}.txt"
        ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
        ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
        # Determine when trial starts, rounded down
        start = np.floor(ev["onset"] / TR).astype(int)
        # Use trial duration to determine how many frames to include for trial
        duration = np.ceil(ev["duration"] / TR).astype(int)
        # Take the range of frames that correspond to this specific trial
        '''
        print(start)
        print(start.shape)
        print(duration)
        print(duration.shape)
        print(list(zip(start,duration)))
        '''
        frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
        # print(frames)
        frames_list.append(frames)
    
      return frames_list
    
    my_exp = 'EMOTION'
    my_subj = subjects[1]
    my_run = 1
    
    data = load_single_timeseries(subject=my_subj,
                                  experiment=my_exp,
                                  run=my_run,
                                  remove_mean=True)
    # print(data.shape)
    
    evs = load_evs(subject=my_subj, experiment=my_exp, run=my_run)
    
    # A function to average all frames from any given condition
    
    def average_frames3(data, evs, experiment, cond):
        # Find the index of the given condition within the experiment
        idx = EXPERIMENTS[experiment]['cond'].index(cond)
    
        # List to store the mean data for each set of event frames
        mean_data_list = []
    
        # Iterate over each set of event frames for the given condition
        for i in range(len(evs[idx])):
            # Debugging print statements
            '''
            print(f"\nProcessing event frame set {i + 1}/{len(evs[idx])} for condition '{cond}'")
            print(f"Event indices: {evs[idx][i]}")
            print(f"Data shape before extraction: {data.shape}")
            '''
    
            try:
                # Extract the data corresponding to the current set of event frames
                current_data = data[:, evs[idx][i]]
                print(f"Extracted data shape: {current_data.shape}")
            except IndexError as e:
                print(f"IndexError: {e}")
                continue
    
            # Compute the mean of the extracted data along the time axis
            mean_current_data = np.mean(current_data, axis=1, keepdims=True)
            # print(f"Mean current data shape: {mean_current_data.shape}")
    
            # Append the mean data to the list
            mean_data_list.append(mean_current_data)
    
        if not mean_data_list:
            print("Error: No valid event frames found")
            return None
    
        stacked_means = np.stack(mean_data_list, axis=-1)
        # print(f"Stacked means shape: {stacked_means.shape}")
    
        #return overall_mean
        return stacked_means
    
    fr_activity = average_frames3(data, evs, my_exp, 'fear')
    nt_activity = average_frames3(data, evs, my_exp, 'neut')
    #contrast = fr_activity - nt_activity  # difference between 'fear' and 'neutral' conditions
    if fr_activity is not None and nt_activity is not None:
        final_concatenated_means = np.concatenate((fr_activity, nt_activity), axis=-1)
        # print(f"Final concatenated means shape: {final_concatenated_means.shape}")
    
    
    all_trials = []
    all_labels = []
    
    for i in range(len(subjects)):
        for r in [0, 1]:
            data = load_single_timeseries(subject=subjects[i],
                                          experiment=my_exp,
                                          run=r,
                                          remove_mean=True)
    
            # Get the trials for both conditions
            neut_trials = average_frames3(data, evs, my_exp, 'neut')
            fear_trials = average_frames3(data, evs, my_exp, 'fear')
    
            # Append each trial individually, flattened
            if neut_trials is not None:
                for trial in neut_trials.transpose(2, 0, 1):  # Iterate over trials and reshape
                    all_trials.append(trial.flatten())
                    all_labels.append(0)
            if fear_trials is not None:
                for trial in fear_trials.transpose(2, 0, 1):  # Iterate over trials and reshape
                    all_trials.append(trial.flatten())
                    all_labels.append(1)
    
    # Convert to numpy array
    all_trials_array = np.array(all_trials)
    
    # Write trials and labels to separate csv files
    np.savetxt('all_trials.csv', all_trials_array, delimiter=',')
    np.savetxt('all_labels.csv', all_labels, delimiter=',')
    
    # for 108 trials
    all_trials = []
    all_labels = []
    
    for i in range(len(subjects)):
        for r in [0, 1]:
            data = load_single_timeseries(subject=subjects[i],
                                          experiment=my_exp,
                                          run=r,
                                          remove_mean=True)
    
            # Get the trials for both conditions
            neut_trials = average_frames3(data, evs, my_exp, 'neut')
            fear_trials = average_frames3(data, evs, my_exp, 'fear')
    
            # Append each trial individually, flattened
            if neut_trials is not None:
                for trial in neut_trials.transpose(2, 0, 1):  # Iterate over trials and reshape
                    all_trials.append(trial.flatten())
                    all_labels.append(0)
            if fear_trials is not None:
                for trial in fear_trials.transpose(2, 0, 1):  # Iterate over trials and reshape
                    all_trials.append(trial.flatten())
                    all_labels.append(1)
    
    # Convert to numpy array
    all_trials_array = np.array(all_trials)
    all_labels = np.array(all_labels)
    
    # separate trials for neutral and fear
    neutral_trials_array = all_trials_array[all_labels == 0]
    fear_trials_array = all_trials_array[all_labels == 1]
    
    # for 108 trials
    # sample 54 trials from each condition
    if len(neutral_trials_array) > 54:
        sampled_neutral_trials = neutral_trials_array[np.random.choice(neutral_trials_array.shape[0], size=54, replace=False)]
    else:
        sampled_neutral_trials = neutral_trials_array
    
    if len(fear_trials_array) > 54:
        sampled_fear_trials = fear_trials_array[np.random.choice(fear_trials_array.shape[0], size=54, replace=False)]
    else:
        sampled_fear_trials = fear_trials_array
    
    # Concatenate the sampled trials
    sampled_trials = np.vstack((sampled_neutral_trials, sampled_fear_trials))
    sampled_labels = np.concatenate(([0] * sampled_neutral_trials.shape[0], [1] * sampled_fear_trials.shape[0]))
    
    # Write trials and labels to separate csv files
    np.savetxt('all_trials.csv', sampled_trials, delimiter=',')
    np.savetxt('all_labels.csv', sampled_labels, delimiter=',')
    
    print(f'sampled_trials.shape: {sampled_trials.shape}')
    print(f'sampled_labels.shape: {sampled_labels.shape}')
    
    # write a foor loop o iterate over 360 input features (brain regions)
    df_trials = pd.read_csv("./all_trials.csv")
    df_labels = pd.read_csv("./all_labels.csv")
    
    #Train & Test Set
    X= df_trials.iloc[: , :-1]
    y= df_labels.iloc[: , -1] #target at the end column
    
    train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42,test_size=0.2)
    print("\n--Training data samples--")
    print(f'train_x.shape: {train_x.shape}')
    print(f'test_x.shape: {test_x.shape}')
    print("\n--Testing data samples--")
    print(f'train_y.shape: {train_y.shape}')
    print(f'test_y.shape: {test_y.shape}')
    
    ###First use a MinMaxscaler to scale all the features of Train & Test dataframes
    
    scaler = preprocessing.MinMaxScaler() #normalizes the features
    x_train = scaler.fit_transform(train_x.values)
    x_test =  scaler.fit_transform(test_x.values)    
    
    ###Then convert the Train and Test sets into Tensors
    
    x_tensor =  torch.from_numpy(x_train).float() #converts numpy array to pytorch tensor
    y_tensor =  torch.from_numpy(train_y.values.ravel()).float() #flattens the array into a one-dimensional array
    xtest_tensor =  torch.from_numpy(x_test).float()
    ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float() #flattens the array into a one-dimensional array
    
    # lazy man's cross validation
    # This loss function is typically used for binary classification tasks where the model outputs probabilities (e.g., values between 0 and 1)
    loss_func = nn.BCELoss()
    
    try:
        mp.set_start_method('spawn')
    except:
        pass
    
    def ecb(e):
        assert False, print(e)
        
    m = mp.Manager() # sometime you have to use it!
    
    def get_indices_from_file (file_path, cutoff):
      data = np.load(file_path)
      indices_bool = (data >= cutoff)
      indices_bool = np.invert(indices_bool)
      indices = np.where(indices_bool)[0]
      return indices
    
    #target_num_ROIs = int(2/3*num_ROIs)
    num_ROIs = x_tensor.shape[1]
    
    def find_optimal_cutoffs(file_path):
      accuracy_data = np.load(file_path)
      cutoff = 1
      step = 0.005
      num_ROIs = len(accuracy_data)
      target_num_ROIs = int(2/3*num_ROIs)
    
      while num_ROIs > target_num_ROIs:
        cutoff -= step
        num_ROIs = len(get_indices_from_file(file_path, cutoff))
    
      prev_num_ROIs = len(get_indices_from_file(file_path, cutoff+step))
      if abs(prev_num_ROIs - target_num_ROIs) < abs(num_ROIs - target_num_ROIs):
        cutoff += step
    
      return cutoff
    
    bs = 32
    epochs = 50
    x_tensor =  torch.from_numpy(x_train).float() #converts numpy array to pytorch tensor
    y_tensor =  torch.from_numpy(train_y.values.ravel()).float() #flattens the array into a one-dimensional array
    xtest_tensor =  torch.from_numpy(x_test).float()
    ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float() #flattens the array into a one-dimensional array
    y_tensor = y_tensor.unsqueeze(1)
    
    #For the validation/test dataset
    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds = TensorDataset(xtest_tensor, ytest_tensor)
    test_loader = DataLoader(test_ds, batch_size=32)#model will process 32 samples at a time
    accuracies = []
    num_rounds = [1]
    num_splits = 2 # 70
    
    for k in num_rounds:
        new_x_tensor = x_tensor.detach().clone().to("cuda")
        new_xtest_tensor = xtest_tensor.detach().clone().to("cuda")
        ROI_names = np.array(all_ROI_names.copy())
    
        for l in range(k):
            file_path = f'./{l+7}_subset_sampling_acc.npy'
            cutoff = find_optimal_cutoffs(file_path)
            indices = get_indices_from_file(file_path, cutoff)
            new_x_tensor = new_x_tensor[:, indices]
            new_xtest_tensor = new_xtest_tensor[:, indices]
            ROI_names = ROI_names[indices]
            print(ROI_names)
    
        num_ROIs = new_x_tensor.shape[1]
        accuracies = []
        
        q = m.Queue()
        gpu_limits = {0: 2, 1: 1}  # Limit to 2 processes on CUDA:0, 1 process on CUDA:1
        gpu_usage = {0: 0, 1: 0}
        num_processes = sum(gpu_limits.values())
        train_pool = mp.Pool(5)
        processes = []
        for i in range(num_ROIs):
            print(f'Submitting region worker for ROI {i}')
            # train_pool.apply_async(train_region_worker, args=(i, new_x_tensor, y_tensor, bs, epochs, num_splits, q))
            # avg_acc = train_region_worker(i, new_x_tensor, y_tensor, bs, epochs, num_splits)
            ROI_indices = [j for j in range(new_x_tensor.shape[1]) if j != i]
            x_subset = new_x_tensor[:, ROI_indices]
            
            for split in range(num_splits):
                while len(processes) >= num_processes:
                    time.sleep(0.5)  # Wait for a process to finish
                    processes = [p for p in processes if not p.ready()]
                
                for gpu_device in [0] * gpu_limits[0] + [1] * gpu_limits[1]:
                   print(f'[hi mom]: {gpu_usage[gpu_device] < gpu_limits[gpu_device]}')
                   if gpu_usage[gpu_device] < gpu_limits[gpu_device]:
                       X, y = x_tensor.detach().clone().to(f"cuda:{gpu_device}"), y_tensor.detach().clone().to(f"cuda:{gpu_device}")
                       train_x_split, test_x_split, train_y_split, test_y_split = train_test_split(X, y, random_state=split, test_size=0.15)
                       train_ds_split = TensorDataset(train_x_split, train_y_split)
                       train_dl_split = DataLoader(train_ds_split, batch_size=bs)
                       test_ds_split = TensorDataset(test_x_split, test_y_split)
                       test_loader_split = DataLoader(test_ds_split, batch_size=32)
                       n_input_dim = train_x_split.shape[1]
                       
                       model = ChurnModel(n_input_dim).to(f"cuda:{gpu_device}")
                       optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
                       loss_func = torch.nn.BCELoss()
                       args = (
                           epochs, model, loss_func, optimizer, 
                           evaluate_model, train_dl_split, 
                           test_loader_split, q, gpu_device
                       )
                       def little_fx(device):
                           def littler_fx(_): 
                               gpu_usage[device] -= 1
                           return littler_fx
                       process = train_pool.apply_async(
                           func=train_worker, 
                           args=args, 
                           callback=little_fx(gpu_device),
                           error_callback=ecb
                       )
                       gpu_usage[gpu_device] += 1
                       processes.append(process)
            
        train_pool.close()
        train_pool.join()
            
        # Collect results from queue
        split_accuracies = []
        for i in range(num_ROIs):
            for j in range(num_splits):
                accuracy = q.get()
                split_accuracies.append(accuracy)

            avg_acc = np.mean(split_accuracies)
            accuracies.append(avg_acc)
    
        save_file = f'./{k+7}_subset_sampling_acc.npy'
        np.save(save_file, accuracies)
        
    data = np.load(r"./8_subset_sampling_acc.npy")
    print(data)
    print(f'num_ROIs: {num_ROIs}')
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Plotting example, assuming train_loss and val_acc are collected somewhere
'''
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Model #{i}')

plt.figure()
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Model #{i}')
plt.show()
'''
    
    
    
    
    
    
    
    
    
    
    























































