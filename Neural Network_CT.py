import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv('./normalized_data_resampling.csv')
print(data)
print(data.shape)

df_labels = data.iloc[:,-1]
df_labels.head()
df_labels.to_numpy()
np.savetxt('all_labels.csv', df_labels, delimiter=",")

# import feature data
data2 = pd.read_csv('./NN-inputs-ants-resampling.csv')
df_trials = data2.iloc[:,1:]
print(df_trials)
df_trials.to_numpy()
# write to a csv file
np.savetxt('all_trials.csv', df_trials, delimiter=",")

print(df_labels.min())
print(df_labels.max())

# put all features in a list
all_features = df_trials.columns.tolist()
print(len(all_features))
print(all_features)
print(type(all_features))
print(type(df_trials))

"""## Start Neural Network"""

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
device = torch.device("cuda:1")
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class ChurnModel(nn.Module):
    def __init__(self, n_input_dim):
        super(ChurnModel, self).__init__()
        self.n_hidden1 = 80
        self.n_hidden2 = 80
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

def get_indices_from_file (file_path, cutoff):
  data = np.load(file_path)
  indices_bool = (data >= cutoff)
  indices_bool = np.invert(indices_bool)
  indices = np.where(indices_bool)[0]
  return indices

#target_num_ROIs = int(2/3*num_ROIs)
#num_ROIs = x_tensor.shape[1]

def find_optimal_cutoffs(file_path):
  accuracy_data = np.load(file_path)
  cutoff = 1
  step = 0.005
  num_features = len(accuracy_data)
  target_num_features = int(2/3*num_features)

  while num_features > target_num_features:
    cutoff -= step
    num_features = len(get_indices_from_file(file_path, cutoff))

  #check if previous cutoff was closer to the target
  prev_num_features = len(get_indices_from_file(file_path, cutoff+step))
  if abs(prev_num_features - target_num_features) < abs(num_features - target_num_features):
    cutoff += step

  return cutoff

save_dir = "./indiv_CT"
os.makedirs(save_dir, exist_ok=True)


bs = 32
test_ratio = 0.15
num_rounds = 5 # 9
num_splits = 100 # 800

X = pd.read_csv("./all_trials.csv")
y = pd.read_csv("./all_labels.csv")

loss_func = nn.BCELoss()
accuracies = []

train_x, test_x, train_y, test_y = train_test_split(X,y,random_state=42,test_size=test_ratio)
print("\n--Training data samples--", train_x.shape, test_x.shape)
print("\n--Testing data samples--", train_y.shape, test_y.shape)
# converts numpy array to pytorch tensor
x_tensor = torch.from_numpy(train_x.values).float()
# ravel() flattens the array into a one-dimensional array
y_tensor = torch.from_numpy(train_y.values.ravel()).float()
xtest_tensor = torch.from_numpy(test_x.values).float()
ytest_tensor = torch.from_numpy(test_y.values.ravel()).float()
y_tensor = y_tensor.unsqueeze(1)
ytest_tensor = ytest_tensor.unsqueeze(1)


for k in range(num_rounds):
  print(f'round: {k}')
  # explicity make sure we have a copy of the tensor
  new_x_tensor = x_tensor.detach().clone()
  new_xtest_tensor = xtest_tensor.detach().clone()
  features = np.array(all_features.copy())
  # we want to do this unless it's the first time:
  # load the ith accuracy file, get its indices via cutoff function, then apply
  for l in range(k): # this for loop is for trim trim trim!!!
    file_path = f'./indiv_CT/{l+1}_Mean_acc.npy'
    cutoff = find_optimal_cutoffs(file_path)
    indices = get_indices_from_file(file_path, cutoff)
    new_x_tensor = new_x_tensor[:, indices]
    new_xtest_tensor = new_xtest_tensor[:, indices]
    features = features[indices]
    print(features)

  #new_x_tensor = new_x_tensor[:, :10]
  #new_xtest_tensor = new_xtest_tensor[:, :10]
  #features = features[:100]
  num_features = new_x_tensor.shape[1]
  print(f"num_features: {num_features}")
  accuracies = []
  for i in range(num_features):
    feature_indices = [j for j in range(num_features) if j!=i]
    # select a subset of brain regions with highest accuracies
    x_subset = new_x_tensor[:, feature_indices]
    xtest_subset = new_xtest_tensor[:, feature_indices]
    # add a for loop with split in  for cross validation

    split_accuracies = []
    for split in range(num_splits):
      # First use a MinMaxscaler to scale all the features of Train & Test dataframes
      # this should not be done up here, it should be done everytime the datasets change
      scaler = preprocessing.MinMaxScaler() # normalizes the features
      og_num_data_pts = x_subset.shape[0]
      combined_x = torch.cat((x_subset, xtest_subset), dim=0)
      combined_y = torch.cat((y_tensor, ytest_tensor), dim=0)
      num_data_pts = combined_x.shape[0]
      # choose a new split
      x_train_indices  = np.random.choice(num_data_pts, size=og_num_data_pts, replace=False)
      inverted_indices = [i for i in range(num_data_pts) if i not in x_train_indices]
      train_x_split = combined_x[x_train_indices]
      _y_tensor = combined_y[x_train_indices].to(device)
      # god forgive me
      test_x_split = combined_x[inverted_indices]
      _ytest_tensor = combined_y[inverted_indices].to(device)
      train_x_split = scaler.fit_transform(train_x_split)
      test_x_split =  scaler.fit_transform(test_x_split)
      train_x_split = torch.tensor(train_x_split, dtype=torch.float32).to(device)
      test_x_split = torch.tensor(test_x_split, dtype=torch.float32).to(device)
      train_ds_split = TensorDataset(train_x_split, _y_tensor)
      test_ds_split = TensorDataset(test_x_split, _ytest_tensor)
      # might explicity set shuffling to be random or not
      train_dl_split = DataLoader(train_ds_split, batch_size=bs)
      test_loader_split = DataLoader(test_ds_split, batch_size=32)
      n_input_dim = train_x_split.shape[1]
      model = ChurnModel(n_input_dim)
      model = model.to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
      #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

      train_loss = []
      val_acc = []
      epochs = 130
      for epoch in range(epochs):
          model.train()
          #Within each epoch run the subsets of data = batch sizes.
          epoch_loss = 0
          for xb, yb in train_dl_split:
              #print(xb.shape)
              y_pred = model(xb)            # Forward Propagation
              #print(y_pred)
              #print(yb)
              loss = loss_func(y_pred, yb)  # Loss Computation
              epoch_loss += loss.item()
              optimizer.zero_grad()         # Clearing all previous gradients, setting to zero
              loss.backward()               # Back Propagation
              optimizer.step()              # Updating the parameters
          #print(f"Loss @ Epoch #{epoch}: {epoch_loss:.4f}")
          train_loss.append(epoch_loss)
          # get validation accuracy
          v_acc = evaluate_model(model, test_loader_split)
          val_acc.append(v_acc)
      # we are getting the maximum test accuracy from all 50 training epochs for each model
      accuracy = max(val_acc)
      split_accuracies.append(accuracy)

    accuracies.append(np.array(split_accuracies))
    print("AVG VAL ACCURACY", np.mean(split_accuracies))
    # save plots
    dir = 'indiv_figures/'
    if not os.path.exists(dir):
      os.makedirs(dir)
    # save training loss plot
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Model #{i}')
    plt.savefig(os.path.join(dir, f'model_{i}_training_loss.png'))
    plt.close()

    # save test accuracy plot
    plt.figure()
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Model #{i}')
    plt.savefig(os.path.join(dir, f'model_{i}_test_accuracy.png'))
    plt.close()

  save_file = f'./indiv_CT/{k+1}_acc.npy'
  np.save(save_file, accuracies)

  save_file_mean = f'./indiv_CT/{k+1}_Mean_acc.npy'
  np.save(save_file_mean, np.mean(accuracies, axis=1))
