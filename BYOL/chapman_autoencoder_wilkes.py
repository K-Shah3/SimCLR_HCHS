import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
import pickle5 as pickle
import numpy as np
from copy import deepcopy
from itertools import chain
import datetime

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as func
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
# from torchvision.models import resnet18
import pytorch_lightning as pl
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import torch

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

import byol_chapman_utilities

def stack_user_datasets(user_dataset):
    data_list = []
    for user, data_label in user_dataset.items():
        data_list.append(data_label[0])

    return np.stack(data_list, axis=0)

def get_normalised_training_testing_data(testing_flag):
    dataset_save_path = os.path.join(os.path.dirname(os.getcwd()), "PickledData", "chapman")
    path_to_patient_to_rhythm_dict = os.path.join(dataset_save_path, 'patient_to_rhythm_dict.pickle')

    # paths to user datasets with no nan values
    if testing_flag:
        path_to_user_datasets = os.path.join(dataset_save_path, 'reduced_four_lead_user_datasets_no_nan.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict_no_nan.pickle')
    else:
        path_to_user_datasets  = os.path.join(dataset_save_path, 'four_lead_user_datasets_no_nan.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, "test_train_split_dict_no_nan.pickle")

    with open(path_to_user_datasets, 'rb') as f:
        user_datasets = pickle.load(f)

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)
    
    train_user_list = test_train_split_dict['train']
    test_user_list = test_train_split_dict['test']

    means, stds, mins, maxs = byol_chapman_utilities.get_mean_std_min_max_from_user_list_format(user_datasets, train_user_list)
    normalised_user_datasets_train = {}
    normalised_user_datasets_test = {}
    for user, data_label in user_datasets.items():
        data = data_label[0]
        normalised_data = byol_chapman_utilities.normalise(data, means, stds, mins, maxs)
        if user in train_user_list:
            normalised_user_datasets_train[user] = (normalised_data, data_label[1])
        if user in test_user_list:
            normalised_user_datasets_test[user] = (normalised_data, data_label[1])

    x_train = stack_user_datasets(normalised_user_datasets_train)
    x_test = stack_user_datasets(normalised_user_datasets_test)

    return x_train, x_test 

class autoencoder(nn.Module):
    def __init__(self, latent_dim, x_dim=2500, y_dim=4):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x_dim*y_dim, latent_dim),
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, x_dim*y_dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_trained_autoencoder(testing_flag, latent_dim=512):
    dataset_save_path = os.path.join(os.path.dirname(os.getcwd()), "PickledData", "chapman")
    path_to_patient_to_rhythm_dict = os.path.join(dataset_save_path, 'patient_to_rhythm_dict.pickle')

    # paths to user datasets with no nan values
    if testing_flag:
        path_to_user_datasets = os.path.join(dataset_save_path, 'reduced_four_lead_user_datasets_no_nan.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict_no_nan.pickle')
    else:
        path_to_user_datasets  = os.path.join(dataset_save_path, 'four_lead_user_datasets_no_nan.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, "test_train_split_dict_no_nan.pickle")

    with open(path_to_user_datasets, 'rb') as f:
        user_datasets = pickle.load(f)

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)
    
    train_user_list = test_train_split_dict['train']
    test_user_list = test_train_split_dict['test']

    with open(path_to_patient_to_rhythm_dict, 'rb') as f:
        patient_to_rhythm_dict = pickle.load(f)
    
    train_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'train')
    test_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'test')

    batch_size = 128

    train_loader = DataLoader(
        train_chapman_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_chapman_dataset,
        batch_size=len(test_chapman_dataset),
    )
    # train autoencoder
    model = autoencoder(latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        for data_label in train_loader:
            data, _ = data_label 
            input_data = Variable(data)
            loss_data = data.view(data.size(0), -1)
            output = model(input_data.float())
            loss = criterion(output, loss_data.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # get validation 
    encoder = model.encoder
    decoder = model.decoder 

    for data_label in val_loader:
        data, _ = data_label 
        input_data = Variable(data)
        loss_data = data.view(data.size(0), -1)
        encoded_data = encoder(input_data.float())
        decoded_data = decoder(encoded_data.float())
        loss = criterion(decoded_data.float(), loss_data.float())
        print(f'val_loss: {loss.item()}')

    return autoencoder, encoder, decoder
    
def chapman_autoencoder_latent_features(testing_flag, latent_dim, epoch_number):
    x_train, x_test = get_normalised_training_testing_data(testing_flag)
    x_dim, y_dim = x_train.shape[1], x_train.shape[2]
    autoencoder = Autoencoder(latent_dim, x_dim, y_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(x_train, x_train,
        epochs=epoch_number,
        shuffle=True)

    encoded_x_train, encoded_x_test = autoencoder.encoder(x_train).numpy(), autoencoder.encoder(x_test).numpy()
    decoded_x_train, decoded_x_test = autoencoder.decoder(encoded_x_train).numpy(), autoencoder.decoder(encoded_x_test).numpy()

    return x_train, x_test, encoded_x_train, encoded_x_test, decoded_x_train, decoded_x_test 

def mse(original, decoded):
    return (np.square(original - decoded)).mean(axis=None)

def plot_latent_space_mse(testing_flag, latent_dims, epoch_number, save_path):
    train_mse = []
    test_mse = []
    for latent_dim in latent_dims:
        x_train, x_test, encoded_x_train, encoded_x_test, decoded_x_train, decoded_x_test = chapman_autoencoder_latent_features(testing_flag, latent_dim, epoch_number)
        train_mse.append(mse(x_train, decoded_x_train))
        test_mse.append(mse(x_test, decoded_x_test))

    fig, ax = plt.subplots()
    labels = [str(latent_dim) for latent_dim in latent_dims]
    ax.plot(labels, train_mse, label='train mse', c='magenta')
    ax.plot(labels, test_mse, label='test mse', c='cyan')
    ax.legend()
    ax.set_title('Plot to show how increasing the latent dimension affects the loss')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Measure Squared Error between Original Data and Decoded Data')
    plt.savefig(save_path)
    plt.show()

def testing(testing_flag):
    dataset_save_path = os.path.join(os.path.dirname(os.getcwd()), "PickledData", "chapman")
    path_to_patient_to_rhythm_dict = os.path.join(dataset_save_path, 'patient_to_rhythm_dict.pickle')

    # paths to user datasets with no nan values
    if testing_flag:
        path_to_user_datasets = os.path.join(dataset_save_path, 'reduced_four_lead_user_datasets_no_nan.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict_no_nan.pickle')
    else:
        path_to_user_datasets  = os.path.join(dataset_save_path, 'four_lead_user_datasets_no_nan.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, "test_train_split_dict_no_nan.pickle")

    with open(path_to_user_datasets, 'rb') as f:
        user_datasets = pickle.load(f)

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)
    
    train_user_list = test_train_split_dict['train']
    test_user_list = test_train_split_dict['test']

    with open(path_to_patient_to_rhythm_dict, 'rb') as f:
        patient_to_rhythm_dict = pickle.load(f)
    
    train_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'train')
    test_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'test')
    batch_size = 128

    train_loader = DataLoader(
        train_chapman_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_chapman_dataset,
        batch_size=batch_size,
    )

    model = autoencoder(512)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        for data_label in train_loader:
            data, _ = data_label 
            data = Variable(data)
            output = model(data.float())
            # print(data.size())
            # print(output.size())
            flattened_input = data.view(data.size(0), -1)
            loss = criterion(output, flattened_input.float())
            # print(flattened_input.size())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    encoder = model.encoder
    decoder = model.decoder 

    for data_label in val_loader:
        data, _ = data_label
        # data = data.view(data.size(0), -1)
        input_data = Variable(data)
        input_data = input_data.float()
        encoded_data = encoder(input_data)
        print(encoded_data.size())
        decoded_data = decoder(encoded_data.float())
        data = data.view(data.size(0), -1)
        loss = criterion(decoded_data.float(), data.float())
        print(f'val_loss: {loss.item()}')
