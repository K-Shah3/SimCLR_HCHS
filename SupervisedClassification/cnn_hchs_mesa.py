import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
import pickle as pickle
import numpy as np
from copy import deepcopy
from itertools import chain
import datetime
import ast

import torch
import sys
from torch import nn, Tensor, optim
import torch.nn.functional as func
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
sns.set_context('poster')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression

import actigraphy_utilities

dataset_diseases = {'hchs': ['hypertension', 'diabetes', 'sleep_apnea', 'metabolic_syndrome', 'insomnia'],
            'mesa': ['sleep_apnea', 'insomnia']}

def parse_arguments(args):
    testing_flag = args[1] == 'True'
    bs = ast.literal_eval(args[2])
    lr = ast.literal_eval(args[3])
    max_epochs = int(args[4])
    
    return testing_flag, bs, lr, max_epochs

def get_working_directory(testing_flag, hchs_or_mesa):
    '''Get the correct directories depending on what dataset is being used'''
    if testing_flag:
        working_directory = f'byol_{hchs_or_mesa}_testing/'
    else:
        working_directory = f'byol_{hchs_or_mesa}/'

    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    return working_directory

def get_datasets_from_path(testing_flag, hchs_or_mesa):
    working_directory = get_working_directory(testing_flag, hchs_or_mesa)

    dataset_save_path = os.path.join(os.path.dirname(os.getcwd()), "PickledData", hchs_or_mesa)

    if testing_flag:
        # path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, '100_users_reduced_test_train_users.pickle')
    else:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'test_train_split_dict.pickle')
    
    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)
    
    disease_user_datasets = {}
    diseases = dataset_diseases[hchs_or_mesa]
    for disease in diseases:
        disease_user_dataset_path = os.path.join(dataset_save_path, f'{disease}_user_datasets.pickle')
        with open(disease_user_dataset_path, 'rb') as f:
            user_dataset = pickle.load(f)
        new_user_dataset = {}
        if testing_flag:
            reduced_user_dataset = {}
            for user, data in user_dataset.items():
                if user in test_train_split_dict['train']:
                    reduced_user_dataset[user] = data
                if user in test_train_split_dict['test']:
                    reduced_user_dataset[user] = data

            user_dataset = reduced_user_dataset
        
        disease_user_datasets[disease] = user_dataset

    return disease_user_datasets, test_train_split_dict, working_directory

def get_rid_of_nan_labels_from_user_datasets(user_datasets):
    new_user_dataset = {}
    for user, data_label in user_datasets.items():
        data, label = data_label
        if not (label == 0.0 or label == 1.0):
            print(f'{user} : {label}')
            continue
        new_user_dataset[user] = data_label
    
    return new_user_dataset

def cnn_disease(testing_flag, hchs_or_mesa, disease, bs=128, lr=0.1, max_epochs=20):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, hchs_or_mesa)
    user_datasets = disease_user_datasets[disease]
    user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
    train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs
    )

    input_x = train_dataset.average_length
    input_y = 5

    input_model = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, hchs_or_mesa=hchs_or_mesa)
    model = actigraphy_utilities.fit_model(input_model, train_loader, val_loader, lr, max_epochs)

    val_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset)
    )

    f1_macro_micro = []

    for data, label in val_loader:
        y_pred = model(data.float())
        y_pred = torch.heaviside(y_pred, values=torch.zeros(1))
        label = torch.unsqueeze(label, 1)
        y_pred = y_pred.detach().numpy()
        y = label.detach().numpy()
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_macro_micro.append(f1_macro)
        f1_micro = f1_score(y, y_pred, average='micro')
        f1_macro_micro.append(f1_micro)

    return f1_macro_micro
        
def main(testing_flag, bs=128, lr=0.1, max_epochs=20):
    # hchs
    hchs_metabolic_syndrome_f1_macro, hchs_metabolic_syndrome_f1_micro = cnn_disease(testing_flag, 'hchs', 'metabolic_syndrome', bs=bs, lr=lr, max_epochs=max_epochs)
    print(f'hchs metabolic syndrome:')
    print(f'    f1 macro: {hchs_metabolic_syndrome_f1_macro}')
    print(f'    f1 micro: {hchs_metabolic_syndrome_f1_micro}')
    # mesa 
    mesa_sleep_apnea_f1_macro, mesa_sleep_apnea_f1_micro = cnn_disease(testing_flag, 'mesa', 'sleep_apnea', bs=bs, lr=lr, max_epochs=max_epochs)
    print(f'mesa sleep apnea:')
    print(f'    f1 macro: {mesa_sleep_apnea_f1_macro}')
    print(f'    f1 micro: {mesa_sleep_apnea_f1_micro}')
    mesa_insomnia_f1_macro, mesa_insomnia_f1_micro = cnn_disease(testing_flag, 'mesa', 'insomnia', bs=bs, lr=lr, max_epochs=max_epochs)
    print(f'mesa insomnia:')
    print(f'    f1 macro: {mesa_insomnia_f1_macro}')
    print(f'    f1 micro: {mesa_insomnia_f1_micro}')
    


def testing(testing_flag=True, hchs_or_mesa='mesa', disease='metabolic_syndrome', bs=128):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, hchs_or_mesa)
    user_datasets = disease_user_datasets[disease]
    user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
    train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs
    )

    input_x = train_dataset.average_length
    input_y = 5
    my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, hchs_or_mesa=hchs_or_mesa, disease=disease)
    
    model = actigraphy_utilities.fit_model(my_nn, train_loader, val_loader, 0.01, 1)

    # i = 0
    # for data, label in val_loader:
    #     pred = model(data.float())
    #     if i == 0:
    #         break 
    val_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset)
    )

    for data, label in val_loader:
        y_pred = model(data.float())
        y_pred = torch.heaviside(y_pred, values=torch.zeros(1))
        label = torch.unsqueeze(label, 1)
        y_pred = y_pred.detach().numpy()
        y = label.detach().numpy()
        f1_micro = f1_score(y, y_pred, average='micro')
        f1_macro = f1_score(y, y_pred, average='macro')

        print(f'{hchs_or_mesa} - {disease} - f1 micro: {f1_micro}')
        print(f'{hchs_or_mesa} - {disease} - f1 macro: {f1_macro}')

    

if __name__ == "__main__":
    # testing_flag, bs, lr, max_epochs = parse_arguments(sys.argv)
    testing_flag, bs, lr, max_epochs = False, 1024, 0.01, 2
    # main(testing_flag, bs, lr, max_epochs)
    testing(False, 'hchs', 'hypertension', 128)
