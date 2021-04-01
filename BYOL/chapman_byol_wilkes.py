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


import byol_chapman_utilities
import chapman_autoencoder_wilkes as chapman_autoencoder 

def parse_arguments(args):
    testing_flag = args[1] == 'True'
    batch_size = int(args[2])
    epoch_number = int(args[3])
    latent_dim = int(args[4])
    plotting_flag = args[5] == 'True'
    
    return testing_flag, batch_size, epoch_number, latent_dim, plotting_flag

def get_datasets_from_paths(testing_flag):
    '''Depending on whether or not we are in testing mode unpickle the correct user_datasets,
    patient_to_rhythm_dict, test_train_split_dict'''

    if testing_flag:
        working_directory = 'byol_chapman_testing/'
    else:
        working_directory = 'byol_chapman/'
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

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

    print(f'number of patients: {len(user_datasets)}')

    with open(path_to_patient_to_rhythm_dict, 'rb') as f:
        patient_to_rhythm_dict = pickle.load(f)

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)

    return user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory


def autoencoder_main(testing_flag, batch_size, epoch_number, latent_dim):
    # get trained autoencoder 
    autoencoder, encoder, decoder = chapman_autoencoder.get_trained_autoencoder(testing_flag, latent_dim)
    # we will use the encoder as input into byol 
    model = deepcopy(encoder)

    # get chapman datasets 
    user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag)
    
    # get 4 unique rhythms
    unique_rhythms_words = set(list(patient_to_rhythm_dict.values()))
    # {'AFIB': 0, 'SB': 1, 'SR': 2, 'GSVT': 3}
    rythm_to_label_encoding = {rhythm : index for index, rhythm in enumerate(unique_rhythms_words)}

    # get train and test datasets and create dataloaders
    train_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'train')
    test_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_chapman_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_chapman_dataset,
        batch_size=batch_size
    )

    # byol training model 
    byol_model = deepcopy(model)
    byol = byol_chapman_utilities.BYOL(byol_model, image_size=(2500, 4))
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        weights_summary=None
    )
    byol_trainer.fit(byol, train_loader, val_loader) 

    byol_encoder = byol.encoder

    return byol_encoder, test_chapman_dataset, train_chapman_dataset, working_directory

def testing(testing_flag, batch_size):
    autoencoder, encoder, decoder = chapman_autoencoder.get_trained_autoencoder(testing_flag)
    model = deepcopy(encoder)

    # get chapman datasets 
    user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag)
    
    # get 4 unique rhythms
    unique_rhythms_words = set(list(patient_to_rhythm_dict.values()))
    # {'AFIB': 0, 'SB': 1, 'SR': 2, 'GSVT': 3}
    rythm_to_label_encoding = {rhythm : index for index, rhythm in enumerate(unique_rhythms_words)}

    train_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'train')
    test_chapman_dataset = byol_chapman_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_chapman_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_chapman_dataset,
        batch_size=batch_size,
    )

    byol_model = deepcopy(model)
    byol = byol_chapman_utilities.BYOL(byol_model, image_size=(2500, 4))
    byol_trainer = pl.Trainer(
        max_epochs=10,
        accumulate_grad_batches=2048 // batch_size,
        weights_summary=None,
    )
    byol_trainer.fit(byol, train_loader, val_loader)

    byol_encoder = byol.encoder

    for data_label in val_loader:
        data, label = data_label
        byol_encoded_data = byol_encoder(data.float())
        print(f'byol encoded size {byol_encoded_data.size()}')
        print(label)

def downstream_evaluation(byol_encoder, test_chapman_dataset, train_chapman_dataset):
    train_loader = DataLoader(train_chapman_dataset, batch_size=len(train_chapman_dataset))
    test_loader = DataLoader(test_chapman_dataset, batch_size=len(test_chapman_dataset))
    for data_label in train_loader:
        data, label = data_label
        byol_train = byol_encoder(data.float())
        byol_train_numpy_x = byol_train.detach().numpy()
        byol_train_numpy_y = label.detach().numpy()

    X_train, y_train = byol_train_numpy_x, byol_train_numpy_y

    for data_label in test_loader:
        data, label = data_label
        byol_test = byol_encoder(data.float())
        byol_test_numpy_x = byol_test.detach().numpy()
        byol_test_numpy_y = label.detach().numpy()

    X_test, y_test = byol_test_numpy_x, byol_test_numpy_y

    log_reg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    log_reg_clf.fit(X_train, y_train)
    y_pred = log_reg_clf.predict(X_test)

    averages = ['micro', 'macro']
    metrics = {}
    for average in averages:
        f1 = f1_score(y_test, y_pred, average=average)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        metrics[f'f1_{average}'] = f1
        metrics[f'precision_{average}'] = precision
        metrics[f'recall_{average}'] = recall
    

    accuracy = accuracy_score(y_test, y_pred)
    roc_ovr = roc_auc_score(y_test, log_reg_clf.predict_proba(X_test), multi_class='ovr')
    roc_ovo = roc_auc_score(y_test, log_reg_clf.predict_proba(X_test), multi_class='ovo')
    
    metrics['accuracy'] = accuracy
    metrics['auc_ovr'] = roc_ovr
    metrics['auc_ovo'] = roc_ovo

    return metrics

def latent_dim_plot(testing_flag, batch_size, epoch_number, latent_dims):
    '''Create plot that shows how varying the latent dimension of BYOL affects the AUC score achieved while keeping 
    batch size and the epoch number the same'''
    auc_scores = []
    for latent_dim in latent_dims:
        byol_encoder, test_chapman_dataset, train_chapman_dataset, working_directory = autoencoder_main(testing_flag, batch_size, epoch_number, latent_dim)
        metrics = downstream_evaluation(byol_encoder, test_chapman_dataset, train_chapman_dataset)
        print(metrics)
        auc_scores.append(metrics['auc_ovr'])

    fig, ax = plt.subplots(figsize=(6, 5))
    latent_dim_labels = [str(latent_dim) for latent_dim in latent_dims]
    ax.bar(latent_dim_labels, auc_scores, color='magenta', width=0.3)
    ax.set_xlabel('BYOL Latent Dimension', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylabel('AUC score',  fontsize=10)
    ax.set_ylim(top=1)
    ax.set_title('Plot to show the relationship between \n latent dimension and AUC scores',  fontsize=12)

    latent_dim_save_name = '-'.join(latent_dim_labels)
    save_name = f'latent_dim_auc_{testing_flag}_{batch_size}_{latent_dim_save_name}_chapman.png'
    save_directory = os.path.join(working_directory, 'plots')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, save_name)

    plt.savefig(save_path)


if __name__ == '__main__':
    # parse arguments
    testing_flag, batch_size, epoch_number, latent_dim, plotting_flag = parse_arguments(sys.argv)
    
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    print(start_time_str)
    
    if not plotting_flag:
        print('proceeding as normal')
        byol_encoder, test_chapman_dataset, train_chapman_dataset, working_directory = autoencoder_main(testing_flag, batch_size, epoch_number, latent_dim)
        metrics = downstream_evaluation(byol_encoder, test_chapman_dataset, train_chapman_dataset)
        print(metrics)
        save_name = f'{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-chapman-metrics.pickle'
        print(save_name)
        save_path = os.path.join(working_directory, save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)
    else:
        print('plotting')
        latent_dims = [32, 64, 128, 256, 512]
        latent_dim_plot(testing_flag, batch_size, epoch_number, latent_dims)