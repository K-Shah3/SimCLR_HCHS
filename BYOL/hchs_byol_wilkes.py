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

import actigraphy_utilities 

diseases = ['hypertension', 'diabetes', 'sleep_apnea', 'metabolic_syndrome', 'insomnia']

def parse_arguments(args):
    testing_flag = args[1] == 'True'
    batch_size = int(args[2])
    epoch_number = int(args[3])
    latent_dim = int(args[4])
    projection_dim = int(args[5])
    plotting_flag = args[6] == 'True'
    
    return testing_flag, batch_size, epoch_number, latent_dim, projection_dim, plotting_flag

def get_datasets_from_path(testing_flag):
    if testing_flag:
        working_directory = 'byol_hchs_testing/'
    else:
        working_directory = 'byol_hchs/'
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    dataset_save_path = os.path.join(os.path.dirname(os.getcwd()), "PickledData", "hchs")

    if testing_flag:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict.pickle')
        # path_to_test_train_split_dict = os.path.join(dataset_save_path, '100_users_reduced_test_train_users.pickle')
    else:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'test_train_split_dict.pickle')
    
    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)
    
    disease_user_datasets = {}
    diseases = ['hypertension', 'diabetes', 'sleep_apnea', 'metabolic_syndrome', 'insomnia']
    for disease in diseases:
        disease_user_dataset_path = os.path.join(dataset_save_path, f'{disease}_user_datasets.pickle')
        with open(disease_user_dataset_path, 'rb') as f:
            user_dataset = pickle.load(f)
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

def main(testing_flag, batch_size, epoch_number, latent_dim, projection_dim, disease):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag)
    user_datasets = disease_user_datasets[disease]
    # get trained autoencoder 
    autoencoder, encoder, decoder = actigraphy_utilities.get_trained_autoencoder(user_datasets, test_train_split_dict, batch_size, latent_dim)
    # use encoder as input into BYOL 
    model = deepcopy(encoder)

    train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    byol_model = deepcopy(model)
    image_size = (train_dataset.average_length, 5)
    byol = actigraphy_utilities.BYOL(byol_model, image_size=image_size, projection_size=projection_dim)
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        weights_summary=None,
        logger=False
    )
    byol_trainer.fit(byol, train_loader, val_loader) 
    byol_encoder = byol.encoder

    return byol_encoder, test_dataset, train_dataset, working_directory

def downstream_evaluation(byol_encoder, test_dataset, train_dataset):
    '''Given trained byol encoder, use the representations to test how well it performs on a 
    downstream disease classification'''
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

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
    # roc_ovr = roc_auc_score(y_test, log_reg_clf.predict_proba(X_test), multi_class='ovr')
    # roc_ovo = roc_auc_score(y_test, log_reg_clf.predict_proba(X_test), multi_class='ovo')
    
    metrics['accuracy'] = accuracy
    # metrics['auc_ovr'] = roc_ovr
    # metrics['auc_ovo'] = roc_ovo

    return metrics

if __name__ == "__main__":
    testing_flag, batch_size, epoch_number, latent_dim, projection_dim, plotting_flag = parse_arguments(sys.argv)

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    print(start_time_str)

    autoencoder_function = main
    downstream_function = downstream_evaluation
    combined_metrics = {}
    for disease in diseases:
        byol_encoder, test_dataset, train_dataset, working_directory = autoencoder_function(testing_flag, batch_size, epoch_number, latent_dim, projection_dim, disease)
        metrics = downstream_function(byol_encoder, test_dataset, train_dataset)
        print(f'{disease}: {metrics}')
        combined_metrics[disease] = metrics

    save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-hchs-metrics.pickle'
    save_path = os.path.join(working_directory, save_name)

    with open(save_path, 'wb') as f:
        pickle.dump(combined_metrics, f)

    print(save_name)

    print('this was with no auc roc ')
    

    

