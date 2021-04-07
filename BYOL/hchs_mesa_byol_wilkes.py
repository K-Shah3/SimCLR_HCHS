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

dataset_diseases = {'hchs': ['hypertension', 'diabetes', 'sleep_apnea', 'metabolic_syndrome', 'insomnia'],
            'mesa': ['sleep_apnea', 'insomnia']}

def parse_arguments(args):
    testing_flag = args[1] == 'True'
    hchs_or_mesa = args[2]
    batch_size = int(args[3])
    epoch_number = int(args[4])
    latent_dim = int(args[5])
    projection_dim = int(args[6])
    segment_number = int(args[7])
    main_type = args[8]
    
    return testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, segment_number, main_type

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
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict.pickle')
        # path_to_test_train_split_dict = os.path.join(dataset_save_path, '100_users_reduced_test_train_users.pickle')
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

def byol_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, hchs_or_mesa)
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

def combined_main(testing_flag, batch_size, epoch_number, latent_dim, projection_dim, disease='insomnia'):
    mesa_disease_user_datasets, mesa_test_train_split_dict, mesa_working_directory = get_datasets_from_path(testing_flag, 'mesa')
    hchs_disease_user_datasets, hchs_test_train_split_dict, hchs_working_directory = get_datasets_from_path(testing_flag, 'hchs')
    
    mesa_user_datasets = mesa_disease_user_datasets[disease]
    hchs_user_datasets = hchs_disease_user_datasets[disease]

    autoencoder, encoder, decoder = actigraphy_utilities.get_combined_trained_autoencoder(mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, batch_size, latent_dim)
    model = deepcopy(encoder)

    train_dataset = actigraphy_utilities.CombinedDataset(mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, 'train')
    test_dataset = actigraphy_utilities.CombinedDataset(mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, 'test')

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

    return byol_encoder, mesa_working_directory, hchs_working_directory, train_dataset.average_length

def combined_downstream_evaluation(hchs_or_mesa, byol_encoder, disease, average_length):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, hchs_or_mesa)
    user_datasets = disease_user_datasets[disease]

    train_dataset = actigraphy_utilities.SpecificLengthDataset(user_datasets, test_train_split_dict, 'train', average_length)
    test_dataset = actigraphy_utilities.SpecificLengthDataset(user_datasets, test_train_split_dict, 'test', average_length)

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

def multiple_segment_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease, segment_number):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, hchs_or_mesa)
    user_datasets = disease_user_datasets[disease]

    autoencoder, encoder, decoder = actigraphy_utilities.get_trained_autoencoder_ms(user_datasets, test_train_split_dict, batch_size, latent_dim, segment_number)
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
    image_size = (train_dataset.average_length // segment_number, 5)
    byol = actigraphy_utilities.BYOL_MS(byol_model, image_size=image_size, projection_size=projection_dim, segment_number=segment_number)
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        accumulate_grad_batches=2048 // batch_size,
        weights_summary=None,
        logger=False
    )
    byol_trainer.fit(byol, train_loader, val_loader) 
    byol_encoder = byol.encoder

    return byol_encoder, test_dataset, train_dataset, working_directory

def downstream_evaluation_ms(byol_encoder, test_dataset, train_dataset, segment_number):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


    for data_label in train_loader:
        data, label = data_label
        segment_size = data.size()[2] // segment_number
        split = torch.split(data, segment_size, dim=2)
        split = split[:segment_number]
        encoded_split_numpy = [byol_encoder(split_data.float()).detach().numpy() for split_data in split]
        # take mean of encoded segments 
        byol_train_numpy_x = np.mean(np.array(encoded_split_numpy), axis=0)
        byol_train_numpy_y = label.detach().numpy()

    X_train, y_train = byol_train_numpy_x, byol_train_numpy_y

    for data_label in test_loader:
        data, label = data_label 
        data, label = data_label
        segment_size = data.size()[2] // segment_number
        split = torch.split(data, segment_size, dim=2)
        split = split[:segment_number]
        encoded_split_numpy = [byol_encoder(split_data.float()).detach().numpy() for split_data in split]
        # take mean of encoded segments 
        byol_test_numpy_x = np.mean(np.array(encoded_split_numpy), axis=0)
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
    testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, segment_number, main_type = parse_arguments(sys.argv)

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    print(start_time_str)

    if main_type == 'normal':
        all_metrics = {}
        for disease in dataset_diseases[hchs_or_mesa]:
            byol_encoder, test_dataset, train_dataset, working_directory = byol_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease)
            metrics = downstream_evaluation(byol_encoder, test_dataset, train_dataset)
            print(f'{disease}: {metrics}')
            all_metrics[disease] = metrics

        save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-{hchs_or_mesa}-metrics.pickle'
        save_path = os.path.join(working_directory, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(all_metrics, f)

        print(save_name)

    elif main_type == 'ms':
        all_metrics = {}
        for disease in dataset_diseases[hchs_or_mesa]:
            byol_encoder, test_dataset, train_dataset, working_directory = multiple_segment_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease, segment_number)
            metrics = downstream_evaluation_ms(byol_encoder, test_dataset, train_dataset, segment_number)
            print(f'{disease}: {metrics}')
            all_metrics[disease] = metrics

        save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-{segment_number}-{hchs_or_mesa}-ms-metrics.pickle'
        save_path = os.path.join(working_directory, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(all_metrics, f)

        print(save_name)

    else:
        assert main_type == 'combined'
        byol_encoder, mesa_working_directory, hchs_working_directory, average_length = combined_main(testing_flag, batch_size, epoch_number, latent_dim, projection_dim)
        diseases = dataset_diseases[hchs_or_mesa]
        all_metrics = {}
        for disease in diseases:
            metrics = combined_downstream_evaluation(hchs_or_mesa, byol_encoder, disease, average_length)
            all_metrics[disease] = metrics 
            print(f'{disease}: {metrics}')
        
        save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-{hchs_or_mesa}-combined-metrics.pickle'
        if hchs_or_mesa == 'hchs':
            working_directory = hchs_working_directory
        else:
            working_directory = mesa_working_directory
        save_path = os.path.join(working_directory, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(all_metrics, f)

