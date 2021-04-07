import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
import pickle as pickle
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
# from torchvision.models import resnet18
import pytorch_lightning as pl
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression


import byol_chapman_utilities
import chapman_autoencoder 

def parse_arguments(args):
    testing_flag = args[1] == 'True'
    batch_size = int(args[2])
    
    return testing_flag, batch_size

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

def resnet_main(testing_flag, batch_size):
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
        batch_size=batch_size,
    )
    print('got here')
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model_name = 'resnet18'

    # supervised learning before byol 
    supervised_model = deepcopy(model)
    supervised = byol_chapman_utilities.SupervisedLightningModule(supervised_model)
    supervised_trainer = pl.Trainer(max_epochs=25, weights_summary=None)
    supervised_trainer.fit(supervised, train_loader, val_loader)
    supervised_accuracy = byol_chapman_utilities.accuracy_from_val_loader_and_model(val_loader, supervised_model)

    # byol training model 
    byol_model = deepcopy(model)
    byol = byol_chapman_utilities.BYOL(byol_model, image_size=(2500, 4))
    byol_trainer = pl.Trainer(
        max_epochs=10,
        accumulate_grad_batches=2048 // batch_size,
        weights_summary=None,
    )
    byol_trainer.fit(byol, train_loader, val_loader) 

    # supervised learning again after byol 
    state_dict = byol_model.state_dict()
    post_byol_model = deepcopy(model)
    post_byol_model.load_state_dict(state_dict)
    post_byol_supervised = byol_chapman_utilities.SupervisedLightningModule(post_byol_model)
    post_byol_trainer = pl.Trainer(
        max_epochs=10,
        accumulate_grad_batches=2048 // 128,
        weights_summary=None,
    )
    post_byol_trainer.fit(post_byol_supervised, train_loader, val_loader)
    post_byol_accuracy = byol_chapman_utilities.accuracy_from_val_loader_and_model(val_loader, post_byol_model)

    # final results 
    print(f'supervised accuracy - {supervised_accuracy}')
    print(f'post byol supervised accuracy - {post_byol_accuracy}')

    save_dict = {'supervised_acc' : supervised_accuracy, 
                'post_byol_acc' : post_byol_accuracy}

    # save results
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")

    save_filename = f'{testing_flag}-{batch_size}-{model_name}-{start_time_str}-byol-chapman.pickle'
    save_path = os.path.join(working_directory, save_filename)

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

def autoencoder_main(testing_flag, batch_size, epoch_number):
    # get trained autoencoder 
    autoencoder, encoder, decoder = chapman_autoencoder.get_trained_autoencoder(testing_flag)
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
        batch_size=batch_size,
    )

    # byol training model 
    byol_model = deepcopy(model)
    byol = byol_chapman_utilities.BYOL(byol_model, image_size=(2500, 4))
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        accumulate_grad_batches=2048 // batch_size,
        weights_summary=None,
    )
    byol_trainer.fit(byol, train_loader, val_loader) 

    byol_encoder = byol.encoder

    return byol_encoder, test_chapman_dataset, train_chapman_dataset

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
    y_proba = log_reg_clf.predict_proba(X_test)
    roc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
    roc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')
    
    metrics['accuracy'] = accuracy
    metrics['roc_ovr'] = roc_ovr
    metrics['roc_ovo'] = roc_ovo

    return metrics

def multiple_segment_main(testing_flag, batch_size, epoch_number):
    autoencoder, encoder, decoder = chapman_autoencoder.get_trained_autoencoder_ms(testing_flag)
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
        batch_size=batch_size,
    )

    # byol training model 
    byol_model = deepcopy(model)
    byol = byol_chapman_utilities.BYOL_MS(byol_model, image_size=(1250, 4))
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        accumulate_grad_batches=2048 // batch_size,
        weights_summary=None,
    )
    byol_trainer.fit(byol, train_loader, val_loader) 

    byol_encoder = byol.encoder

    return byol_encoder, test_chapman_dataset, train_chapman_dataset

def downstream_evaluation_ms(byol_encoder, test_chapman_dataset, train_chapman_dataset):
    train_loader = DataLoader(train_chapman_dataset, batch_size=len(train_chapman_dataset))
    test_loader = DataLoader(test_chapman_dataset, batch_size=len(test_chapman_dataset))
    for data_label in train_loader:
        data, label = data_label
        data1, data2 = torch.split(data, 1250, dim=2)
        encoded_data1, encoded_data2 = byol_encoder(data1.float()), byol_encoder(data2.float())
        encoded_data1_numpy, encoded_data2_numpy = encoded_data1.detach().numpy(), encoded_data2.detach().numpy()
        # take mean of encoded segments 
        byol_train_numpy_x = np.mean(np.array([encoded_data1_numpy,encoded_data2_numpy]), axis=0)
        byol_train_numpy_y = label.detach().numpy()

    X_train, y_train = byol_train_numpy_x, byol_train_numpy_y

    for data_label in test_loader:
        data, label = data_label
        data1, data2 = torch.split(data, 1250, dim=2)
        encoded_data1, encoded_data2 = byol_encoder(data1.float()), byol_encoder(data2.float())
        encoded_data1_numpy, encoded_data2_numpy = encoded_data1.detach().numpy(), encoded_data2.detach().numpy()
        # take mean of encoded segments 
        byol_test_numpy_x = np.mean(np.array([encoded_data1_numpy,encoded_data2_numpy]), axis=0)
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
    y_proba = log_reg_clf.predict_proba(X_test)
    roc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
    roc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')
    
    metrics['accuracy'] = accuracy
    metrics['roc_ovr'] = roc_ovr
    metrics['roc_ovo'] = roc_ovo

    return metrics


if __name__ == '__main__':
    # parse arguments
    # testing_flag, batch_size = parse_arguments(sys.argv)
    # byol_encoder, test_chapman_dataset, train_chapman_dataset = autoencoder_main(True, 128, 2)
    # metrics = downstream_evaluation(byol_encoder, test_chapman_dataset, train_chapman_dataset)
    # print(metrics)
    byol_encoder, test_chapman_dataset, train_chapman_dataset = multiple_segment_main(True, 128, 3)
    metrics = downstream_evaluation_ms(byol_encoder, test_chapman_dataset, train_chapman_dataset)
    print(metrics)