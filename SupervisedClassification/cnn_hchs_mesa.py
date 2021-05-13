import pickle5 as pickle
# import pickle
import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
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
    with_wake = args[5] == 'True'
    dataset_name = args[6]
    disease = args[7]
    
    return testing_flag, bs, lr, max_epochs, with_wake, dataset_name, disease

def get_working_directory(testing_flag, dataset_name):
    '''Get the correct directories depending on what dataset is being used'''
    # if testing_flag:
    #     working_directory = f'{dataset_name}_testing/'
    # else:
    #     working_directory = f'{dataset_name}/'

    # if not os.path.exists(working_directory):
    #     os.makedirs(working_directory)
    
    # return working_directory
    return 'dont care'

def get_datasets_from_path(testing_flag, dataset_name, with_wake):
    working_directory = get_working_directory(testing_flag, dataset_name)

    dataset_save_path = os.path.join(os.path.dirname(os.getcwd()), "PickledData", dataset_name)

    if testing_flag:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict.pickle')
        # path_to_test_train_split_dict = os.path.join(dataset_save_path, '100_users_reduced_test_train_split_dict.pickle')
    else:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'test_train_split_dict.pickle')
    
    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)
    
    disease_user_datasets = {}
    diseases = dataset_diseases[dataset_name]
    for disease in diseases:
        if with_wake:
            disease_user_dataset_path = os.path.join(dataset_save_path, f'{disease}_with_wake_user_datasets.pickle')
        else: 
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
        if label not in [0.0, 1.0, 2.0, 3.0]:
            print(f'{user} : {label}')
            continue
        new_user_dataset[user] = data_label
    
    return new_user_dataset

def get_number_of_classes(dataset, disease):
    if dataset == 'hchs':
        if disease in ['sleep_apnea', 'hypertension', 'metabolic_syndrome']:
            return 2
        else:
            return 3
    elif dataset == 'chapman':
        return 4
    else:
        assert dataset == 'mesa'
        return 2

def get_confidence_interval_f1_micro_macro(y_true, y_pred, n_bootstraps=500):
    '''Using bootstrap with replacement calculate the f1 micro and macro scores n_bootstrap number of times to get the 
    median at the 95% confidence intervals'''
    np.random.seed(1234)
    rng=np.random.RandomState(1234)
    numbers = list(range(len(y_true)))
    bootstrapped_f1_micro_scores = []
    bootstrapped_f1_macro_scores = []
    for _ in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices 
        indices = random.choices(numbers, k=len(y_true))
        f1_micro = f1_score(y_true[indices], y_pred[indices], average='micro')
        f1_macro = f1_score(y_true[indices], y_pred[indices], average='macro')
        bootstrapped_f1_macro_scores.append(f1_macro)
        bootstrapped_f1_micro_scores.append(f1_micro)

    sorted_f1_micro_scores = np.array(bootstrapped_f1_micro_scores)
    sorted_f1_micro_scores.sort()
    sorted_f1_macro_scores = np.array(bootstrapped_f1_macro_scores)
    sorted_f1_macro_scores.sort()

    lower_micro = sorted_f1_micro_scores[int(0.05 * len(sorted_f1_micro_scores))]
    upper_micro = sorted_f1_micro_scores[int(0.95 * len(sorted_f1_micro_scores))]
    middle_micro = (lower_micro + upper_micro) / 2
    half_micro = upper_micro - middle_micro
    mean_micro = sorted_f1_micro_scores.mean()
    std_micro = sorted_f1_micro_scores.std()

    lower_macro = sorted_f1_macro_scores[int(0.05 * len(sorted_f1_macro_scores))]
    upper_macro = sorted_f1_macro_scores[int(0.95 * len(sorted_f1_macro_scores))]
    middle_macro = (lower_macro + upper_macro) / 2
    half_macro = upper_macro - middle_macro
    mean_macro = sorted_f1_macro_scores.mean()
    std_macro = sorted_f1_macro_scores.std()

    return middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro

def get_confidence_interval_auc(y_true, y_score, n_bootstraps=500):
    '''Using bootstrap with replacement calculate the auc ovo and ovr scores n_bootstrap number of times to get the 
    median at the 95% confidence intervals and the median and the std '''
    np.random.seed(1234)
    rng=np.random.RandomState(1234)
    numbers = list(range(len(y_true)))
    bootstrapped_auc_ovo_scores = []
    bootstrapped_auc_ovr_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices 
        indices = random.choices(numbers, k=len(y_true))
        if len(np.unique(y_true[indices])) < y_score.shape[1]:
            continue
        auc_ovo = roc_auc_score(y_true[indices], y_score[indices], multi_class='ovo')
        auc_ovr = roc_auc_score(y_true[indices], y_score[indices], multi_class='ovr')
        bootstrapped_auc_ovo_scores.append(auc_ovo)
        bootstrapped_auc_ovr_scores.append(auc_ovr)

    sorted_auc_ovo_scores = np.array(bootstrapped_auc_ovo_scores)
    sorted_auc_ovo_scores.sort()
    sorted_auc_ovr_scores = np.array(bootstrapped_auc_ovr_scores)
    sorted_auc_ovo_scores.sort()

    lower_ovo = sorted_auc_ovo_scores[int(0.05 * len(sorted_auc_ovo_scores))]
    upper_ovo = sorted_auc_ovo_scores[int(0.95 * len(sorted_auc_ovo_scores))]
    middle_ovo = (lower_ovo + upper_ovo) / 2
    half_ovo = upper_ovo - middle_ovo
    mean_ovo = sorted_auc_ovo_scores.mean()
    std_ovo = sorted_auc_ovo_scores.std()

    lower_ovr = sorted_auc_ovr_scores[int(0.05 * len(sorted_auc_ovr_scores))]
    upper_ovr = sorted_auc_ovr_scores[int(0.95 * len(sorted_auc_ovr_scores))]
    middle_ovr = (lower_ovr + upper_ovr) / 2
    half_ovr = upper_ovr - middle_ovr
    mean_ovr = sorted_auc_ovr_scores.mean()
    std_ovr = sorted_auc_ovr_scores.std()

    return middle_ovr, half_ovr, mean_ovr, std_ovr, middle_ovo, half_ovo, mean_ovo, std_ovo

def print_confidence_f1_scores(middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro):
    '''Print confidence scores in the following format
    f1 median +/- half 
    f1 mean +/- std
    '''
    print(f"    f1 macro middle half: {middle_macro:.3f} +/- {half_macro:.3f}")
    print(f"    f1 macro mean std: {mean_macro:.3f} +/- {std_macro:.3f}")
    print(f"    f1 micro middle half: {middle_micro:.3f} +/- {half_micro:.3f}")
    print(f"    f1 micro mean std: {mean_micro:.3f} +/- {std_micro:.3f}")

def print_confidence_auc_scores(middle_ovr, half_ovr, mean_ovr, std_ovr, middle_ovo, half_ovo, mean_ovo, std_ovo):
    '''Print confidence scores in the following format
    auc median +/- half 
    auc mean +/- std
    '''
    print(f"    auc ovr middle half: {middle_ovr:.3f} +/- {half_ovr:.3f}")
    print(f"    auc ovr mean std: {mean_ovr:.3f} +/- {std_ovr:.3f}")
    print(f"    auc ovo middle half: {middle_ovo:.3f} +/- {half_ovo:.3f}")
    print(f"    auc ovo mean std: {mean_ovo:.3f} +/- {std_ovo:.3f}")



def cnn_disease_confidence_levels(testing_flag, dataset_name, disease, bs, lr, max_epochs=20, with_wake=True, N=500):
    if dataset_name != 'chapman':
        disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, dataset_name, with_wake)
        user_datasets = disease_user_datasets[disease]
        user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
        train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
        test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')
        
        input_x = train_dataset.average_length
        if with_wake:
            input_y = 6
        else:
            input_y = 5
        number_of_classes = get_number_of_classes(dataset_name, disease)
    
    else:
        train_dataset, test_dataset = get_chapman_datasets(testing_flag)
        input_x = 2500
        input_y = 4
        number_of_classes = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=3
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        num_workers=3
    )

    if number_of_classes > 2:
        my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, output_dim=number_of_classes, dataset_name=dataset_name, disease=disease)
        model = actigraphy_utilities.fit_model_multiclass(my_nn, train_loader, val_loader, lr, max_epochs=max_epochs, number_of_classes=number_of_classes)

    else:
        my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, dataset_name=dataset_name, disease=disease)
        model = actigraphy_utilities.fit_model(my_nn, train_loader, val_loader, lr, max_epochs=max_epochs)

    save_model_directory = os.path.join("save_models", "cnn")
    try:
        if not os.path.exists(save_model_directory):
            os.makedirs(save_model_directory)
    except OSError as err:
        print(err)

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_model_directory, f'{start_time_str}-{testing_flag}-{bs}-{dataset_name}-{disease}.h5')
    torch.save(model, save_path)

    # downstream 
    val_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset),
        num_workers=3
    )
    for data, label in val_loader:
        pred = model(data.float()).detach().numpy()
        pred_single = np.argmax(pred, axis=1)
        y_true = label
        y_pred = pred_single
    # bootstrap
    if dataset_name != 'chapman':
        middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = get_confidence_interval_f1_micro_macro(y_true, y_pred, n_bootstraps=N)
        print_confidence_f1_scores(middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro)
    else:
        middle_ovr, half_ovr, mean_ovr, std_ovr, middle_ovo, half_ovo, mean_ovo, std_ovo = get_confidence_interval_auc(y_true, pred)
        print_confidence_auc_scores(middle_ovr, half_ovr, mean_ovr, std_ovr, middle_ovo, half_ovo, mean_ovo, std_ovo)

def cnn_different_percentage(testing_flag, dataset_name, disease, bs, lr, max_epochs=20, with_wake=True):
    if dataset_name != 'chapman':
        disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, dataset_name, with_wake)
        user_datasets = disease_user_datasets[disease]
        user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
        train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
        test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')
        
        input_x = train_dataset.average_length
        if with_wake:
            input_y = 6
        else:
            input_y = 5
        number_of_classes = get_number_of_classes(dataset_name, disease)
    
    else:
        train_dataset, test_dataset = get_chapman_datasets(testing_flag)
        input_x = 2500
        input_y = 4
        number_of_classes = 4

    if testing_flag:
        percentages = [0.2, 0.4, 0.6]
    else:
        percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    f1_macros_auc_ovrs = []
    f1_micros_auc_ovos = []
    
    for percentage in percentages:
        reduced_length = int(percentage * len(train_dataset))
        reduced_train_dataset = train_dataset[:reduced_length]

        train_loader = DataLoader(
            reduced_train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=3
        )
        val_loader = DataLoader(
            test_dataset,
            batch_size=bs,
            num_workers=3
        )
        
        if number_of_classes > 2:
            my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, output_dim=number_of_classes, dataset_name=dataset_name, disease=disease)
            model = actigraphy_utilities.fit_model_multiclass(my_nn, train_loader, val_loader, lr, max_epochs=max_epochs, number_of_classes=number_of_classes)

        else:
            my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, dataset_name=dataset_name, disease=disease)
            model = actigraphy_utilities.fit_model(my_nn, train_loader, val_loader, lr, max_epochs=max_epochs)

        # downstream 
        val_loader = DataLoader(
            test_dataset, 
            batch_size=len(test_dataset),
            num_workers=3
        )
        for data, label in val_loader:
            pred = model(data.float()).detach().numpy()
            pred_single = np.argmax(pred, axis=1)
            y_true = label
            y_pred = pred_single
        
        if dataset_name != 'chapman': 
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_macros_auc_ovrs.append(round(f1_macro, 2))
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_micros_auc_ovos.append(round(f1_micro, 2))
        else:
            auc_ovr = roc_auc_score(y_true, pred, multi_class='ovr')
            f1_macros_auc_ovrs.append(round(auc_ovr, 2))
            auc_ovo = roc_auc_score(y_true, pred, multi_class='ovo')
            f1_micros_auc_ovos.append(round(auc_ovo, 2))

    print(f'percentage used for training: {percentages}')
    print(f'f1 macros/auc ovrs: {f1_macros_auc_ovrs}')
    print(f'f1 micros/auc ovos: {f1_micros_auc_ovos}')

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")

    path_to_training_percentage = os.path.join(os.path.dirname(os.getcwd()), "training_percentage", dataset_name, disease, "cnn")
    if not os.path.exists(path_to_training_percentage):
        os.makedirs(path_to_training_percentage)

    save_name = f'{start_time_str}_testing-{testing_flag}_bs-{bs}_lr-{lr}_eps-{max_epochs}_different_training_percentages.pickle'
    save_path = os.path.join(path_to_training_percentage, save_name)
    print(save_path)

    with open(save_path, 'wb') as f:
        data = [percentages, f1_macros_auc_ovrs, f1_micros_auc_ovos]
        pickle.dump(data, f)

def cnn_get_representations(testing_flag, dataset_name, disease, bs, lr, max_epochs=20, with_wake=True):
    if dataset_name != 'chapman':
        disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, dataset_name, with_wake)
        user_datasets = disease_user_datasets[disease]
        user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
        train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
        test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')
        
        input_x = train_dataset.average_length
        if with_wake:
            input_y = 6
        else:
            input_y = 5
        number_of_classes = get_number_of_classes(dataset_name, disease)
    
    else:
        train_dataset, test_dataset = get_chapman_datasets(testing_flag)
        input_x = 2500
        input_y = 4
        number_of_classes = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=3
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        num_workers=3
    )

    if number_of_classes > 2:
        my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, output_dim=number_of_classes, dataset_name=dataset_name, disease=disease)
        model = actigraphy_utilities.fit_model_multiclass(my_nn, train_loader, val_loader, lr, max_epochs=max_epochs, number_of_classes=number_of_classes)

    else:
        my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, dataset_name=dataset_name, disease=disease)
        model = actigraphy_utilities.fit_model(my_nn, train_loader, val_loader, lr, max_epochs=max_epochs)

    train_model = deepcopy(model)
    test_model = deepcopy(model)

    train_embeddings = []
    def get_train_embeddings_hook(module, input, output):
        output = output.detach().numpy()
        train_embeddings.append(output)

    test_embeddings = []
    def get_test_embeddings_hook(module, input, output):
        output = output.detach().numpy()
        test_embeddings.append(output)
    
    # register forward hook on the fc1 layer 
    train_model.fc1.register_forward_hook(get_train_embeddings_hook)
    test_model.fc1.register_forward_hook(get_test_embeddings_hook)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=3
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        num_workers=3
    )
    training_labels = []
    for data, label in train_loader:
        pred = train_model(data.float())
        training_labels.append(label)

    testing_labels = []
    for data, label in val_loader:
        pred = test_model(data.float())
        testing_labels.append(label)

    
    X_train = np.concatenate(train_embeddings, axis=0)
    X_test = np.concatenate(test_embeddings, axis=0)

    y_train = np.concatenate(training_labels)
    y_test = np.concatenate(testing_labels)

    path_to_embeddings = os.path.join(os.path.dirname(os.getcwd()), "embeddings", dataset_name, disease, "cnn")
    if not os.path.exists(path_to_embeddings):
        os.makedirs(path_to_embeddings)

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")

    save_name = f'{start_time_str}_testing-{testing_flag}_bs-{bs}_lr-{lr}_eps-{max_epochs}_embeddings.pickle'
    save_path = os.path.join(path_to_embeddings, save_name)
    print(save_path)

    with open(save_path, 'wb') as f:
        data = [X_train, X_test, y_train, y_test]
        pickle.dump(data, f)
    
def multiclass_testing(testing_flag, dataset_name, disease, bs, with_wake, number_of_classes):
    if dataset_name == 'chapman':
        train_dataset, test_dataset = get_chapman_datasets(testing_flag)
    else:
        disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, dataset_name, with_wake)
        user_datasets = disease_user_datasets[disease]
        user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
        train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
        print(f'length of dataset: {len(train_dataset)}')
        half_dataset = train_dataset[int(0.5*len(train_dataset)):]
        print(len(half_dataset))
        test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=3
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        num_workers=3
    )
    if dataset_name == 'chapman':
        input_x = 2500
        input_y = 4
    else:
        input_x = train_dataset.average_length
        if with_wake:
            input_y = 6
        else:
            input_y = 5

    if number_of_classes > 2:
        my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, output_dim=number_of_classes, dataset_name=dataset_name, disease=disease)
        model = actigraphy_utilities.fit_model_multiclass(my_nn, train_loader, val_loader, 0.1, max_epochs=2, number_of_classes=number_of_classes)

    else:
        my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, dataset_name=dataset_name, disease=disease)
        model = actigraphy_utilities.fit_model(my_nn, train_loader, val_loader, 0.2, max_epochs=2)
    
    
    i = 0
    for data, label in val_loader:
        pred = model(data.float()).detach().numpy()
        print(label)
        print(pred)
        pred_single = np.argmax(pred, axis=1)

        print(pred_single)
        auc = roc_auc_score(label.detach().numpy(), pred, multi_class='ovr')
        f1_micro = f1_score(label, pred_single, average='micro')
        f1_macro = f1_score(label, pred_single, average='macro')
        print(auc, f1_macro, f1_micro)
        if i == 0:
            break 

def testing(testing_flag=True, dataset_name='mesa', disease='metabolic_syndrome', bs=128, with_wake=False):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_path(testing_flag, dataset_name, with_wake)
    user_datasets = disease_user_datasets[disease]
    user_datasets = get_rid_of_nan_labels_from_user_datasets(user_datasets)
    train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=3
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        num_workers=3
    )

    input_x = train_dataset.average_length
    if with_wake:
        input_y = 6
    else:
        input_y = 5
    my_nn = actigraphy_utilities.Net(input_x, input_y, testing_flag=testing_flag, dataset_name=dataset_name, disease=disease)
    
    model = actigraphy_utilities.fit_model(my_nn, train_loader, val_loader, 0.01, 1)

    i = 0
    for data, label in val_loader:
        pred = model(data.float())
        if i == 0:
            break 
    # val_loader = DataLoader(
    #     test_dataset,
    #     batch_size=len(test_dataset)
    # )

    # for data, label in val_loader:
    #     y_pred = model(data.float())
    #     y_pred = torch.heaviside(y_pred, values=torch.zeros(1))
    #     label = torch.unsqueeze(label, 1)
    #     y_pred = y_pred.detach().numpy()
    #     y = label.detach().numpy()
    #     f1_micro = f1_score(y, y_pred, average='micro')
    #     f1_macro = f1_score(y, y_pred, average='macro')

    #     print(f'{dataset_name} - {disease} - f1 micro: {f1_micro}')
    #     print(f'{dataset_name} - {disease} - f1 macro: {f1_macro}')

def get_chapman_datasets(testing_flag):
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

    # get 4 unique rhythms
    unique_rhythms_words = set(list(patient_to_rhythm_dict.values()))
    # {'AFIB': 0, 'SB': 1, 'SR': 2, 'GSVT': 3}
    rythm_to_label_encoding = {rhythm : index for index, rhythm in enumerate(unique_rhythms_words)}

    train_chapman_dataset = actigraphy_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'train')
    test_chapman_dataset = actigraphy_utilities.ChapmanDataset(user_datasets, patient_to_rhythm_dict, test_train_split_dict, 'test')

    return train_chapman_dataset, test_chapman_dataset

    

    

if __name__ == "__main__":
    testing_flag, bs, lr, max_epochs, with_wake, dataset_name, disease = parse_arguments(sys.argv)
    # testing_flag, bs, lr, max_epochs, with_wake, dataset_name, disease = True, 10, 0.1, 2, True, 'chapman', 'cardiac'
    print(f'{dataset_name}-{disease}')
    cnn_disease_confidence_levels(testing_flag, dataset_name, disease, bs, lr, max_epochs=max_epochs, with_wake=with_wake, N=500)
    cnn_different_percentage(testing_flag, dataset_name, disease, bs, lr, max_epochs=max_epochs, with_wake=with_wake)
    cnn_get_representations(testing_flag, dataset_name, disease, bs, lr, max_epochs=max_epochs, with_wake=with_wake)
    # multiclass_testing(testing_flag, dataset_name, disease, bs, with_wake, 4)