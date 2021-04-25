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
import ast
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

import hchs_data_pre_processing
import hchs_transformations
import simclr_models
import simclr_utitlities

diseases = ['sleep_apnea', 'insomnia']

def parse_arguments(args):
    testing_flag = args[1] == 'True'
    transformation_indices = ast.literal_eval(args[2])
    lr = ast.literal_eval(args[3])
    batch_size = ast.literal_eval(args[4])
    epoch_number = int(args[5])
    disease = args[6]
    temperature = ast.literal_eval(args[7])
    with_wake = args[8] == 'True'

    return testing_flag, transformation_indices, lr, batch_size, epoch_number, disease, temperature, with_wake

def crop_or_pad_data(data, average_length):
    data_length = data.shape[0]
    if data_length > average_length:
        crop_top = (data_length - average_length) // 2
        crop_bottom = (data_length - average_length) - crop_top
        modified_data = data[crop_top : (data_length - crop_bottom), :]
        
    else:
        padded_top = (average_length - data_length) // 2
        padded_bottom = (average_length - data_length) - padded_top
        modified_data = np.pad(data, ((padded_top, padded_bottom), (0,0)), 'constant')

    return modified_data

def get_cropped_padded_disease_dataset(disease_dataset):
    total_length = 0
    for patient_id, data_label in disease_dataset.items():
        data = data_label[0]
        total_length += data.shape[0]

    average_length = int(total_length / len(disease_dataset))

    cropped_padded_disease_dataset = {}

    for patient_id, data_label in disease_dataset.items():
        data = data_label[0]
        label = data_label[1]
        modified_data = crop_or_pad_data(data, average_length)
        cropped_padded_disease_dataset[patient_id] = [modified_data, label]
    
    return cropped_padded_disease_dataset
 
def get_datasets_from_paths(testing_flag, with_wake):
    if testing_flag:
        working_directory = 'mesa_testing/'
    else:
        working_directory = 'mesa/'
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    dataset_save_path = os.path.join(os.getcwd(), "PickledData", "mesa")
    
    # paths to user datasets with no nan values
    if testing_flag:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'reduced_test_train_split_dict.pickle')
    else:
        path_to_test_train_split_dict = os.path.join(dataset_save_path, 'test_train_split_dict.pickle')

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)

    disease_user_datasets = {}
    for disease in diseases:
        if with_wake:
            disease_user_dataset_path = os.path.join(dataset_save_path, f'{disease}_with_wake_user_datasets.pickle')
        else:
            disease_user_dataset_path = os.path.join(dataset_save_path, f'{disease}_user_datasets.pickle')
        with open(disease_user_dataset_path, 'rb') as f:
            user_dataset = pickle.load(f)
        user_dataset = get_cropped_padded_disease_dataset(user_dataset)
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

def get_mean_std_min_max_from_user_list_format(user_datasets, train_users):
    """
    Obtain and means and standard deviations from a 'user-list' dataset from training users only
    Take the mean and standard deviation for activity, white, blue, green and red light
    

    Parameters:

        user_datasets
            dataset in the 'user-list' format {user_id: [data, label]}
        
        train_users
            list or set of users (corresponding to the user_ids) from which the mean and std are extracted

    Return:
        (means, stds, mins, maxs)
            means, standard deviations, minimums and maximums of the particular users
            shape: (num_channels)

    """
    all_data = []
    for user in user_datasets.keys():
        if user in train_users:
            user_data = user_datasets[user][0]
            all_data.append(user_data)

    data_combined = np.concatenate(all_data)
    
    means = np.mean(data_combined, axis=0)
    stds = np.std(data_combined, axis=0)
    mins = np.min(data_combined, axis=0)
    maxs = np.max(data_combined, axis=0)
    
    return (list(means), list(stds), list(mins), list(maxs))

def z_normalise(data, means, stds, mins, maxs):
    """
    Z-Normalise along the column for each of the leads, based on the means and stds given
    x' = (x - mu) / std
    """
    data_copy = deepcopy(data)
    for index, values in enumerate(zip(means, stds)):
        mean = means[index]
        std = stds[index]
        data_copy[:,index] = (data_copy[:,index] - mean) / std
    
    return data_copy

def normalise(data, means, stds, mins, maxs):
    """
    Normalise along the column for each of the leads, using the min and max values seen in the train users. 
    x' = (x - x_min) / (x_max - x_min)
    """
    data_copy = deepcopy(data)
    for index, values in enumerate(zip(mins, maxs)):
        x_min = mins[index]
        x_max = maxs[index]
        data_copy[:, index] = (data_copy[:, index] - x_min) / (x_max - x_min)
    
    return data_copy

def create_train_test_datasets(disease_user_datasets, disease, test_train_split_dict, working_directory, normalisation_function=normalise):
    '''From the user dataset format, normalise the data with the train means, get the rhythm and split into test and train'''
    train_users = test_train_split_dict['train']
    test_users = test_train_split_dict['test']
    user_datasets = disease_user_datasets[disease]
    means, stds, mins, maxs = get_mean_std_min_max_from_user_list_format(user_datasets, train_users)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_labels_dict = {}
    test_labels_dict = {}

    for patient_id, data_label in user_datasets.items():
        data = data_label[0]
        label = data_label[1]
        if not (label == 0.0 or label == 1.0):
            print(f'{patient_id} : {label}')
            continue
        
        normalised_data = normalisation_function(data, means, stds, mins, maxs)
        if patient_id in train_users:
            train_data.append(normalised_data)
            train_labels.append(label)
            train_labels_dict[patient_id] = label
        else:
            test_data.append(normalised_data)
            test_labels.append(label)
            test_labels_dict[patient_id] = label

    
    np_train_data = np.rollaxis(np.dstack(train_data), -1)
    np_test_data = np.rollaxis(np.dstack(test_data), -1)

    return np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict

def train_simclr(testing_flag, np_train, transformation_indices=[0,1], lr=0.01, batch_size=128, epoch_number=100, temperature=0.1):
    decay_steps = 1000
    if testing_flag:
        epochs = 1
    else:
        epochs = epoch_number

    input_shape = (np_train.shape[1], np_train.shape[2])
    print(f'input_shape: {input_shape}')

    transform_funcs_vectorised = [
        hchs_transformations.noise_transform_vectorized, 
        hchs_transformations.scaling_transform_vectorized, 
        hchs_transformations.negate_transform_vectorized, 
        hchs_transformations.time_flip_transform_vectorized, 
        hchs_transformations.time_segment_permutation_transform_improved, 
        hchs_transformations.time_warp_transform_low_cost, 
        hchs_transformations.channel_shuffle_transform_vectorized
    ]

    transform_funcs_names = ['noised', 'scaled', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']
    tf.keras.backend.set_floatx('float32')

    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    transformation_function = simclr_utitlities.generate_combined_transform_function(transform_funcs_vectorised, indices=transformation_indices)

    base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
    simclr_model = simclr_models.attach_simclr_head(base_model)
    # simclr_model.summary()

    trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train, optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    return trained_simclr_model, epoch_losses    

def downstream_evaluation(trained_simclr_model, np_train, np_test, train_labels, test_labels, intermediate_layer=7):
    intermediate_model = simclr_models.extract_intermediate_model_from_base_model(trained_simclr_model, intermediate_layer=7)
    # intermediate_model.summary()
    X_train = intermediate_model.predict(np_train)
    X_test = intermediate_model.predict(np_test)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

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
    metrics['accuracy'] = accuracy

    return metrics            

def main(testing_flag, transformation_indices, lr, batch_size, epoch_number, temperature, with_wake):
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag, with_wake)
    disease_metrics = {}
    for disease in diseases:
        np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict = create_train_test_datasets(disease_user_datasets, disease, test_train_split_dict, working_directory)
        trained_simclr_model, epoch_losses = train_simclr(testing_flag, np_train_data, transformation_indices=transformation_indices, lr=lr, batch_size=batch_size, epoch_number=epoch_number, temperature=temperature)
        metrics = downstream_evaluation(trained_simclr_model, np_train_data, np_test_data, train_labels, test_labels)
        disease_metrics[disease] = metrics
        
        print(disease)
        print(metrics)
        print('***********')

    for disease, metrics in disease_metrics.items():
        print('***********')
        print(f"{disease} f1 micro: {metrics['f1_micro']}")
        print(f"{disease} f1 macro: {metrics['f1_macro']}")

    return disease_metrics

def disease_main(testing_flag, transformation_indices, lr, batch_size, epoch_number, disease, temperature, with_wake):
    print(f'starting with_wake: {with_wake}')
    disease_user_datasets, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag, with_wake)
    disease_metrics = {}
    print('getting datasets')
    np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict = create_train_test_datasets(disease_user_datasets, disease, test_train_split_dict, working_directory)
    print('training simclr')
    trained_simclr_model, epoch_losses = train_simclr(testing_flag, np_train_data, transformation_indices=transformation_indices, lr=lr, batch_size=batch_size, epoch_number=epoch_number, temperature=temperature)
    print('downstream evaluation')
    metrics = downstream_evaluation(trained_simclr_model, np_train_data, np_test_data, train_labels, test_labels)
    disease_metrics[disease] = metrics
    print('***********')
    print(f"    {disease} f1 micro: {metrics['f1_micro']}")
    print(f"    {disease} f1 macro: {metrics['f1_macro']}")
    
def try_different_transformations(testing_flag, transformation_indices_list, lr, batch_size, epoch_number, temperature, with_wake):
    disease_metrics_list = []

    for transformation_indices in transformation_indices_list:
        disease_metrics = main(testing_flag, transformation_indices, lr, batch_size, epoch_number, temperature, with_wake)
        disease_metrics_list.append(disease_metrics)
        
        print(f'transformation indices: {tranformation_indices}')
        print(disease_metrics)
        print('********')

    print('*********')
    for transformation_indices, disease_metrics in zip(transformation_indicies_list, disease_metrics_list):
        print(f'transformation indicies : {transformation_indices}')
        for disease, metrics in disease_metrics.items():
            f1_micro = metrics['f1_micro']
            f1_macro = metrics['f1_macro']
            print(f'    {disease}')
            print(f'        f1 macro: {f1_macro}')
            print(f'        f1 micro: {f1_micro}')

def try_different_temperatures(testing_flag, transformation_indices, lr, batch_size, epoch_number, temperatures, with_wake):
    disease_metrics_list = []

    for temperature in temperatures:
        disease_metrics = main(testing_flag, transformation_indices, lr, batch_size, epoch_number, temperature, with_wake)
        disease_metrics_list.append(disease_metrics)
        
        print(f'tempature: {temperature}')
        print(disease_metrics)
        print('********')

    print('*********')
    for temperature, disease_metrics in zip(temperature, disease_metrics_list):
        print(f'tempature: {temperature}')
        for disease, metrics in disease_metrics.items():
            f1_micro = metrics['f1_micro']
            f1_macro = metrics['f1_macro']
            print(f'    {disease}')
            print(f'        f1 macro: {f1_macro}')
            print(f'        f1 micro: {f1_micro}')

def try_different_batch_sizes(testing_flag, transformation_indices, lr, batch_sizes, epoch_number, temperatures, with_wake):
    disease_metrics_list = []

    for batch_size in batch_sizes:
        disease_metrics = main(testing_flag, transformation_indices, lr, batch_size, epoch_number, temperatures, with_wake)
        disease_metrics_list.append(disease_metrics)
        
        print(f'batch size: {batch_size}')
        print(disease_metrics)
        print('********')

    print('*********')
    for batch_size, disease_metrics in zip(batch_sizes, disease_metrics_list):
        print(f'batch size: {batch_size}')
        for disease, metrics in disease_metrics.items():
            f1_micro = metrics['f1_micro']
            f1_macro = metrics['f1_macro']
            print(f'    {disease}')
            print(f'        f1 macro: {f1_macro}')
            print(f'        f1 micro: {f1_micro}')

def try_different_learning_rates(testing_flag, transformation_indices, learning_rates, batch_size, epoch_number, temperatures, with_wake):
    disease_metrics_list = []

    for lr in learning_rates:
        disease_metrics = main(testing_flag, transformation_indices, lr, batch_size, epoch_number, temperatures, with_wake)
        disease_metrics_list.append(disease_metrics)
        
        print(f'learning rate: {lr}')
        print(disease_metrics)
        print('********')

    print('*********')
    for lr, disease_metrics in zip(learning_rates, disease_metrics_list):
        print(f'learning rates: {lr}')
        for disease, metrics in disease_metrics.items():
            f1_micro = metrics['f1_micro']
            f1_macro = metrics['f1_macro']
            print(f'    {disease}')
            print(f'        f1 macro: {f1_macro}')
            print(f'        f1 micro: {f1_micro}')
            
if __name__ == "__main__":
    testing_flag, transformation_indices, lr, batch_size, epoch_number, disease, temperature, with_wake = parse_arguments(sys.argv)
    # testing_flag, transformation_indices, lr, batch_size, epoch_number, disease, temperature, with_wake = True, [0,1], 0.1, 128, 1, 'sleep_apnea', 0.1, True
    # main(testing_flag, transformation_indices, lr, batch_size, epoch_number)
    # try_different_transformations(testing_flag, transformation_indices_list=transformation_indices, lr, batch_size, epoch_number, temperature, with_wake)
    # try_different_batch_sizes(testing_flag, transformation_indices, lr, batch_sizes=batch_size, epoch_number, temperature, with_wake)  
    # try_different_learning_rates(testing_flag, transformation_indices, learning_rates=lr, batch_size, epoch_number, temperature, with_wake)
    disease_main(testing_flag, transformation_indices, lr, batch_size, epoch_number, disease, temperature, with_wake)