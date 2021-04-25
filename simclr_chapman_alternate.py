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

import chapman_data_pre_processing
import chapman_transformations
import simclr_models
import simclr_utitlities
from lars_optimizer import LARSOptimizer


def parse_arguments(args):
    testing_flag = args[1] == 'True'
    transformation_indices = ast.literal_eval(args[2])
    lr = ast.literal_eval(args[3])
    batch_size = ast.literal_eval(args[4])
    epoch_number = int(args[5])
    
    return testing_flag, transformation_indices, lr, batch_size, epoch_number

def get_datasets_from_paths(testing_flag):
    if testing_flag:
        working_directory = 'chapman_testing/'
    else:
        working_directory = 'chapman/'
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    dataset_save_path = os.path.join(os.getcwd(), "PickledData", "chapman")
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

def create_train_test_datasets(user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory, normalisation_function=normalise):
    '''From the user dataset format, normalise the data with the train means, get the rhythm and split into test and train'''
    train_users = test_train_split_dict['train']
    test_users = test_train_split_dict['test']
    means, stds, mins, maxs = get_mean_std_min_max_from_user_list_format(user_datasets, train_users)

    unique_rhythms_words = set(list(patient_to_rhythm_dict.values()))
    rythm_to_label_encoding = {rhythm : index for index, rhythm in enumerate(unique_rhythms_words)}
        
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_labels_dict = {}
    test_labels_dict = {}

    for patient_id, data_label in user_datasets.items():
        data = data_label[0]
        normalised_data = normalisation_function(data, means, stds, mins, maxs)
        rhythm = patient_to_rhythm_dict[patient_id]
        rhythm_label = rythm_to_label_encoding[rhythm]
        if patient_id in train_users:
            train_data.append(normalised_data)
            train_labels.append(rhythm_label)
            train_labels_dict[patient_id] = rhythm_label
        else:
            test_data.append(normalised_data)
            test_labels.append(rhythm_label)
            test_labels_dict[patient_id] = rhythm_label

    # convert list of data to single 3D array shape: (len(train_data/test_data), 2500, 4)
    np_train_data = np.rollaxis(np.dstack(train_data), -1)
    np_test_data = np.rollaxis(np.dstack(test_data), -1)

    return np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict

def train_simclr(testing_flag, np_train, transformation_indices=[0,1], lr=0.01, batch_size=128, epoch_number=100):
    decay_steps = 1000
    if testing_flag:
        epochs = 1
    else:
        epochs = epoch_number

    temperature = 0.1

    input_shape = (np_train.shape[1], np_train.shape[2])

    transform_funcs_vectorised = [
        chapman_transformations.noise_transform_vectorized, 
        chapman_transformations.scaling_transform_vectorized, 
        chapman_transformations.negate_transform_vectorized, 
        chapman_transformations.time_flip_transform_vectorized, 
        chapman_transformations.time_segment_permutation_transform_improved, 
        chapman_transformations.time_warp_transform_low_cost, 
        chapman_transformations.channel_shuffle_transform_vectorized
    ]

    transform_funcs_names = ['noised', 'scaled', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']
    tf.keras.backend.set_floatx('float32')

    # lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=decay_steps)
    # optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    learning_rate = 0.3 * (batch_size / 256)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
    optimizer = LARSOptimizer(learning_rate, weight_decay=0.000001)

    transformation_function = simclr_utitlities.generate_combined_transform_function(transform_funcs_vectorised, indices=transformation_indices)

    base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
    simclr_model = simclr_models.attach_simclr_head(base_model)
    # simclr_model.summary()

    trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train, optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    return trained_simclr_model, epoch_losses

def get_confidence_interval_auc(y_true, y_pred, n_bootstraps=500):
    '''Using bootstrap with replacement calculate the f1 micro and macro scores n_bootstrap number of times to get the 
    median at the 95% confidence intervals'''
    np.random.seed(1234)
    rng=np.random.RandomState(1234)
    bootstrapped_auc_ovo_scores = []
    bootstrapped_auc_ovr_scores = []
    for _ in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices 
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        # we need at least one positive and one negative sample for ROC AUC 
        if len(np.unique(y_true[indices])) < 2:
            # reject the sample
            continue 

        auc_ovo = roc_auc_score(y_true[indices], y_pred[indices], multi_class='ovo')
        auc_ovr = roc_auc_score(y_true[indices], y_pred[indices], multi_class='ovr')
        bootstrapped_auc_ovo_scores.append(auc_ovo)
        bootstrapped_auc_ovr_scores.append(auc_ovr)

    sorted_auc_ovo_scores = np.array(bootstrapped_auc_ovo_scores)
    sorted_auc_ovo_scores.sort()
    sorted_auc_ovr_scores = np.array(bootstrapped_auc_ovr_scores)
    sorted_auc_ovr_scores.sort()

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

    return middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr 

def different_percentage_training(X_train, X_test, y_train, y_test):
    '''Given train and test representations with labelled data, use different percentages of the train data to 
    train the logistic regression classifier'''
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_length = X_train.shape[0]
    auc_ovo_scores = []
    auc_ovr_scores = []
    for percentage in percentages:
        log_reg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        # randomly select subsection of X_test to train the classifier with 
        number_of_rows = int(percentage*train_length)
        idx = [0]
        while len(np.unique(y_train[idx])) < 2:
            idx = np.random.choice(train_length, number_of_rows, replace=False)
        new_X_train = X_train[idx, :]
        new_y_train = y_train[idx]
        log_reg_clf.fit(new_X_train, new_y_train)
        # predict X_test with the trained classifier
        y_proba = log_reg_clf.predict_proba(X_test)
        auc_ovo = roc_auc_score(y_test, y_proba, multiclass='ovo')
        auc_ovr = roc_auc_score(y_test, y_proba, multiclass='ovr')
        auc_ovo_scores.append(auc_ovo)
        auc_ovr_scores.append(auc_ovr)

    for percentage, auc_ovo, auc_ovr in zip(percentages, auc_ovo_scores, auc_ovr_scores):
        print(f'training percentage: {percentage}')
        print(f'   auc ovo: {auc_ovo}')
        print(f'   auc ovr: {auc_ovr}')

    print(percentages)
    print(auc_ovo_scores)
    print(auc_ovr_scores)
    
    return percentages, auc_ovo_scores, auc_ovr_scores

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
    y_proba = log_reg_clf.predict_proba(X_test)
    auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr')
    auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo')
    
    metrics['accuracy'] = accuracy
    metrics['auc_ovr'] = auc_ovr
    metrics['auc_ovo'] = auc_ovo

    # bootstrap auc 
    middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr = get_confidence_interval_auc(y_test, y_proba)
    # test different percentage training 
    percentages, auc_ovo_scores, auc_ovr_scores = different_percentage_training(X_train, X_test, y_train, y_test)
    
    return metrics, middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr

def try_different_transformations(testing_flag, transformation_indices, lr, batch_size, epoch_number):
    auc_scores = []
    auc_interval_scores = []
    for indices in transformation_indices:
        user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag)
        np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict = create_train_test_datasets(user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory)
        trained_simclr_model, epoch_losses = train_simclr(testing_flag, np_train_data, transformation_indices=indices, lr=lr, batch_size=batch_size, epoch_number=epoch_number)
        metrics, middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr = downstream_evaluation(trained_simclr_model, np_train_data, np_test_data, train_labels, test_labels)
        print(f'{indices}: {metrics}')
        auc_scores.append(metrics['auc_ovr'])
        auc_interval_scores.append([middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr])

    print('*******')
    for indices, score, interval_scores in zip(transformation_indices, auc_scores, auc_interval_scores):
        print(f'{indices}: {score}')
        print_auc_intervals(*interval_scores)
        print('----------------')

def try_different_batch_sizes(testing_flag, transformation_indices, lr, batch_sizes, epoch_number):
    auc_scores = []
    auc_interval_scores = []
    for bs in batch_sizes:
        user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag)
        np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict = create_train_test_datasets(user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory)
        trained_simclr_model, epoch_losses = train_simclr(testing_flag, np_train_data, transformation_indices=transformation_indices, lr=lr, batch_size=bs, epoch_number=epoch_number)
        metrics, middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr = downstream_evaluation(trained_simclr_model, np_train_data, np_test_data, train_labels, test_labels)
        print(f'{bs}: {metrics}')
        auc_scores.append(metrics['auc_ovr'])
        auc_interval_scores.append([middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr])

    print('*******')
    for bs, score, interval_scores in zip(batch_sizes, auc_scores, auc_interval_scores):
        print(f'{bs}: {score}')
        print_auc_intervals(*interval_scores)
        print('----------------')

def try_different_learning_rates(testing_flag, transformation_indices, lrs, batch_size, epoch_number):
    auc_scores = []
    auc_interval_scores = []
    for lr in lrs:
        user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag)
        np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict = create_train_test_datasets(user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory)
        trained_simclr_model, epoch_losses = train_simclr(testing_flag, np_train_data, transformation_indices=transformation_indices, lr=lr, batch_size=batch_size, epoch_number=epoch_number)
        metrics, middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr = downstream_evaluation(trained_simclr_model, np_train_data, np_test_data, train_labels, test_labels)
        print(f'{lr}: {metrics}')
        auc_scores.append(metrics['auc_ovr'])
        auc_interval_scores.append([middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr])

    print('*******')
    for lr, score, interval_scores in zip(lrs, auc_scores, auc_interval_scores):
        print(f'{lr}: {score}')
        print_auc_intervals(*interval_scores)
        print('----------------')

def main(testing_flag, transformation_indices, lr, batch_size, epoch_number):
    user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory = get_datasets_from_paths(testing_flag)
    np_train_data, train_labels, train_labels_dict, np_test_data, test_labels, test_labels_dict = create_train_test_datasets(user_datasets, patient_to_rhythm_dict, test_train_split_dict, working_directory)

    trained_simclr_model, epoch_losses = train_simclr(testing_flag, np_train_data, transformation_indices=transformation_indices, lr=lr, batch_size=batch_size, epoch_number=epoch_number)
    
    metrics, middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr = downstream_evaluation(trained_simclr_model, np_train_data, np_test_data, train_labels, test_labels)
    print(metrics)
    print_auc_intervals(middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr)

def print_auc_intervals(middle_ovo, half_ovo, mean_ovo, std_ovo, middle_ovr, half_ovr, mean_ovr, std_ovr):
    '''Print auc intervals in the following format:
    mean +/- std
    median +/- half_range'''
    print(f"    auc ovo middle half: {middle_ovo:.2f} +/- {half_ovo:.2f}")
    print(f"    auc ovo mean std: {mean_ovo:.2f} +/- {std_ovo:.2f}")
    print(f"    auc ovr middle half: {middle_ovr:.2f} +/- {half_ovr:.2f}")
    print(f"    auc ovr mean std: {mean_ovr:.2f} +/- {std_ovr:.2f}")


if __name__ == "__main__":
    # testing_flag, transformation_indices, lr, batch_size, epoch_number = True, [[0,1], [0, 2]], 0.01, 128, 100
    testing_flag, transformation_indices, lr, batch_size, epoch_number = parse_arguments(sys.argv)
    main(testing_flag, transformation_indices, lr, batch_size, epoch_number)
    # try_different_transformations(testing_flag, transformation_indices, lr, batch_size, epoch_number)
    # try_different_batch_sizes(testing_flag, transformation_indices, lr, batch_size, epoch_number)
    # try_different_learning_rates(testing_flag, transformation_indices, lr, batch_size, epoch_number)