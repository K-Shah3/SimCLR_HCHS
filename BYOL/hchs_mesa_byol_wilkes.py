# import pickle 
import pickle5 as pickle
import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
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
    batch_size = int(args[2])
    epoch_number = int(args[3])
    latent_dim = int(args[4])
    projection_dim = int(args[5])
    segment_number = int(args[6])
    main_type = args[7]
    hchs_or_mesa = args[8]
    disease = args[9]
    
    return testing_flag, batch_size, epoch_number, latent_dim, projection_dim, segment_number, main_type, hchs_or_mesa, disease

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
    path_to_embeddings = os.path.join(os.path.dirname(os.getcwd()), "embeddings", hchs_or_mesa)
    
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

    return disease_user_datasets, test_train_split_dict, working_directory, path_to_embeddings

def byol_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease):
    disease_user_datasets, test_train_split_dict, working_directory, path_to_embeddings = get_datasets_from_path(testing_flag, hchs_or_mesa)
    path_to_embeddings = os.path.join(path_to_embeddings, disease, "byol")
    if not os.path.exists(path_to_embeddings):
        os.makedirs(path_to_embeddings)

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
        shuffle=True,
        num_workers=28
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=28
    )

    byol_model = deepcopy(model)
    image_size = (train_dataset.average_length, 5)
    byol = actigraphy_utilities.BYOL(byol_model, image_size=image_size, projection_size=projection_dim, batch_size=batch_size)
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False
    )
    byol_trainer.fit(byol, train_loader, val_loader) 
    # byol_encoder = byol.encoder
    state_dict = byol_model.state_dict()
    byol_encoder = deepcopy(encoder)
    byol_encoder.load_state_dict(state_dict)

    return byol_encoder, test_dataset, train_dataset, working_directory, path_to_embeddings

def downstream_evaluation(byol_encoder, test_dataset, train_dataset, path_to_embeddings, save_name):
    '''Given trained byol encoder, use the representations to test how well it performs on a 
    downstream disease classification'''
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=28)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=28)                         

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

    save_name = f'{save_name}-train_test.pickle'
    save_path = os.path.join(path_to_embeddings, save_name)
    with open(save_path, 'wb') as f:
        data = [X_train, X_test, y_train, y_test]
        pickle.dump(data, f)

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

    middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = get_confidence_interval_f1_micro_macro(y_test, y_pred)
    percentages, f1_macro_scores, f1_micro_scores = different_percentage_training(X_train, X_test, y_train, y_test)

    return metrics, middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro

def print_confidence_f1_scores(middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro):
    '''Print confidence scores in the following format
    f1 median +/- half 
    f1 mean +/- std
    '''
    print(f"    f1 macro middle half: {middle_macro:.2f} +/- {half_macro:.2f}")
    print(f"    f1 macro mean std: {mean_macro:.2f} +/- {std_macro:.2f}")
    print(f"    f1 micro middle half: {middle_micro:.2f} +/- {half_micro:.2f}")
    print(f"    f1 micro mean std: {mean_micro:.2f} +/- {std_micro:.2f}")

def get_confidence_interval_f1_micro_macro(y_true, y_pred, n_bootstraps=500):
    '''Using bootstrap with replacement calculate the f1 micro and macro scores n_bootstrap number of times to get the 
    median at the 95% confidence intervals'''
    np.random.seed(1234)
    rng=np.random.RandomState(1234)
    bootstrapped_f1_micro_scores = []
    bootstrapped_f1_macro_scores = []
    for _ in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices 
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
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

def combined_main(testing_flag, batch_size, epoch_number, latent_dim, projection_dim, disease='insomnia'):
    mesa_disease_user_datasets, mesa_test_train_split_dict, mesa_working_directory, path_to_embeddings = get_datasets_from_path(testing_flag, 'mesa')
    hchs_disease_user_datasets, hchs_test_train_split_dict, hchs_working_directory, path_to_embeddings = get_datasets_from_path(testing_flag, 'hchs')
    
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
    byol = actigraphy_utilities.BYOL(byol_model, image_size=image_size, projection_size=projection_dim, batch_size=batch_size)
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False
    )
    byol_trainer.fit(byol, train_loader, val_loader) 
    # byol_encoder = byol.encoder
    state_dict = byol_model.state_dict()
    byol_encoder = deepcopy(encoder)
    byol_encoder.load_state_dict(state_dict)

    return byol_encoder, mesa_working_directory, hchs_working_directory, train_dataset.average_length, path_to_embeddings

def combined_downstream_evaluation(hchs_or_mesa, byol_encoder, disease, average_length):
    disease_user_datasets, test_train_split_dict, working_directory, path_to_embeddings = get_datasets_from_path(testing_flag, hchs_or_mesa)
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
    metrics['accuracy'] = accuracy

    middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = get_confidence_interval_f1_micro_macro(y_test, y_pred)
    percentages, f1_macro_scores, f1_micro_scores = different_percentage_training(X_train, X_test, y_train, y_test)

    return metrics, middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro 

def multiple_segment_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease, segment_number):
    disease_user_datasets, test_train_split_dict, working_directory, path_to_embeddings = get_datasets_from_path(testing_flag, hchs_or_mesa)
    path_to_embeddings = os.path.join(path_to_embeddings, disease, "byol")
    if not os.path.exists(path_to_embeddings):
        os.makedirs(path_to_embeddings)

    user_datasets = disease_user_datasets[disease]

    autoencoder, encoder, decoder = actigraphy_utilities.get_trained_autoencoder_ms(user_datasets, test_train_split_dict, batch_size, latent_dim, segment_number)
    model = deepcopy(encoder)

    train_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = actigraphy_utilities.ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=28
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=28
    )

    byol_model = deepcopy(model)
    image_size = (train_dataset.average_length // segment_number, 5)
    byol = actigraphy_utilities.BYOL_MS(byol_model, image_size=image_size, projection_size=projection_dim, segment_number=segment_number, batch_size=batch_size)
    byol_trainer = pl.Trainer(
        max_epochs=epoch_number,
        accumulate_grad_batches=2048 // batch_size,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False
    )
    byol_trainer.fit(byol, train_loader, val_loader) 
    # byol_encoder = byol.encoder
    state_dict = byol_model.state_dict()
    byol_encoder = deepcopy(encoder)
    byol_encoder.load_state_dict(state_dict)

    return byol_encoder, test_dataset, train_dataset, working_directory, path_to_embeddings

def different_percentage_training(X_train, X_test, y_train, y_test):
    '''Given train and test representations with labelled data, use different percentages of the train data to 
    train the logistic regression classifier'''
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_length = X_train.shape[0]
    f1_micro_scores = []
    f1_macro_scores = []
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
        y_pred = log_reg_clf.predict(X_test)
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro_scores.append(f1_micro)
        f1_macro_scores.append(f1_macro)

    for percentage, f1_macro, f1_micro in zip(percentages, f1_macro_scores, f1_micro_scores):
        print(f'training percentage: {percentage}')
        print(f'   f1 macro: {f1_macro}')
        print(f'   f1 micro: {f1_micro}')

    print(percentages)
    print(f1_macro_scores)
    print(f1_micro_scores)
    
    return percentages, f1_macro_scores, f1_micro_scores

def downstream_evaluation_ms(byol_encoder, test_dataset, train_dataset, segment_number, path_to_embeddings, save_name):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=28)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=28)

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

    save_name = f'{save_name}-train_test.pickle'
    save_path = os.path.join(path_to_embeddings, save_name)
    with open(save_path, 'wb') as f:
        data = [X_train, X_test, y_train, y_test]
        pickle.dump(data, f)

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

    middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = get_confidence_interval_f1_micro_macro(y_test, y_pred)    

    percentages, f1_macro_scores, f1_micro_scores = different_percentage_training(X_train, X_test, y_train, y_test)

    return metrics, middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro


if __name__ == "__main__":
    testing_flag, batch_size, epoch_number, latent_dim, projection_dim, segment_number, main_type, hchs_or_mesa, disease = parse_arguments(sys.argv)
    # testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, segment_number, main_type = True, 'mesa', 512, 1, 512, 128, 2, 'normal'

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    print(start_time_str)

    if main_type == 'normal':
        save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-{hchs_or_mesa}-{disease}'
        byol_encoder, test_dataset, train_dataset, working_directory, path_to_embeddings = byol_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease)
        metrics, middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = downstream_evaluation(byol_encoder, test_dataset, train_dataset, path_to_embeddings, save_name)
        print(f'{disease}: {metrics}')
        print_confidence_f1_scores(middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro)

        save_name = f'{save_name}-metrics.pickle'
        save_path = os.path.join(working_directory, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)

        print(save_name)

    elif main_type == 'ms':
        save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-{segment_number}-{hchs_or_mesa}-{disease}-ms'
        byol_encoder, test_dataset, train_dataset, working_directory, path_to_embeddings = multiple_segment_main(testing_flag, hchs_or_mesa, batch_size, epoch_number, latent_dim, projection_dim, disease, segment_number)
        metrics, middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = downstream_evaluation_ms(byol_encoder, test_dataset, train_dataset, segment_number, path_to_embeddings, save_name)
        print(f'{disease}: {metrics}')
        print_confidence_f1_scores(middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro)

        save_name = f'{save_name}-metrics.pickle'
        save_path = os.path.join(working_directory, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)

        print(save_name)

    else:
        assert main_type == 'combined'
        byol_encoder, mesa_working_directory, hchs_working_directory, average_length, path_to_embeddings = combined_main(testing_flag, batch_size, epoch_number, latent_dim, projection_dim)
        diseases = dataset_diseases[hchs_or_mesa]
        all_metrics = {}
        for disease in diseases:
            metrics, middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro = combined_downstream_evaluation(hchs_or_mesa, byol_encoder, disease, average_length)
            all_metrics[disease] = metrics 
            print(f'{disease}: {metrics}')
            print_confidence_f1_scores(middle_macro, half_macro, mean_macro, std_macro, middle_micro, half_micro, mean_micro, std_micro)
        
        save_name = f'{start_time_str}-{testing_flag}-{batch_size}-{epoch_number}-{latent_dim}-{projection_dim}-{hchs_or_mesa}-combined-metrics.pickle'
        if hchs_or_mesa == 'hchs':
            working_directory = hchs_working_directory
        else:
            working_directory = mesa_working_directory
        save_path = os.path.join(working_directory, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(all_metrics, f)

