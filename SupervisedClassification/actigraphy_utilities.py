import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
import numpy as np
from copy import deepcopy
from itertools import chain, combinations
import datetime

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as func
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import tensorflow as tf
import math
import functools
import operator 

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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall

import matplotlib.pyplot as plt

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

class ActigraphyDataset(Dataset):
    def __init__(self, user_datasets, test_train_split_dict, train_or_test, normalisation_function=normalise):
        self.np_samples = []
        self.samples = []
        self.max_length = - math.inf
        self.total_length = 0
        relevant_keys = test_train_split_dict[train_or_test]
        means, stds, mins, maxs = get_mean_std_min_max_from_user_list_format(user_datasets, test_train_split_dict['train'])
        unique_label = set([data_label[1] for data_label in user_datasets.values()])
        label_encoding = {rhythm : index for index, rhythm in enumerate(unique_label)}

        for patient_id, data_label in user_datasets.items():
            data = data_label[0]
            self.total_length += data.shape[0]
            if data.shape[0] > self.max_length:
                self.max_length = data.shape[0]
            if patient_id not in relevant_keys:
                continue
            # normalise data 
            normalised_data = normalisation_function(data, means, stds, mins, maxs)
            # get max length 
            label = label_encoding[data_label[1]]
            tensor_label = torch.tensor(label)
            tensor_label = tensor_label.type(torch.DoubleTensor)
            self.np_samples.append((normalised_data, tensor_label))

        self.average_length = int(self.total_length / len(user_datasets))

        for np_sample in self.np_samples:
            data, tensor_label = np_sample
            modified_data = self.crop_or_pad_data(data)
            assert modified_data.shape[0] == self.average_length
            # convert to tensors 
            tensor_data = torch.tensor(modified_data)
            tensor_data_size = tensor_data.size()
            tensor_data = torch.reshape(tensor_data, (1, tensor_data_size[0], tensor_data_size[1]))
            tensor_data = tensor_data.type(torch.DoubleTensor)
            self.samples.append((tensor_data, tensor_label))

    def crop_or_pad_data(self, data):
        data_length = data.shape[0]
        if data_length > self.average_length:
            crop_top = (data_length - self.average_length) // 2
            crop_bottom = (data_length - self.average_length) - crop_top
            modified_data = data[crop_top : (data_length - crop_bottom), :]
            
        else:
            padded_top = (self.average_length - data_length) // 2
            padded_bottom = (self.average_length - data_length) - padded_top
            modified_data = np.pad(data, ((padded_top, padded_bottom), (0,0)), 'constant')

        return modified_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ChapmanDataset(Dataset):
    def __init__(self, user_datasets, patient_to_rhythm_dict, test_train_split_dict, train_or_test, normalisation_function=normalise):
        self.samples = []
        relevant_keys = test_train_split_dict[train_or_test]
        means, stds, mins, maxs = get_mean_std_min_max_from_user_list_format(user_datasets, test_train_split_dict['train'])

        unique_rhythms_words = set(list(patient_to_rhythm_dict.values()))
        rythm_to_label_encoding = {rhythm : index for index, rhythm in enumerate(unique_rhythms_words)}
        
        for patient_id, data_label in user_datasets.items():
            if patient_id not in relevant_keys:
                continue
            data = data_label[0]
            normalised_data = normalisation_function(data, means, stds, mins, maxs)
            tensor_data = torch.tensor(normalised_data)
            tensor_data_size = tensor_data.size()
            tensor_data = torch.reshape(tensor_data, (1, tensor_data_size[0], tensor_data_size[1]))
            tensor_data = tensor_data.type(torch.DoubleTensor)
            rhythm = patient_to_rhythm_dict[patient_id]
            rhythm_label = rythm_to_label_encoding[rhythm]
            tensor_rhythm_label = torch.tensor(rhythm_label)
            tensor_rhythm_label = tensor_rhythm_label.type(torch.DoubleTensor)
            self.samples.append((tensor_data, tensor_rhythm_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def accuracy(pred: Tensor, labels: Tensor) -> float:
    print(f'pred size: {pred.size()}')
    print(f'y size: {labels.size()}')
    return (pred.argmax(dim=-1) == labels).float().mean().item()

def accuracy_from_val_loader_and_model(val_loader, model) -> float:
    print('in this funciton')
    acc = sum([accuracy(model(x.float()), y.float()) for x, y in val_loader]) / len(val_loader)
    return acc

def covert_dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    dataset_array = next(iter(loader))[0].numpy()
    return dataset_array

class Net(torch.nn.Module):
    def __init__(self, input_dim_x, input_dim_y=5, output_dim=1, testing_flag=False, dataset_name='mesa', disease='sleep_apnea'):
        super(Net, self).__init__()
        self.disease = disease
        
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(0.5),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, 1, input_dim_x, input_dim_y)).shape))
        
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=num_features_before_fcnn, out_features=128),
        #     torch.nn.Linear(in_features=128, out_features=output_dim),
        #     torch.nn.Softmax()
        # )

        self.fc1 = torch.nn.Linear(in_features=num_features_before_fcnn, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=output_dim)
        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        x=x.float()
        batch_size = x.size(0)
        out = self.feature_extractor(x)
        out = out.view(batch_size, -1) # flatten the vector 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out.float())
        # out = self.classifier(out)
        return out

def fit_model(model, train_loader, test_loader, lr, max_epochs=5):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()


    def threshold_output_transform(output):
        y_pred, y = output 
        y_pred = torch.heaviside(y_pred, values=torch.zeros(1))
        # print(f'y_pred size : {y_pred.size()}')
        # print(f'y size : {y.size()}')
        return y_pred, y

    def prepare_batch(batch, device, non_blocking):
        x,y = batch
        x = x.float()
        y = y.float()
        y = torch.unsqueeze(y, 1)
        return (x,y)
    
    def squeeze_y_dims(output):
        prediction, target = output
        # print(f'prediction size: {prediction.size()}')
        # print(f'target size: {target.size()}')
        return prediction, target

    trainer = create_supervised_trainer(model, optimizer, criterion, prepare_batch=prepare_batch)

    val_metrics = {
        "accuracy": Accuracy(threshold_output_transform),
        "bce": Loss(criterion, output_transform=squeeze_y_dims)
        # "precision" : Precision(threshold_output_transform, average=False),
        # "recall": Recall(threshold_output_transform, average=False)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, prepare_batch=prepare_batch)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f}")

    # @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        # print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f} Avg precision : {metrics['precision']:.2f} Avg recall: {metrics['recall']:.2f}")
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_validation_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        # print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f} Avg precision : {metrics['precision']:.2f} Avg recall: {metrics['recall']:.2f}")
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f}")

    trainer.run(train_loader, max_epochs=max_epochs)

    return model

def fit_model_multiclass(model, train_loader, test_loader, lr, max_epochs=5, number_of_classes=2):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def threshold_output_transform(output):
        y_pred, y = output 
        print('got in here threshold ')
        y_pred = torch.heaviside(y_pred, values=torch.zeros(1))
        # print(f'y_pred size : {y_pred.size()}')
        # print(f'y size : {y.size()}')
        return y_pred, y

    def prepare_batch(batch, device, non_blocking):
        x,y = batch
        # print(f'x type: {x.dtype}')
        # print(f'y type: {y.dtype}')
        x = x.to(dtype=torch.long)
        y = y.to(dtype=torch.long)
        # print(f'x type: {x.dtype}')
        # print(f'y type: {y.dtype}')
        # y = torch.unsqueeze(y, 1)
        y = y.squeeze()
        return (x,y)
    
    def squeeze_y_dims(output):
        prediction, target = output
        print('got in here squeeze y dims')
        # print(f'prediction size: {prediction.size()}')
        # print(f'target size: {target.size()}')
        return prediction.long(), target.squeeze().long()

    def correct_shape(output):
        y_pred, y = output
        print('got in here squeeze correct shape')
        one_hot_y = torch.nn.functional.one_hot(y, num_classes=number_of_classes)
        one_hot_y = one_hot_y.squeeze(1)
        
        argmax = y_pred.argmax(1)
        m = torch.zeros(y_pred.shape).scatter(1, argmax.unsqueeze(1), 1.0)

        return m, one_hot_y
    
    def trainer_output_shape(output):
        print('got here output transform trainer ')
        x, y, y_pred, loss = output

    trainer = create_supervised_trainer(model, optimizer, criterion, prepare_batch=prepare_batch)

    val_metrics = {
        "accuracy": Accuracy(output_transform=correct_shape, is_multilabel=True),
        "loss": Loss(criterion, output_transform=squeeze_y_dims)
        # "precision" : Precision(threshold_output_transform, average=False),
        # "recall": Recall(threshold_output_transform, average=False)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, prepare_batch=prepare_batch)

    # @trainer.on(Events.ITERATION_COMPLETED(every=10))
    # def log_training_loss(trainer):
    #     evaluator.run(train_loader)
    #     metrics = evaluator.state.metrics
    #     print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    # @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        # print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f} Avg precision : {metrics['precision']:.2f} Avg recall: {metrics['recall']:.2f}")
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_validation_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        # print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['bce']:.2f} Avg precision : {metrics['precision']:.2f} Avg recall: {metrics['recall']:.2f}")
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

    trainer.run(train_loader, max_epochs=max_epochs)

    return model







