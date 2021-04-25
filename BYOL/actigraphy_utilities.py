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
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
import tensorflow as tf
import math 

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

class CombinedDataset(Dataset):
    '''Dataset that is able to combine both hchs and mesa datasets'''
    def __init__(self, mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, train_or_test, normalisation_function=normalise):
        self.np_samples = []
        self.samples = []
        self.max_length = - math.inf
        self.total_length = 0
        mesa_relevant_keys = mesa_test_train_split_dict[train_or_test]
        hchs_relevant_keys = hchs_test_train_split_dict[train_or_test]
        mesa_means, mesa_stds, mesa_mins, mesa_maxs = get_mean_std_min_max_from_user_list_format(mesa_user_datasets, mesa_test_train_split_dict['train']) 
        hchs_means, hchs_stds, hchs_mins, hchs_maxs = get_mean_std_min_max_from_user_list_format(hchs_user_datasets, hchs_test_train_split_dict['train'])
        mesa_unique_label = set([data_label[1] for data_label in mesa_user_datasets.values()])
        mesa_label_encoding = {rhythm : index for index, rhythm in enumerate(mesa_unique_label)}
        hchs_unique_label = set([data_label[1] for data_label in hchs_user_datasets.values()])
        hchs_label_encoding = {rhythm : index for index, rhythm in enumerate(hchs_unique_label)}

        both_user_datasets = [mesa_user_datasets, hchs_user_datasets]
        both_averages = [[mesa_means, mesa_stds, mesa_mins, mesa_maxs], [hchs_means, hchs_stds, hchs_mins, hchs_maxs]]
        both_label_encodings = [mesa_label_encoding, hchs_label_encoding]
        both_relevant_keys = [mesa_relevant_keys, hchs_relevant_keys]
        for user_datasets, averages, label_encoding, relevant_keys in zip(both_user_datasets, both_averages, both_label_encodings, both_relevant_keys):
            for patient_id, data_label in user_datasets.items():
                data = data_label[0]
                self.total_length += data.shape[0]
                if patient_id not in relevant_keys:
                    continue
                
                normalised_data = normalisation_function(data, *averages)
                label = label_encoding[data_label[1]]
                tensor_label = torch.tensor(label)
                tensor_label = tensor_label.type(torch.DoubleTensor)
                self.np_samples.append((normalised_data, tensor_label))

        datasets_length = len(mesa_user_datasets) + len(hchs_user_datasets)
        self.average_length = int(self.total_length / datasets_length)

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

class SpecificLengthDataset(Dataset):
    def __init__(self, user_datasets, test_train_split_dict, train_or_test, length, normalisation_function=normalise):
        self.samples = []
        self.average_length = length 
        relevant_keys = test_train_split_dict[train_or_test]
        means, stds, mins, maxs = get_mean_std_min_max_from_user_list_format(user_datasets, test_train_split_dict['train'])
        unique_label = set([data_label[1] for data_label in user_datasets.values()])
        label_encoding = {rhythm : index for index, rhythm in enumerate(unique_label)}

        for patient_id, data_label in user_datasets.items():
            data = data_label[0]
            label = label_encoding[data_label[1]]
            if patient_id not in relevant_keys:
                continue 
            #normalise 
            normalised_data = normalisation_function(data, means, stds, mins, maxs)
            # crop or pad to shape 
            modified_data = self.crop_or_pad_data(normalised_data)
            assert modified_data.shape[0] == self.average_length

            # convert to tensors 
            tensor_data = torch.tensor(modified_data)
            tensor_data_size = tensor_data.size()
            tensor_data = torch.reshape(tensor_data, (1, tensor_data_size[0], tensor_data_size[1]))
            tensor_data = tensor_data.type(torch.DoubleTensor)
            tensor_label = torch.tensor(label)
            tensor_label = tensor_label.type(torch.DoubleTensor)
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

class autoencoder(nn.Module):
    def __init__(self, latent_dim, x_dim, y_dim=5):
        super(autoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(x_dim*y_dim, latent_dim),
        #     nn.Sigmoid())
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, x_dim*y_dim),
        #     nn.Sigmoid())

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x_dim*y_dim, 2046),
            nn.ReLU(),
            nn.Linear(2046, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, x_dim*y_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_trained_autoencoder(user_datasets, test_train_split_dict, batch_size=128, latent_dim=256):
    '''With the input data train an autoencoder on the train data only'''
    train_user_list = test_train_split_dict['train']
    test_user_list = test_train_split_dict['test']

    train_dataset = ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset)
    )

    model = autoencoder(latent_dim, x_dim=train_dataset.average_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        for data_label in train_loader:
            data, _ = data_label 
            input_data = Variable(data)
            loss_data = data.view(data.size(0), -1)
            output = model(input_data.float())
            loss = criterion(output, loss_data.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # get validation 
    encoder = model.encoder
    decoder = model.decoder 

    for data_label in val_loader:
        data, _ = data_label 
        input_data = Variable(data)
        loss_data = data.view(data.size(0), -1)
        encoded_data = encoder(input_data.float())
        decoded_data = decoder(encoded_data.float())
        loss = criterion(decoded_data.float(), loss_data.float())
        print(f'val_loss: {loss.item()}')

    return autoencoder, encoder, decoder

def get_trained_autoencoder_ms(user_datasets, test_train_split_dict, batch_size=128, latent_dim=256, segment_number=2):
    train_user_list = test_train_split_dict['train']
    test_user_list = test_train_split_dict['test']

    train_dataset = ActigraphyDataset(user_datasets, test_train_split_dict, 'train')
    test_dataset = ActigraphyDataset(user_datasets, test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset)
    )
    segment_length = train_dataset.average_length // segment_number
    model = autoencoder(latent_dim, x_dim=segment_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    for epoch in range(num_epochs):
        for data_label in train_loader:
            data, _ = data_label 
            split_data = torch.split(data, segment_length, dim=2)
            for i in range(segment_number):
                d = split_data[i]
                input_data = Variable(d)
                loss_data = d.view(d.size(0), -1)
                output = model(input_data.float())
                loss = criterion(output, loss_data.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # get validation
    encoder = model.encoder 
    decoder = model.decoder

    for data_label in val_loader:
        data, _ = data_label 
        split_data = torch.split(data, segment_length, dim=2)

        for i in range(segment_number):
            d = split_data[i]
            input_data = Variable(d)
            loss_data = d.view(d.size(0), -1)
            encoded_data = encoder(input_data.float())
            decoded_data = decoder(encoded_data.float())
            loss = criterion(decoded_data.float(), loss_data.float())
            print(f'val_loss: {loss.item()}')

    return autoencoder, encoder, decoder

def get_combined_trained_autoencoder(mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, batch_size=128, latent_dim=256):
    '''With the input data train an autoencoder on the train data only'''
    mesa_train_user_list = mesa_test_train_split_dict['train']
    mesa_test_user_list = mesa_test_train_split_dict['test']
    hchs_train_user_list = hchs_test_train_split_dict['train']
    hchs_test_user_list = hchs_test_train_split_dict['test']

    train_dataset = CombinedDataset(mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, 'train')
    test_dataset = CombinedDataset(mesa_user_datasets, mesa_test_train_split_dict, hchs_user_datasets, hchs_test_train_split_dict, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset)
    )

    model = autoencoder(latent_dim, x_dim=train_dataset.average_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        for data_label in train_loader:
            data, _ = data_label 
            input_data = Variable(data)
            loss_data = data.view(data.size(0), -1)
            output = model(input_data.float())
            loss = criterion(output, loss_data.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # get validation 
    encoder = model.encoder
    decoder = model.decoder 

    for data_label in val_loader:
        data, _ = data_label 
        input_data = Variable(data)
        loss_data = data.view(data.size(0), -1)
        encoded_data = encoder(input_data.float())
        decoded_data = decoder(encoded_data.float())
        loss = criterion(decoded_data.float(), loss_data.float())
        print(f'val_loss: {loss.item()}')

    return autoencoder, encoder, decoder

## data augmentations
class GaussianNoiseTransform(nn.Module):
    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        size = x.size()
        noise = torch.normal(0, self.sigma, size)
        return x + noise

class ScaleTransform(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        scalar = torch.normal(1, self.sigma, size=(1,))
        return scalar * x

class Negate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return -1 * x

def default_augmentation(data_size: Tuple[int, int] = (2500, 4)) -> nn.Module:
    return nn.Sequential(
        GaussianNoiseTransform(sigma=0.05),
        ScaleTransform(sigma=0.1),
        Negate(),
    )

# encoder wrapper 
def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )

class EncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0)
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            self._projector = mlp(
                self._projector_dim, self.projection_size, self.hidden_size
            )
        return self._projector

    def _hook(self, _, __, output):
        output = output.flatten(start_dim=1)
        if self._projector_dim is None:
            self._projector_dim = output.shape[-1]
        self._encoded = self.projector(output)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self._encoded    

def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = func.normalize(x, dim=-1)
    y = func.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)

class BYOL(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (128, 128),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.999,
        batch_size: int = 512,
        **hparams
    ):
        super().__init__()
        self.augment = default_augmentation(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.batch_size = batch_size
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self.hparams = hparams
        self._target = None

        self.encoder(torch.zeros(2, 1, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        # optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        # lr = self.hparams.get("lr", 1e-4)
        # weight_decay = self.hparams.get("weight_decay", 1e-6)
        # return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        optimizer = LARS(self.parameters(), 
                        lr=(0.2*(self.batch_size/256)),
                        momentum=0.9,
                        weight_decay=self.hparams.get("weight_decay", 1e-6),
                        trust_coefficient=0.001)

        train_iters_per_epoch = 8000 // self.batch_size
        warmup_steps = train_iters_per_epoch * 10
        total_steps = train_iters_per_epoch * 100

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
           
        pred1, pred2 = self.forward(x1.float()), self.forward(x2.float())
        with torch.no_grad():
            targ1, targ2 = self.target(x1.float()), self.target(x2.float())
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("train_loss", loss.item())
        self.update_target()

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.forward(x1.float()), self.forward(x2.float())
        targ1, targ2 = self.target(x1.float()), self.target(x2.float())
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        # self.log("val_loss", val_loss.item())    


class BYOL_MS(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (128, 128),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.999,
        segment_number = 2,
        batch_size = 512, 
        **hparams,
    ):
        super().__init__()
        self.augment = default_augmentation(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = nn.Linear(projection_size, projection_size, hidden_size)
        self.hparams = hparams
        self._target = None
        self.segment_number = segment_number
        self.encoder(torch.zeros(2, 1, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    # --- Methods required for PyTorch Lightning only! ---

    def configure_optimizers(self):
        # optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        # lr = self.hparams.get("lr", 1e-4)
        # weight_decay = self.hparams.get("weight_decay", 1e-6)
        # return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        optimizer = LARS(self.parameters(), 
                        lr=(0.2*(self.batch_size/256)),
                        momentum=0.9,
                        weight_decay=self.hparams.get("weight_decay", 1e-6),
                        trust_coefficient=0.001)

        train_iters_per_epoch = 8000 // self.batch_size
        warmup_steps = train_iters_per_epoch * 10
        total_steps = train_iters_per_epoch * 100

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            segment_size = x.size()[2] // self.segment_number
            split = torch.split(x, segment_size, dim=2)
            split = split[:self.segment_number]
        
        # get all pairs of segments using itertools combinations
        for x1, x2 in combinations(split, 2):
            x1, x2 = self.augment(x1), self.augment(x2)
            pred1, pred2 = self.forward(x1.float()), self.forward(x2.float())
            with torch.no_grad():
                targ1, targ2 = self.target(x1.float()), self.target(x2.float())
            loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

            self.log("train_loss", loss.item())
            self.update_target()

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        segment_size = x.size()[2] // self.segment_number
        split = torch.split(x, segment_size, dim=2)
        split = split[:self.segment_number]

        for x1, x2 in combinations(split, 2):
            x1, x2 = self.augment(x1), self.augment(x2)
            pred1, pred2 = self.forward(x1.float()), self.forward(x2.float())
            targ1, targ2 = self.target(x1.float()), self.target(x2.float())
            loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        # self.log("val_loss", val_loss.item())


        


    