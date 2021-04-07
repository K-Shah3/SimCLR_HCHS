import random
from typing import Callable, Tuple, Union, Dict, List
import os
from os import cpu_count
import numpy as np
from copy import deepcopy
from itertools import chain

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as func
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
# from torchvision.models import resnet18
import pytorch_lightning as pl
import tensorflow as tf


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
        scalar = torch.normal(0, self.sigma, size=(1,))
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

# byol 
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

        # self.encoder(torch.zeros(2, 3, *image_size))
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
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

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
        self.log("val_loss", val_loss.item())

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

        # self.encoder(torch.zeros(2, 3, *image_size))
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
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            half_size = x.size()[2] // 2
            split = torch.split(x, half_size, dim=2)
            x1, x2 = split[0], split[1]
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
        half_size = x.size()[2] // 2
        split = torch.split(x, half_size, dim=2)
        x1, x2 = split[0], split[1]
        x1, x2 = self.augment(x1), self.augment(x2)
        pred1, pred2 = self.forward(x1.float()), self.forward(x2.float())
        targ1, targ2 = self.target(x1.float()), self.target(x2.float())
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())

# supervised training 
class SupervisedLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, **hparams):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.float())

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        loss = func.cross_entropy(self.forward(x.long()), y.long())
        self.log("train_loss", loss.item())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        y_tensor = torch.tensor(y, dtype=torch.long)

        loss = func.cross_entropy(self.forward(x.float()), y_tensor)
        return {"loss": loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        self.log("val_loss", val_loss.item())

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
