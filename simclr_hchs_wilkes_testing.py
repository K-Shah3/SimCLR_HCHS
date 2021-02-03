import pickle as pickle 
import scipy.constants
import datetime
import tensorflow as tf
import numpy as np
import tqdm
import os
import pandas as pd
# Libraries for plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
sns.set_context('poster')
# Classifiers
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import hchs_data_pre_processing
import hchs_transformations
import simclr_models
import simclr_utitlities
import simclr_predictions

# hchs file paths 
working_directory = 'test_run_hchs/'
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

## FOR TESTING ON A REDUCED DATASET OF 100 USERS ONLY 

dataset_save_path = os.path.join(os.getcwd(), "PickledData", "hchs")
# use for training on smaller number 
path_to_user_datasets = os.path.join(dataset_save_path, '100_users_dataset.pickle')
path_to_test_train_split_dict = os.path.join(dataset_save_path, '100_users_reduced_test_train_users.pickle')

path_to_baseline_sueno_merge_no_na = os.path.join(dataset_save_path, "baseline_sueno_merge_no_na.pickle")
disease_labels = {'diabetes': {1: 'Non-diabetic', 2: 'Pre-diabetic', 3: 'Diabetic'}, 'sleep_apnea': {0: 'No', 1: 'Yes'}, 'hypertension': {0: 'No', 1: 'Yes'}, 'metabolic_syndrome': {0: 'No', 1: 'Yes'}, 'insomnia': {1: 'No clinically significant insomnia', 2: 'Subthreshold insomnia', 3: 'Clinical insomnia'}}

testing_file_path = os.path.join(os.getcwd(), "testing", "plots")

with open(path_to_user_datasets, 'rb') as f:
    user_datasets = pickle.load(f)

# Parameters
window_size = 500
input_shape = (window_size, 5)

# Dataset Metadata 
transformation_multiple = 1
dataset_name = 'hchs.pkl'
dataset_name_user_split = 'hchs_user_split.pkl'

label_list = ["ACTIVE", "REST", "REST-S"]
label_list_full_name = label_list
has_null_class = False

label_map = dict([(l, i) for i, l in enumerate(label_list)])

output_shape = len(label_list)

model_save_name = f"hchs_acc"

sampling_rate = 50.0
unit_conversion = scipy.constants.g

with open(path_to_test_train_split_dict, 'rb') as f:
    test_train_user_dict = pickle.load(f)

test_users = test_train_user_dict['test']
train_users = test_train_user_dict['train']

print(f'Test Numbers: {len(test_users)}, Train Numbers: {len(train_users)}')

# np_train, np_val, np_test = hchs_data_pre_processing.pre_process_dataset_composite(
#     user_datasets=user_datasets, 
#     label_map=label_map, 
#     output_shape=output_shape, 
#     train_users=train_users, 
#     test_users=test_users, 
#     window_size=window_size, 
#     shift=window_size//2, 
#     normalise_dataset=True, 
#     verbose=1
# )


# mesa file path 
mesa_dataset_save_path = os.path.join(os.getcwd(), "PickledData", "mesa")
mesa_user_datasets_path = os.path.join(mesa_dataset_save_path, '100_users_dataset.pickle')
mesa_path_to_test_train_split_dict = os.path.join(mesa_dataset_save_path, '100_users_reduced_test_train_users.pickle')

mesa_sample_key = 3950

with open(mesa_user_datasets_path, 'rb') as f:
    mesa_user_datasets = pickle.load(f)

with open(mesa_path_to_test_train_split_dict, 'rb') as f:
    mesa_test_train_user_dict = pickle.load(f)

mesa_test_users = mesa_test_train_user_dict['test']
mesa_train_users = mesa_test_train_user_dict['train']

print(f'MESA Test Numbers: {len(mesa_test_users)}, MESA Train Numbers: {len(mesa_train_users)}')

# mesa_np_train, mesa_np_val, mesa_np_test = hchs_data_pre_processing.pre_process_dataset_composite(
#     user_datasets=mesa_user_datasets, 
#     label_map=label_map, 
#     output_shape=output_shape, 
#     train_users=mesa_train_users, 
#     test_users=mesa_test_users, 
#     window_size=window_size, 
#     shift=window_size//2, 
#     normalise_dataset=True, 
#     verbose=1
# )

# mesa_train_val_test_user_window_list = hchs_data_pre_processing.get_window_to_user_mapping(mesa_user_datasets, mesa_train_users, mesa_test_users, window_size)
# hchs_train_val_test_user_window_list = hchs_data_pre_processing.get_window_to_user_mapping(user_datasets, train_users, test_users, window_size)

def testing_plots(x):

    X = [i for i in range(x, x+10)]
    y = [2 * i for i in X]
    plt.figure(figsize=(12,8))
    plt.scatter(X, y)
    plt.legend()

    save_file = os.path.join(testing_file_path, f'{x}_with_figsize_picture.png')
    plt.savefig(save_file)



if __name__ == '__main__':
    # print(f'hchs np train shape:{np_train[0].shape}')
    # print(f'mesa np train shape:{mesa_np_train[0].shape}')
    # print(f'mesa user window list train shape: {len(mesa_train_val_test_user_window_list[0])}')
    # print(f'hchs user window list train shape: {len(hchs_train_val_test_user_window_list[0])}')
    for i in range(3):
        testing_plots(i)

