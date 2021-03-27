import pickle5 as pickle
# import pickle 
import scipy.constants
import datetime
import tensorflow as tf
import numpy as np
import tqdm
import os
import pandas as pd
import ast
import sys
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

import chapman_data_pre_processing
import chapman_transformations
import simclr_chapman_predictions
import simclr_models
import simclr_utitlities

def parse_arguments(args):

    testing_flag = args[1] == 'True'
    window_size = int(args[2])
    batch_size = int(args[3])
    non_testing_simclr_epochs = int(args[4])
    transformation_indices = ast.literal_eval(args[5])
    initial_learning_rate = float(args[6])
    non_testing_linear_eval_epochs = int(args[7])
    plot_flag = args[8] == 'True'
    predict_flag = args[9] == 'True'
    aggregate = args[10]

    return testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate


def main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate):

    if testing_flag:
        working_directory = 'chapman_testing/'
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
    else:
        working_directory = 'chapman/'
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)


    transformation_indicies_string = ''.join([str(i) for i in transformation_indices])
    lr_string = str(initial_learning_rate).replace('.','')
    name_of_run = f'testing-{testing_flag}-ws-{window_size}-bs-{batch_size}-transformations-{transformation_indicies_string}-lr-{lr_string}-agg-{aggregate}'
    print(name_of_run)
    
    dataset_save_path = os.path.join(os.getcwd(), "PickledData", "chapman")
    path_to_patient_to_rhythm_dict = os.path.join(dataset_save_path, 'patient_to_rhythm_dict.pickle')

    if testing_flag:
        path_to_user_datasets = os.path.join(dataset_save_path, '100_users_datasets.pickle')
        path_to_test_train_split_dict = os.path.join(dataset_save_path, '100_users_reduced_test_train_split_dict.pickle')
    else:
        path_to_user_datasets  = os.path.join(dataset_save_path, "four_lead_user_datasets.pickle")
        path_to_test_train_split_dict = os.path.join(dataset_save_path, "test_train_split_dict.pickle")

    with open(path_to_user_datasets, 'rb') as f:
        user_datasets = pickle.load(f)

    # Parameters
    number_of_leads = 4
    input_shape = (window_size, number_of_leads)

    # Dataset Metadata 
    transformation_multiple = 1
    dataset_name = 'chapman.pkl'
    dataset_name_user_split = 'chapman_user_split.pkl'

    model_save_name = f"chapman_acc"

    unit_conversion = scipy.constants.g

    # a fixed user-split

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_user_dict = pickle.load(f)

    test_users = test_train_user_dict['test']
    train_users = test_train_user_dict['train']

    print(f'Test Numbers: {len(test_users)}, Train Numbers: {len(train_users)}')

    np_train, np_val, np_test = chapman_data_pre_processing.pre_process_dataset_composite(
        user_datasets=user_datasets, 
        train_users=train_users, 
        test_users=test_users, 
        window_size=window_size, 
        shift=window_size//2, 
        normalise_dataset=True, 
        verbose=1
    )

    decay_steps = 1000
    if testing_flag:
        epochs = 3
    else:
        epochs = non_testing_simclr_epochs

    temperature = 0.1
    

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

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    print(start_time_str)

    tf.keras.backend.set_floatx('float32')

    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    transformation_function = simclr_utitlities.generate_combined_transform_function(transform_funcs_vectorised, indices=transformation_indices)

    base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
    simclr_model = simclr_models.attach_simclr_head(base_model)
    simclr_model.summary()

    print("end of simclr bit")

    trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train, optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)
    simclr_model_save_path = os.path.join(working_directory, f'{start_time_str}_{name_of_run}_simclr.hdf5')
    trained_simclr_model.save(simclr_model_save_path)

    print("got here 2")

    if predict_flag:
        print("starting classifier predictions")
        #### testing predictions #### 
        # lr classifier
        logistic_regression_clf = LogisticRegression(max_iter=10000)

        simclr_model = tf.keras.models.load_model(simclr_model_save_path)
        trained_classifier, scaler, predictions, accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc, confusion_mat = simclr_chapman_predictions.get_prediction_and_scores_from_model_rhythm_f1_micro_f1_macro(user_datasets, path_to_patient_to_rhythm_dict, simclr_model, np_train, np_val, np_test, batch_size, train_users, test_users, window_size, logistic_regression_clf, aggregate=aggregate)

        results = [accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc, confusion_mat]

        results_directory = os.path.join(working_directory, 'predictions', f'{start_time_str}_{name_of_run}')
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        results_path = os.path.join(results_directory, 'lr_results.pickle')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        
if __name__ == '__main__':
    testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate = parse_arguments(sys.argv)
    print('args read successfully, starting main')
    main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate)

