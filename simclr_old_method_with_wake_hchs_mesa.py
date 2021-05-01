# import pickle5 as pickle 
import pickle
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
sns.set_context('poster')
# Classifiers
from sklearn.linear_model import LogisticRegression
import hchs_data_pre_processing
import hchs_transformations
import simclr_models
import simclr_utitlities
import simclr_predictions
# used for only data windowing
import chapman_data_pre_processing
import hchs_mesa_data_pre_processing
from lars_optimizer import LARSOptimizer

dataset_diseases = {'hchs': ['hypertension', 'diabetes', 'sleep_apnea', 'metabolic_syndrome', 'insomnia'],
            'mesa': ['sleep_apnea', 'insomnia']}

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
    temperature = ast.literal_eval(args[11])
    hchs_or_mesa = args[12]
    disease = args[13]

    return testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature, hchs_or_mesa, disease

def get_directories_and_name(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    if testing_flag:
        working_directory = f'simclr_{hchs_or_mesa}_testing/'
    else:
        working_directory = f'simclr_{hchs_or_mesa}'
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    
    transformation_indicies_string = ''.join([str(i) for i in transformation_indices])
    lr_string = str(initial_learning_rate).replace('.','')
    name_of_run = f'testing-{testing_flag}-ws-{window_size}-bs-{batch_size}-transformations-{transformation_indicies_string}-lr-{lr_string}-agg-{aggregate}-temp-{temperature}-{hchs_or_mesa}-{disease}'
    print(f'name of run: {name_of_run}')

    dataset_save_path = os.path.join(os.getcwd(), "PickledData", hchs_or_mesa)
    path_to_embeddings = os.path.join(os.getcwd(), "embeddings", hchs_or_mesa, disease, "simclr")
    if not os.path.exists(path_to_embeddings):
        os.makedirs(path_to_embeddings)

    path_to_baseline_sueno_merge_no_na = os.path.join(dataset_save_path, "baseline_sueno_merge_no_na.pickle") 
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
        disease_user_dataset_path = os.path.join(dataset_save_path, f'{disease}_with_wake_user_datasets.pickle')
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

    return working_directory, name_of_run, test_train_split_dict, disease_user_datasets, path_to_baseline_sueno_merge_no_na, path_to_embeddings

def train_simclr(np_train, testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature):
    decay_steps = 1000
    
    if testing_flag:
        epochs = 1
    else:
        epochs = non_testing_simclr_epochs

    input_shape = (window_size, 6)

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

    # lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
    # optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    
    learning_rate = 0.3 * (batch_size / 256)
    optimizer = LARSOptimizer(learning_rate, weight_decay=0.000001)
    transformation_function = simclr_utitlities.generate_combined_transform_function(transform_funcs_vectorised, indices=transformation_indices)

    base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
    simclr_model = simclr_models.attach_simclr_head(base_model)
    # simclr_model.summary()
    trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train, optimizer, batch_size, 
                                            transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    return trained_simclr_model

def get_user_level_activation(embedding_dataframe, aggregate):
    '''Take the output of the SimCLR model, and group together the same user according to the average 
    specified to get the user level activations'''
    if aggregate == 'median':
        user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).median()
    elif aggregate == 'std':
        user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).std()
    elif aggregate == 'min':
        user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).min()
    elif aggregate == 'max':
        user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).max()
    else:
        # default is mean activation 
        user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).mean()

    return user_level_activation

def model_to_user_level_embeddings(model, np_train, np_test, batch_size, window_size, aggregate, train_user_window_list, test_user_window_list, number_of_layers=7):
    '''From the trained SimCLR Model extract the representations for the train and test datasets. Group all users together using the given aggregate
    Return the group embeddings, and the corresponding patient ids to each embedding.'''

    # get intermediate model 
    intermediate_model = simclr_models.extract_intermediate_model_from_base_model(model, intermediate_layer=number_of_layers)
    np_train_test = [np_train, np_test]
    user_window_lists_train_test = [train_user_window_list, test_user_window_list]
    user_level_activations_train_test = []
    user_id_lists_train_test = []
    for np_dataset, user_window_list in zip(np_train_test, user_window_lists_train_test):
        embedding = intermediate_model.predict(np_dataset, batch_size=batch_size)
        embedding_dataframe = pd.DataFrame(embedding)
        embedding_dataframe.index = user_window_list
        user_level_activation = get_user_level_activation(embedding_dataframe, aggregate)
        user_id_lists_train_test.append(list(user_level_activation.index))
        user_level_activations_train_test.append(user_level_activation)

    return user_level_activations_train_test, user_id_lists_train_test
    
def get_train_test_val_baseline_sueno(path_to_baseline_sueno_merge_no_na, user_datasets, train_users, test_users, window_size):
    """
    Split the baseline sueno dataset into train, val and test parts
    Parameters:
        path_to_baseline_sueno_merge_no_na - path to pickled dataset
        user_datasets - dictionary with key of users and value of the data and labels
        train_users - list of user ids that were used for training (and validation)
        test_users - list of user ids that were used for testing 
        window_size - usually 500

    Returns:
        List of train, val and test baseline_sueno_merged with no null values parts 
    """
    # get window to user mappings
    train_val_test_user_window_list = hchs_data_pre_processing.get_window_to_user_mapping(user_datasets, train_users, test_users, window_size)
    with open(path_to_baseline_sueno_merge_no_na, 'rb') as f:
        baseline_sueno_merged_no_na = pickle.load(f)
    # split the baseline sueno information into train, val and test sets 
    train_val_test_baseline_sueno_merged_no_na = []
    for user_window_list in train_val_test_user_window_list:
        mask = baseline_sueno_merged_no_na.index.isin(set(user_window_list))
        baseline_sueno_merged_sectioned = baseline_sueno_merged_no_na[mask]
        train_val_test_baseline_sueno_merged_no_na.append(baseline_sueno_merged_sectioned)

    return train_val_test_baseline_sueno_merged_no_na

def get_labels_from_user_id_lists(user_id_lists_train_test, user_datasets):
    '''From a list of users that belong to the train and test datasets, get the labels in the same order from user_datasets'''
    train_user_id_list = user_id_lists_train_test[0]
    test_user_id_list = user_id_lists_train_test[1]
    train_y = []
    test_y = []
    for user_id in train_user_id_list:
        label = user_datasets[user_id][1]
        train_y.append(label)
    for user_id in test_user_id_list:
        label = user_datasets[user_id][1]
        test_y.append(label)

    return train_y, test_y

def get_train_valid_test_X_y_disease_specific(user_datasets, model, np_train, np_test, batch_size, window_size, aggregate, train_user_window_list, test_user_window_list, number_of_layers=7):

    user_level_activations_train_test, user_id_lists_train_test = model_to_user_level_embeddings(model, np_train, np_test, batch_size, window_size, aggregate, train_user_window_list, test_user_window_list, number_of_layers)
    
    train_X = user_level_activations_train_test[0].values.tolist()
    test_X = user_level_activations_train_test[1].values.tolist()

    train_y, test_y = get_labels_from_user_id_lists(user_id_lists_train_test, user_datasets)

    return train_X, test_X, train_y, test_y

def predict(user_datasets, model, np_train, np_test, batch_size, 
                            window_size, classifier, aggregate, train_user_window_list, test_user_window_list, path_to_embeddings, save_name):
    """
    We predict the diseases that the test set have by first getting user level activations from an intermediate layer of the trained simclr model. We use these user level activations to train simple classifiers
    Parameters:
        disease - disease to be predicted (diabetes, hypertension, sleep apnea, metabolic syndrome, insomnia)
        user_datasets - all data 
        model - trained simclr model
        np_train, np_test - windowed datasets that were used for the simclr model 
        batch_size - usually 512
        path_to_baseline_sueno_merge_no_na - path to dataset with disease information about each user 
        classifier - classifier being used to classify disease with the user level activation input 
        aggregate - metric used to group the window into users - mean, min, max, std, median
    
    Returns:
        trained_classifier - classifier trained on the input data for the particular disease
        scaler - scaler fit to X train data
        predictions - condition of each patient in the test set as predicted by the simple classifier 
        accuracy - how accurate the predictions were 
        precision_micro - precision of the predictions using micro average
        recall_micro - recall of the predictions using micro average
        f1_micro - f1 of the predictions micro mode using micro average
        precision_macro - precision of the predictions using macro average
        recall_macro - recall of the predictions using macro average
        f1_macro - f1 of the predictions micro mode using macro average
        confusion_matrix - confusion_matrix of the predictions 
    """
    
    train_X, test_X, train_y, test_y = get_train_valid_test_X_y_disease_specific(user_datasets, model, np_train, np_test, batch_size, window_size, aggregate, train_user_window_list, test_user_window_list, number_of_layers=7)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # save train and test X and y
    save_path = os.path.join(path_to_embeddings, f'{save_name}.pickle')
    print(f'save path of embeddings: {save_path}')
    with open(save_path, 'wb') as f:
        data = [train_X, test_X, train_y, test_y]
        pickle.dump(data, f)

    classifier.fit(train_X, train_y)
    prediction = classifier.predict(test_X)
    truth = test_y
    
    lower_macro, upper_macro, middle_macro, mean_macro, std_macro, lower_micro, upper_micro, middle_micro, mean_micro, std_micro = get_confidence_interval_f1_micro_macro(truth, prediction)
    f1_macro_average = (lower_macro + upper_macro) / 2
    f1_micro_average = (lower_micro + upper_micro) / 2 
    confidences = {'lower_macro': lower_macro, 'upper_macro': upper_macro, 'middle_macro': middle_macro, 'average_macro': f1_macro_average, 'mean_macro': mean_macro, 'std_macro': std_macro,
    'lower_micro': lower_micro, 'upper_micro': upper_micro, 'middle_micro': middle_micro, 'average_micro': f1_micro_average, 'mean_micro': mean_micro, 'std_micro': std_micro}

    accuracy = accuracy_score(truth, prediction)
    f1_micro = f1_score(truth, prediction, average='micro')
    f1_macro = f1_score(truth, prediction, average='macro')
    confusion_mat = confusion_matrix(truth, prediction)

    metrics = {'accuracy' : accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'confusion_mat': confusion_mat}
    percentages, f1_macro_scores, f1_micro_scores = different_percentage_training(train_X, test_X, train_y, test_y)
    percentage_data = [percentages, f1_macro_scores, f1_micro_scores]
    return metrics, confidences, percentage_data

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
    middle_micro = sorted_f1_micro_scores[int(0.5 * len(sorted_f1_micro_scores))]
    mean_micro = sorted_f1_micro_scores.mean()
    std_micro = sorted_f1_micro_scores.std()
    lower_macro = sorted_f1_macro_scores[int(0.05 * len(sorted_f1_macro_scores))]
    upper_macro = sorted_f1_macro_scores[int(0.95 * len(sorted_f1_macro_scores))]
    middle_macro = sorted_f1_macro_scores[int(0.5 * len(sorted_f1_macro_scores))]
    mean_macro = sorted_f1_macro_scores.mean()
    std_macro = sorted_f1_macro_scores.std()

    return lower_macro, upper_macro, middle_macro, mean_macro, std_macro, lower_micro, upper_micro, middle_micro, mean_micro, std_micro
    
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

def main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    np.random.seed(7)
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    string_indices = "".join([str(num) for num in transformation_indices])
    save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_transformations-{string_indices}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}'

    # get working directory and datasets from PickledFolder 
    working_directory, name_of_run, test_train_split_dict, disease_user_datasets, path_to_baseline_sueno_merge_no_na, path_to_embeddings = get_directories_and_name(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    user_datasets = disease_user_datasets[disease]
    # dataset with wake 
    input_shape = (window_size, 6)
    test_users = test_train_split_dict['test']
    train_users = test_train_split_dict['train']

    # np_train, np_val, np_test = chapman_data_pre_processing.pre_process_dataset_composite(user_datasets, train_users, 
    #                            test_train_split_dict, window_size, shift=window_size//2, normalise_dataset=True)

    np_train, np_test, train_user_window_list, test_user_window_list = hchs_mesa_data_pre_processing.pre_process_dataset_composite(user_datasets, train_users, test_users, window_size=window_size, shift=window_size//2, normalise_dataset=True)
    print(f'np train shape: {np_train.shape}')
    print(f'np test shape: {np_test.shape}')
    
    trained_simclr_model = train_simclr(np_train, testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature)

    logistic_regression_clf = LogisticRegression(max_iter=10000)

    metrics, confidences, percentage_data = predict(user_datasets, trained_simclr_model, np_train, np_test, batch_size, window_size, logistic_regression_clf, aggregate, train_user_window_list, test_user_window_list, path_to_embeddings, save_name)
    
    path_to_training_percentages = os.path.join(os.getcwd(), "training_percentages", hchs_or_mesa, disease, "simclr")
    if not os.path.exists(path_to_training_percentages):
        os.makedirs(path_to_training_percentages)
    
    training_path = os.path.join(path_to_training_percentages, f'{save_name}.pickle')
    print(f'save path of training percentages: {training_path}')
    with open(training_path, 'wb') as f:
        pickle.dump(percentage_data, f)

    print(f'{disease}')
    print(f"    f1 macro: {metrics['f1_macro']:.3f}")
    print(f"    f1 micro: {metrics['f1_micro']:.3f}")
    print(f'--------------------')

    pm_micro = confidences['upper_micro'] - confidences['average_micro']
    pm_macro = confidences['upper_macro'] - confidences['average_macro']
    print(f"    f1 micro 90% range: {confidences['average_micro']:.3f} +/- {pm_micro:.3f}")
    print(f"    f1 macro 90% range: {confidences['average_macro']:.3f} +/- {pm_macro:.3f}")
    print(f"    f1 micro mean std: {confidences['mean_micro']:.3f} +/- {confidences['std_micro']:.3f} ")
    print(f"    f1 macro mean std: {confidences['mean_macro']:.3f} +/- {confidences['std_macro']:.3f} ")

def different_transforms_main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices_list, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    for transformation_indices in transformation_indices_list:
        print(f'transformations: {transformation_indices}')
        main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
        print(f'-----------------------')

def all_double_transforms_combo_00_16(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    '''For all 49 combinations of 2 transformations, return a list of f1_macro and micro scores '''
    transformation_metrics = {}
    transformation_confidences = {}
    transformation_percentages = {}
    transformation_macro_at_50_percent = {}
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    for i in [0, 1]:
        for j in [0, 1, 2, 3, 4, 5, 6]:
            transformation_indices = [i, j]
            np.random.seed(7)
            start_time = datetime.datetime.now()
            start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
            string_indices = "".join([str(num) for num in transformation_indices])
            save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_transformations-{string_indices}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}'

            # get working directory and datasets from PickledFolder 
            working_directory, name_of_run, test_train_split_dict, disease_user_datasets, path_to_baseline_sueno_merge_no_na, path_to_embeddings = get_directories_and_name(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
            user_datasets = disease_user_datasets[disease]
            # dataset with wake 
            input_shape = (window_size, 6)
            test_users = test_train_split_dict['test']
            train_users = test_train_split_dict['train']

            np_train, np_test, train_user_window_list, test_user_window_list = hchs_mesa_data_pre_processing.pre_process_dataset_composite(user_datasets, train_users, test_users, window_size=window_size, shift=window_size//2, normalise_dataset=True)

            trained_simclr_model = train_simclr(np_train, testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature)

            logistic_regression_clf = LogisticRegression(max_iter=10000)

            metrics, confidences, percentage_data = predict(user_datasets, trained_simclr_model, np_train, np_test, batch_size, window_size, logistic_regression_clf, aggregate, train_user_window_list, test_user_window_list, path_to_embeddings, save_name)
            
            # store to dictionary 
            transformation_metrics[string_indices] = metrics 
            transformation_confidences[string_indices] = confidences
            transformation_percentages[string_indices] = percentage_data
            macro_50_percent_data = percentage_data[1][4]
            transformation_macro_at_50_percent[string_indices] = macro_50_percent_data

        save_data = [transformation_metrics, transformation_confidences, transformation_percentages, transformation_macro_at_50_percent]
        
        path_to_double_transforms = os.path.join(os.getcwd(), "embeddings", "double_transforms")
        if not os.path.exists(path_to_double_transforms):
            os.makedirs(path_to_double_transforms)
        save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}_{hchs_or_mesa}_{disease}-00-16.pickle'
        save_path = os.path.join(path_to_double_transforms, save_name)
        print(f'all double transform path: {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

def all_double_transforms_combo_20_36(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    '''For all 49 combinations of 2 transformations, return a list of f1_macro and micro scores '''
    transformation_metrics = {}
    transformation_confidences = {}
    transformation_percentages = {}
    transformation_macro_at_50_percent = {}
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    for i in [2,3]:
        for j in [0, 1, 2, 3, 4, 5, 6]:
            transformation_indices = [i, j]
            np.random.seed(7)
            start_time = datetime.datetime.now()
            start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
            string_indices = "".join([str(num) for num in transformation_indices])
            save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_transformations-{string_indices}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}'

            # get working directory and datasets from PickledFolder 
            working_directory, name_of_run, test_train_split_dict, disease_user_datasets, path_to_baseline_sueno_merge_no_na, path_to_embeddings = get_directories_and_name(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
            user_datasets = disease_user_datasets[disease]
            # dataset with wake 
            input_shape = (window_size, 6)
            test_users = test_train_split_dict['test']
            train_users = test_train_split_dict['train']

            np_train, np_test, train_user_window_list, test_user_window_list = hchs_mesa_data_pre_processing.pre_process_dataset_composite(user_datasets, train_users, test_users, window_size=window_size, shift=window_size//2, normalise_dataset=True)

            trained_simclr_model = train_simclr(np_train, testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature)

            logistic_regression_clf = LogisticRegression(max_iter=10000)

            metrics, confidences, percentage_data = predict(user_datasets, trained_simclr_model, np_train, np_test, batch_size, window_size, logistic_regression_clf, aggregate, train_user_window_list, test_user_window_list, path_to_embeddings, save_name)
            
            # store to dictionary 
            transformation_metrics[string_indices] = metrics 
            transformation_confidences[string_indices] = confidences
            transformation_percentages[string_indices] = percentage_data
            macro_50_percent_data = percentage_data[1][4]
            transformation_macro_at_50_percent[string_indices] = macro_50_percent_data

        save_data = [transformation_metrics, transformation_confidences, transformation_percentages, transformation_macro_at_50_percent]
        
        path_to_double_transforms = os.path.join(os.getcwd(), "embeddings", "double_transforms")
        if not os.path.exists(path_to_double_transforms):
            os.makedirs(path_to_double_transforms)
        save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}_{hchs_or_mesa}_{disease}-20-36.pickle'
        save_path = os.path.join(path_to_double_transforms, save_name)
        print(f'all double transform path: {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

def all_double_transforms_combo_40_56(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    '''For all 49 combinations of 2 transformations, return a list of f1_macro and micro scores '''
    transformation_metrics = {}
    transformation_confidences = {}
    transformation_percentages = {}
    transformation_macro_at_50_percent = {}
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    for i in [4, 5]:
        for j in [0, 1, 2, 3, 4, 5, 6]:
            transformation_indices = [i, j]
            np.random.seed(7)
            start_time = datetime.datetime.now()
            start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
            string_indices = "".join([str(num) for num in transformation_indices])
            save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_transformations-{string_indices}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}'

            # get working directory and datasets from PickledFolder 
            working_directory, name_of_run, test_train_split_dict, disease_user_datasets, path_to_baseline_sueno_merge_no_na, path_to_embeddings = get_directories_and_name(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
            user_datasets = disease_user_datasets[disease]
            # dataset with wake 
            input_shape = (window_size, 6)
            test_users = test_train_split_dict['test']
            train_users = test_train_split_dict['train']

            np_train, np_test, train_user_window_list, test_user_window_list = hchs_mesa_data_pre_processing.pre_process_dataset_composite(user_datasets, train_users, test_users, window_size=window_size, shift=window_size//2, normalise_dataset=True)

            trained_simclr_model = train_simclr(np_train, testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature)

            logistic_regression_clf = LogisticRegression(max_iter=10000)

            metrics, confidences, percentage_data = predict(user_datasets, trained_simclr_model, np_train, np_test, batch_size, window_size, logistic_regression_clf, aggregate, train_user_window_list, test_user_window_list, path_to_embeddings, save_name)
            
            # store to dictionary 
            transformation_metrics[string_indices] = metrics 
            transformation_confidences[string_indices] = confidences
            transformation_percentages[string_indices] = percentage_data
            macro_50_percent_data = percentage_data[1][4]
            transformation_macro_at_50_percent[string_indices] = macro_50_percent_data

        save_data = [transformation_metrics, transformation_confidences, transformation_percentages, transformation_macro_at_50_percent]
        
        path_to_double_transforms = os.path.join(os.getcwd(), "embeddings", "double_transforms")
        if not os.path.exists(path_to_double_transforms):
            os.makedirs(path_to_double_transforms)
        save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}_{hchs_or_mesa}_{disease}-40-56.pickle'
        save_path = os.path.join(path_to_double_transforms, save_name)
        print(f'all double transform path: {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

def all_double_transforms_combo_60_66(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease):
    '''For all 49 combinations of 2 transformations, return a list of f1_macro and micro scores '''
    transformation_metrics = {}
    transformation_confidences = {}
    transformation_percentages = {}
    transformation_macro_at_50_percent = {}
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    for i in [6]:
        for j in [0, 1, 2, 3, 4, 5, 6]:
            transformation_indices = [i, j]
            np.random.seed(7)
            start_time = datetime.datetime.now()
            start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
            string_indices = "".join([str(num) for num in transformation_indices])
            save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_transformations-{string_indices}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}'

            # get working directory and datasets from PickledFolder 
            working_directory, name_of_run, test_train_split_dict, disease_user_datasets, path_to_baseline_sueno_merge_no_na, path_to_embeddings = get_directories_and_name(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
            user_datasets = disease_user_datasets[disease]
            # dataset with wake 
            input_shape = (window_size, 6)
            test_users = test_train_split_dict['test']
            train_users = test_train_split_dict['train']

            np_train, np_test, train_user_window_list, test_user_window_list = hchs_mesa_data_pre_processing.pre_process_dataset_composite(user_datasets, train_users, test_users, window_size=window_size, shift=window_size//2, normalise_dataset=True)

            trained_simclr_model = train_simclr(np_train, testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature)

            logistic_regression_clf = LogisticRegression(max_iter=10000)

            metrics, confidences, percentage_data = predict(user_datasets, trained_simclr_model, np_train, np_test, batch_size, window_size, logistic_regression_clf, aggregate, train_user_window_list, test_user_window_list, path_to_embeddings, save_name)
            
            # store to dictionary 
            transformation_metrics[string_indices] = metrics 
            transformation_confidences[string_indices] = confidences
            transformation_percentages[string_indices] = percentage_data
            macro_50_percent_data = percentage_data[1][4]
            transformation_macro_at_50_percent[string_indices] = macro_50_percent_data

        save_data = [transformation_metrics, transformation_confidences, transformation_percentages, transformation_macro_at_50_percent]
        
        path_to_double_transforms = os.path.join(os.getcwd(), "embeddings", "double_transforms")
        if not os.path.exists(path_to_double_transforms):
            os.makedirs(path_to_double_transforms)
        save_name = f'{start_time_str}_testing-{testing_flag}_window-{window_size}_bs-{batch_size}_lr-{initial_learning_rate}_agg-{aggregate}_temp-{temperature}_{hchs_or_mesa}_{disease}-60-66.pickle'
        save_path = os.path.join(path_to_double_transforms, save_name)
        print(f'all double transform path: {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)



        



if __name__ == '__main__':
    testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature, hchs_or_mesa, disease = parse_arguments(sys.argv)
    # testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature, hchs_or_mesa, disease = True, 512, 512, 3, [2,4,3,6,1], 0.1, 2, False, True, 'mean', 0.1, 'mesa', 'insomnia'
    print('args read successfully, starting main')
    main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    # different_transforms_main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    all_double_transforms_combo_00_16(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    all_double_transforms_combo_20_36(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    all_double_transforms_combo_40_56(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    all_double_transforms_combo_60_66(testing_flag, window_size, batch_size, non_testing_simclr_epochs, non_testing_linear_eval_epochs, aggregate, temperature, hchs_or_mesa, disease)
    print('done')








