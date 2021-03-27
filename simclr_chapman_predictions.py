import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle5 as pickle
import simclr_models
import chapman_data_pre_processing

rhythms = ['SB', 'GSVT', 'SR', 'AFIB']
mapping_from_rhythms_to_number = {rhythms[i]: i for i in range(4)}
mapping_from_number_to_rhythms = {i: rhythms[i] for i in range(4)}

def get_prediction_and_scores_from_model_rhythm_f1_micro_f1_macro(user_datasets, path_to_patient_to_rhythm_dict, model, np_train, np_val, np_test, batch_size, train_users, test_users, window_size, classifier, aggregate='mean'):
    """
    We predict the diseases that the test set have by first getting user level activations from an intermediate layer of the trained simclr model. We use these user level activations to train simple classifiers
    Parameters:
        user_datasets - all data
        patient_to_rhythm_dict - patient to rhythm mapping 
        model - trained simclr model
        np_train, np_val, np_test - windowed datasets that were used for the simclr model 
        batch_size - usually 256
        train_users, test_users - train and test user split 
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
    train_X, val_X, test_X, train_y, val_y, test_y = get_train_valid_test_X_y_disease_specific(user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_patient_to_rhythm_dict, aggregate, number_of_layers=7)
    # over hear 
    predictions, trained_classifier, scaler, test_X_scaled = classify_disease(classifier, train_X, val_X, test_X, train_y, val_y, test_y)

    accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc, confusion_mat = prediction_metric_scores_micro_macro(trained_classifier, predictions, test_y, test_X_scaled)  

    return trained_classifier, scaler, predictions, accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc, confusion_mat

def get_train_valid_test_X_y_disease_specific(user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_patient_to_rhythm_dict, aggregate, number_of_layers=7):

    train_val_test_user_level_activations, train_val_test_rhythm_number_labels = model_to_user_level_activations(user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_patient_to_rhythm_dict, aggregate=aggregate, number_of_layers=number_of_layers)

    train_X = train_val_test_user_level_activations[0].values.tolist()
    val_X = train_val_test_user_level_activations[1].values.tolist()
    test_X = train_val_test_user_level_activations[2].values.tolist()

    train_y = train_val_test_rhythm_number_labels[0]
    val_y = train_val_test_rhythm_number_labels[1]
    test_y = train_val_test_rhythm_number_labels[2]

    return train_X, val_X, test_X, train_y, val_y, test_y

def model_to_user_level_activations(user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_patient_to_rhythm_dict, aggregate='mean', number_of_layers=7):
    """
    Go from model to user level activations - representations of user learnt from the model using window pooling
    Parameters:
        user_datsets - dictionary with key of users and value of the data and labels 
        model - simclr model trained on precursor task
        np_train, np_val, np_test - windowed datasets from simclr training 
        batch_size - usually 256
        train_users - list of user ids that were used for training (and validation)
        test_users - list of user ids that were used for testing 
        window_size - usually 200
        path_to_patient_to_rhythm_dict - path to pickled patient to rhythm mapping for chapman
        aggregate - what aggregate is used for window pooling (mean, min, max, std, median)
        number_of_layers - how many layers of the model is used for embeddings prediction

    Returns:
        List of train, valid and test user_level_activations - pandas datafram
        List of train, valid and test rhythm labels (numbers form) 
    """
    
    # get window to user mappings
    train_val_test_user_window_list = chapman_data_pre_processing.get_window_to_user_mapping(user_datasets, train_users, test_users, window_size)


    # get window level embeddings 
    intermediate_model = simclr_models.extract_intermediate_model_from_base_model(model, intermediate_layer=number_of_layers)
    intermediate_model.summary()

    np_train_val_test = [np_train, np_val, np_test]

    embeddings_train_val_test_dataframes = []
    for np_dataset in np_train_val_test:
        embedding = intermediate_model.predict(np_dataset, batch_size=batch_size)
        embedding_dataframe = pd.DataFrame(embedding)
        embeddings_train_val_test_dataframes.append(embedding_dataframe)
    
    train_val_test_user_level_activations = []
    for embedding_dataframe, user_window_list in zip(embeddings_train_val_test_dataframes, train_val_test_user_window_list):
        embedding_dataframe.index = user_window_list
        if aggregate == 'mean':
            user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).mean()
        elif aggregate == 'std':
            user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).std()
        elif aggregate == 'min':
            user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).min()
        elif aggregate == 'max':
            user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).max()
        else:
            user_level_activation = embedding_dataframe.groupby(embedding_dataframe.index).median()

        train_val_test_user_level_activations.append(user_level_activation)

    train_val_test_rhythm_number_labels = get_train_val_test_rhythm_labels(path_to_patient_to_rhythm_dict, train_val_test_user_level_activations)


    return train_val_test_user_level_activations, train_val_test_rhythm_number_labels


def get_train_val_test_rhythm_labels(path_to_patient_to_rhythm_dict, train_val_test_user_level_activations):
    with open(path_to_patient_to_rhythm_dict, 'rb') as f:
        patient_to_rhythm_dict = pickle.load(f)

    train_val_test_rhythm_number_labels = []

    for user_level_activations in train_val_test_user_level_activations:
        index_user_level_activations = list(user_level_activations.index.values)
        rhythm_label_list = [patient_to_rhythm_dict[user] for user in index_user_level_activations]
        number_label_list = [mapping_from_rhythms_to_number[rhythm] for rhythm in rhythm_label_list]
        train_val_test_rhythm_number_labels.append(number_label_list)

    return train_val_test_rhythm_number_labels

def classify_disease(classifier, train_X, val_X, test_X, train_y, val_y, test_y):
    """
    Train a classifier with the user level embeddings for a particular disease. 
    We merge the train and valid datasets as we no longer need a validation dataset 

    Parameters:
        classifier - the type of classifier we wish to use to classify our data (e.g. RandomForest, SVC, SGDClassifier)
        train_X, val_X, test_X - user level embeddings 
        train_y, val_y, test_y - disease labels for each user 

    Returns:
        predictions - predictions of the classifier trained on the user embeddings 
        classifier - classifier trained on X_train data
        scaler - fit to train_X data
        test_X_scaled - return test data scaled by train data for auc score
    """
    # merge the training and validation datasets as we have no long need a validation set
    train_X = train_X + val_X
    train_y = train_y + val_y 

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    classifier.fit(train_X_scaled, train_y)
    predictions = classifier.predict(test_X_scaled)

    return predictions, classifier, scaler, test_X_scaled

def prediction_metric_scores_micro_macro(classifier, prediction, truth, X):
    accuracy = accuracy_score(truth, prediction)
    
    precision_micro = precision_score(truth, prediction, average='micro')
    recall_micro = recall_score(truth, prediction, average='micro')
    f1_micro = f1_score(truth, prediction, average='micro')

    precision_macro = precision_score(truth, prediction, average='macro')
    recall_macro = recall_score(truth, prediction, average='macro')
    f1_macro = f1_score(truth, prediction, average='macro')

    auc = roc_auc_score(truth, classifier.predict_proba(X), multi_class='ovr')
    confusion_mat = confusion_matrix(truth, prediction)

    return accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc, confusion_mat
