import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle

import simclr_models
import hchs_data_pre_processing

disease_labels = {'diabetes': {1: 'Non-diabetic', 2: 'Pre-diabetic', 3: 'Diabetic'}, 'sleep_apnea': {0: 'No', 1: 'Yes'}, 'hypertension': {0: 'No', 1: 'Yes'}, 'metabolic_syndrome': {0: 'No', 1: 'Yes'}, 'insomnia': {1: 'No clinically significant insomnia', 2: 'Subthreshold insomnia', 3: 'Clinical insomnia'}}


def model_to_user_level_activations(user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_baseline_sueno_merge_no_na, aggregate='mean', number_of_layers=7):
    """
    Go from model to user level activations - representations of user learnt from the model using window pooling
    Parameters:
        user_datsets - dictionary with key of users and value of the data and labels 
        model - simclr model trained on precurson task
        np_train, np_val, np_test - windowed datasets from simclr training 
        batch_size - usually 200
        train_users - list of user ids that were used for training (and validation)
        test_users - list of user ids that were used for testing 
        window_size - usually 500
        path_to_baseline_sueno_merge_no_na - path to pickled baseline_sueno_merged data for hchs
        aggregate - what aggregate is used for window pooling (mean, min, max, std, median)
        number_of_layers - how many layers of the model is used for embeddings prediction

    Returns:
        List of train, valid and test user_level_activations 
    """
    
    # get window to user mappings
    train_val_test_user_window_list = hchs_data_pre_processing.get_window_to_user_mapping(user_datasets, train_users, test_users, window_size)


    # get window level embeddings 
    intermediate_model = simclr_models.extract_intermediate_model_from_base_model(model, intermediate_layer=number_of_layers)
    intermediate_model.summary()

    np_train_val_test = [np_train, np_val, np_test]

    embeddings_train_val_test_dataframes = []
    for np_dataset in np_train_val_test:
        embedding = intermediate_model.predict(np_dataset[0], batch_size=batch_size)
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

    train_val_test_baseline_sueno_merged_no_na = get_train_test_val_baseline_sueno(path_to_baseline_sueno_merge_no_na, user_datasets, train_users, test_users, window_size)

    # make sure there are no users in the user_level_activations that do not exist in the baseline_sueno_datasets 
    train_val_test_user_level_activations_filtered = []
    for user_level_activations, baseline_sueno_merged_dataset in zip(train_val_test_user_level_activations, train_val_test_baseline_sueno_merged_no_na):
        mask = user_level_activations.index.isin(list(baseline_sueno_merged_dataset.index))
        user_level_activations_filtered = user_level_activations[mask]
        train_val_test_user_level_activations_filtered.append(user_level_activations_filtered)

    return train_val_test_user_level_activations_filtered, train_val_test_baseline_sueno_merged_no_na


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

def get_train_valid_test_X_y_disease_specific(disease, user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_baseline_sueno_merge_no_na, aggregate='mean', number_of_layers=7):

    train_val_test_user_level_activations_filtered, train_val_test_baseline_sueno_merged_no_na = model_to_user_level_activations(user_datasets, model, np_train, np_val, np_test, batch_size, train_users, test_users, window_size, path_to_baseline_sueno_merge_no_na, aggregate='mean', number_of_layers=7)

    train_X = train_val_test_user_level_activations_filtered[0].values.tolist()
    val_X = train_val_test_user_level_activations_filtered[1].values.tolist()
    test_X = train_val_test_user_level_activations_filtered[2].values.tolist()

    train_y = train_val_test_baseline_sueno_merged_no_na[0][disease].values.tolist()
    val_y = train_val_test_baseline_sueno_merged_no_na[1][disease].values.tolist()
    test_y = train_val_test_baseline_sueno_merged_no_na[2][disease].values.tolist()

    return train_X, val_X, test_X, train_y, val_y, test_y

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

    return predictions, classifier, scaler

def prediction_metric_scores(prediction, truth):
    accuracy = accuracy_score(truth, prediction)
    precision = precision_score(truth, prediction, average='weighted')
    recall = recall_score(truth, prediction, average='weighted')
    f1 = f1_score(truth, prediction, average='weighted')
    confusion_mat = confusion_matrix(truth, prediction)

    return accuracy, precision, recall, f1, confusion_mat

def get_prediction_and_scores_from_model_and_classifier_disease_specific(disease, user_datasets, model, np_train, np_val, np_test, batch_size, 
                                    train_users, test_users, window_size, path_to_baseline_sueno_merge_no_na, classifier, aggregate='mean'):
    """
    We predict the diseases that the test set have by first getting user level activations from an intermediate layer of the trained simclr model. We use these user level activations to train simple classifiers
    Parameters:
        disease - disease to be predicted (diabetes, hypertension, sleep apnea, metabolic syndrome, insomnia)
        user_datasets - all data 
        model - trained simclr model
        np_train, np_val, np_test - windowed datasets that were used for the simclr model 
        batch_size - usually 512
        train_users, test_users - train and test user split 
        path_to_baseline_sueno_merge_no_na - path to dataset with disease information about each user 
        classifier - classifier being used to classify disease with the user level activation input 
        aggregate - metric used to group the window into users - mean, min, max, std, median
    
    Returns:
        trained_classifier - classifier trained on the input data for the particular disease
        scaler - scaler fit to X train data
        predictions - condition of each patient in the test set as predicted by the simple classifier 
        accuracy - how accurate the predictions were 
        precision - precision of the predictions 
        recall - recall of the predictions 
        f1 - f1 of the predictions 
        confusion_matrix - confusion_matrix of the predictions 
    """
    train_X, val_X, test_X, train_y, val_y, test_y = get_train_valid_test_X_y_disease_specific(disease, user_datasets, model, np_train, np_val, np_test, batch_size, train_users, test_users, window_size, path_to_baseline_sueno_merge_no_na, aggregate=aggregate)

    predictions, trained_classifier, scaler = classify_disease(classifier, train_X, val_X, test_X, train_y, val_y, test_y)

    accuracy, precision, recall, f1, confusion_matrix = prediction_metric_scores(predictions, test_y)

    return trained_classifier, scaler, predictions, accuracy, precision, recall, f1, confusion_matrix

def plot_prediction_metrics(path_to_prediction_metric_dictionary, path_to_save_figure_folder, show=False):
    """
    For each disease and for each metric plot the performance of each classifier
    """
    with open(path_to_prediction_metric_dictionary, 'rb') as f:
        disease_predictions = pickle.load(f)

    # {disease:{classifier:[predictions, accuracy, precision, recall, f1, confusion_matrix]}}
    for disease, classifier_data in disease_predictions.items():
        figure_save_name = f'{disease}_prediction_metrics_plot.png'
        figure_save_path = os.path.join(path_to_save_figure_folder, figure_save_name)
        plot_disease_prediction_metrics(disease, classifier_data, figure_save_path, show)

def plot_disease_prediction_metrics(disease, classifier_data, figure_save_path, show):
    classifier_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for classifier_name, classifier_scores in classifier_data.items():
        classifier_names.append(classifier_name)
        accuracies.append(classifier_scores[1])
        precisions.append(classifier_scores[2])
        recalls.append(classifier_scores[3])
        f1s.append(classifier_scores[4])
    
    plt.figure(figsize=(10,6))
    plt.plot(classifier_names, accuracies, label='accuracy')
    plt.scatter(classifier_names, accuracies, marker='*')
    plt.plot(classifier_names, precisions, label='precision')
    plt.scatter(classifier_names, precisions, marker='s')
    plt.plot(classifier_names, recalls, label='recall')
    plt.scatter(classifier_names, recalls, marker='^')
    plt.plot(classifier_names, f1s, label='f1')
    plt.scatter(classifier_names, f1s, marker='p')

    plt.xticks(rotation=45)
    title = f'{disease} prediction metrics'
    plt.title(title)

    plt.legend()

    plt.savefig(figure_save_path)
    plt.show()

def get_best_classifier_based_on_metric(disease, metric, disease_classifier_results, disease_trained_classifiers):
    """
    Each disease has several classifiers with corresponding metrics of accuracy, precision, recall and f1
    Return the best model for a particular disease, based on a particular metric
    Parameters:
        disease - one of diabetes, hypertension, sleep_apnea, metabolic_syndrome, insomnia
        metric - one of accuracy, precision, recall, f1
        disease_classifier_results - of the form {disease_name:{classifier_name:[predictions, accuracy, precision, recall, f1, confusion_matrix]}}
        disease_trained_classifiers - of the form {disease_name:{classifier_name:trained_classifier}}

    Returns:
        trained_classifier - the best classifier for the particular disease and metric
    """
    classifier_results_for_disease = disease_classifier_results[disease]
    trained_classifiers_for_disease = disease_trained_classifiers[disease]

    metric_to_index_dictionary = {'accuracy':1, 'precision':2, 'recall':3, 'f1':4}
    index = metric_to_index_dictionary[metric]

    # we can initialise to -1 as none of the metrics can be lower than 0
    best_metric_seen_so_far = -1
    best_classifier_seen_so_far = ''

    for classifier_name, results in classifier_results_for_disease.items():
        if results[index] > best_metric_seen_so_far:
            best_metric_seen_so_far = results[index]
            best_classifier_seen_so_far = classifier_name

    return trained_classifiers_for_disease[best_classifier_seen_so_far]

def mesa_predictions_from_user_datasets(mesa_user_datasets, mesa_np_train, mesa_np_val, mesa_np_test, mesa_train_users, mesa_test_users, window_size, batch_size, model, disease_classifier_results, disease_trained_classifiers, scaler, aggregate='mean', number_of_layers=7):
    """
    """
    # we can merge all the train val and test datasets together as we are only using the mesa to test 
    mesa_train_val_test_user_window_list = hchs_data_pre_processing.get_window_to_user_mapping(mesa_user_datasets, mesa_train_users, mesa_test_users, window_size)
    mesa_np_test_all = np.concatenate((mesa_np_train[0], mesa_np_val[0], mesa_np_test[0]))
    mesa_test_user_window_list = mesa_train_val_test_user_window_list[0] + mesa_train_val_test_user_window_list[1] + mesa_train_val_test_user_window_list[2]

    # get embeddings and user level activation from model 
    intermediate_model = simclr_models.extract_intermediate_model_from_base_model(model, intermediate_layer=number_of_layers)
    intermediate_model.summary()

    embedding = intermediate_model.predict(mesa_np_test_all, batch_size=batch_size)
    embedding_dataframe = pd.DataFrame(embedding)

    embedding_dataframe.index = mesa_test_user_window_list
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

    # scale train data based on scaler from hchs data
    train_X = np.array(user_level_activation.values.tolist())
    train_X_scaled = scaler.transform(train_X)

    # {disease: {metric: predictions}}
    disease_metric_optimised_predictions = {}
    # for each metric and disease determine which is the best classifier that has been trained on 
    for disease in disease_labels.keys():
        disease_specific_predictions = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            clf = get_best_classifier_based_on_metric(disease, metric, disease_classifier_results, disease_trained_classifiers)
            predictions = clf.predict(train_X_scaled)
            disease_specific_predictions[metric] = predictions

        disease_metric_optimised_predictions[disease] = disease_specific_predictions

    return disease_metric_optimised_predictions


def mesa_plot_predictions(disease_metric_optimised_predictions, save_directory, show=False):
    """
    Given the disease specific and metric optimised predictions plot the number of predictions 
    """
    # for example disease=diabetes metric_predictions = {'accuracy':[predictions], 'precision':[predictions] etc }
    for disease, metric_predictions in disease_metric_optimised_predictions.items():
        # e.g. {1: 'Non-diabetic', 2: 'Pre-diabetic', 3: 'Diabetic'} for diabetes
        disease_label_dict = disease_labels[disease]
        # e.g. {'accuracy': {'Non-diabetic':60, 'Pre-diabetic':30, 'Diabetic':10} etc }
        metric_percentage_dict = {}
        for metric, predictions in metric_predictions.items():
            # e.g. {1: 100, 2:50, 3:50} for diabetes
            count_of_categories = dict(Counter(predictions))
            number_of_predictions = len(predictions)
            # e.g. {'Non-diabetic':50, 'Pre-diabetic':25, 'Diabetic':25}
            percentage_per_label = {}
            for category, count in count_of_categories.items():
                category_full_name = disease_label_dict[category]
                percentage = count / number_of_predictions * 100
                percentage_per_label[category_full_name] = percentage

            metric_percentage_dict[metric] = percentage_per_label

        disease_labels = list(disease_label_dict.values())
        # {'Non-diabetic':[60,58,10,60], 'Pre-diabetic':[40,20,60,30], 'Diabetic':[0,22,30,10]}
        plotting_data = {}
        for disease_label in disease_labels:
            label_plotting_values = []
            for metric, percentage_dict in metric_percentage_dict.items():
                label_plotting_values.append(percentage_dict[disease_label])
            plotting_data[disease_label] = plotting_data

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        # plotting graph now 
        plt.figure(figsize=(12,8))
        bottom_coords = 0
        plts = []
        for layer in plotting_data.values():
            x = plt.bar(metrics, layer, bottom=bottom_coords)
            plts.append(x)
            bottom_coords = layer
        
        plt.xticks(rotation=45)
        plt.xlabel('Metrics')
        plt.ylabel('Predicted Percentage per Condition')

        legend_plotting = [x[0] for x in plts]
        plt.legend(legend_plotting, disease_labels)

        save_path = os.path.join(save_directory, f'mesa_predictions_for_{disease}.png')
        plt.savefig(save_path)
        plt.show()


        



