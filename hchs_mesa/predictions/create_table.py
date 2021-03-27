import os
import pandas as pd
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
import numpy as np
import pickle
from tabulate import tabulate
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

disease_labels = {'diabetes': {1: 'Non-diabetic', 2: 'Pre-diabetic', 3: 'Diabetic'}, 'sleep_apnea': {0: 'No', 1: 'Yes'}, 'hypertension': {0: 'No', 1: 'Yes'}, 'metabolic_syndrome': {0: 'No', 1: 'Yes'}, 'insomnia': {1: 'No clinically significant insomnia', 2: 'Subthreshold insomnia', 3: 'Clinical insomnia'}}


def create_empty_table():
    classifiers = ['random_forest', 'logistic_regression', 'svc', 'bernoulli_nb', 'sgd', 'decision_tree', 'extra_tree', 'ada_boost', 'gradient_boosting']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision', 'recall', 'f1']])
    return pd.DataFrame([], columns=cols)

def create_empty_table_lr_micro_macro():
    classifiers = ['logistic_regression']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']])
    return pd.DataFrame([], columns=cols)


def main():
    classifiers = ['random_forest', 'logistic_regression', 'svc', 'bernoulli_nb', 'sgd', 'decision_tree', 'extra_tree', 'ada_boost', 'gradient_boosting']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision', 'recall', 'f1']])
    diseases = ['diabetes', 'sleep_apnea', 'hypertension', 'metabolic_syndrome', 'insomnia']
    disease_dataframe_dict = {disease: create_empty_table() for disease in diseases}
    models_to_config_dict = {'20210214-025516': 'Mean aggregate Noise Scaled Transformation',
                            '20210214-025529': 'Min aggregate Noise Scaled Transformation', 
                            '20210214-025532': 'Max aggregate Noise Scaled Transformation', 
                            '20210214-025602': 'Std aggregate Noise Scaled Transformation', 
                            '20210214-025619': 'Mean aggregate Scaled Negated Transformation', 
                            '20210214-025625': 'Mean aggregate Negated Time-Flipped Transformation', 
                            '20210214-025626': 'Mean aggregate Time-Flipped Permuted Transformation', 
                            '20210214-030106': 'Mean aggregate Permuted Time-warped Transformation'}

    models = next(os.walk(os.getcwd()))[1]
    for model in models:
        path_to_hchs_disease_predictions = os.path.join(os.getcwd(), model, 'disease_classifier_results.pickle')
        with open(path_to_hchs_disease_predictions, 'rb') as f:
            disease_classifier_results = pickle.load(f)

        for disease, classifier_results in disease_classifier_results.items():
            print(disease)
            classifier_metrics_data = []
            for results in classifier_results.values():
                accuracy, precision, recall, f1 = results[1], results[2], results[3], results[4]
                classifier_metrics_data.append(accuracy)
                classifier_metrics_data.append(precision)
                classifier_metrics_data.append(recall)
                classifier_metrics_data.append(f1)

            np_classifier_metrics_data = np.array(classifier_metrics_data).reshape(1, len(classifier_metrics_data))
            classifier_metrics_dataframe = pd.DataFrame(np_classifier_metrics_data, columns=cols)
            disease_dataframe = disease_dataframe_dict[disease]
            disease_dataframe = disease_dataframe.append(classifier_metrics_dataframe)
            disease_dataframe_dict[disease] = disease_dataframe
        

    for disease, table in disease_dataframe_dict.items():
        path_to_excel = os.path.join(os.getcwd(), f'{disease}_classifier_results.xlsx')
        model_to_config_index = [models_to_config_dict[model] for model in models]
        table.index = model_to_config_index
        table.to_excel(path_to_excel)

def view_results():
    
    models = ['20210225-210908-testing-f-ws-512-bs-512-transformations-0-1-lr-01-agg-mean',
            '20210226-002515-testing-f-ws-512-bs-512-transformations-2-3-lr-01-agg-mean',
            '20210226-002516-testing-f-ws-512-bs-512-transformations-0-1-lr-01-agg-std']
    for model in models:
        print(model)
        path_to_hchs_disease_predictions = os.path.join(os.getcwd(), model, 'disease_classifier_results.pickle')
        with open(path_to_hchs_disease_predictions, 'rb') as f:
            disease_classifier_results = pickle.load(f)

        confusion_matrices = []
        confusion_matrices_labels = []
        disease_names = []
        for disease, classifier_results in disease_classifier_results.items():
            disease_names.append(disease)
            disease_label = disease_labels[disease].values()
            confusion_matrices_labels.append(disease_label)
            for results in classifier_results.values():
                confusion_matrix = results[8]
                confusion_matrices.append(confusion_matrix)
                # plot_conf_matrix(confusion_matrix, disease_label)
        plot_conf_matrix(confusion_matrices, confusion_matrices_labels, disease_names)

def plot_testing():
    a_true = [1, 1, 1, 3, 3, 2, 2]
    a_pred = [1, 1, 1, 2, 3, 3, 3]
    b_true = [4, 5, 4, 5]
    b_pred = [4, 5, 4, 5]

    trues = [a_true, b_true]
    preds = [a_pred, b_pred]
    labels = [['health', 'not health', 'dying'], ['health', 'dying']]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    for ax, true, pred, label in zip(axes, trues, preds, labels):
        cm = confusion_matrix(true, pred, labels=label)
        ax.matshow(cm)
        ax.set_xticklabels([''] + label)
        ax.set_yticklabels([''] + label)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.show()



def plot_conf_matrix(confusion_matrices, labels, disease_names):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for ax, confusion_matrix, label, disease in zip(axs, confusion_matrices, labels, disease_names):
        df_cm = pd.DataFrame(confusion_matrix, list(label), list(label))
        sn.heatmap(df_cm, annot=True, annot_kws={'size': 10}, ax=ax)
        ax.imshow(df_cm)
        ax.xaxis.set_tick_params(labelsize=5, rotation=45)
        ax.yaxis.set_tick_params(labelsize=5, rotation=45)
        ax.set_title(disease)


    plt.show()

def main_micro_macro():
    classifiers = ['logistic_regression']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']])
    diseases = ['diabetes', 'sleep_apnea', 'hypertension', 'metabolic_syndrome', 'insomnia']
    disease_dataframe_dict = {disease: create_empty_table_lr_micro_macro() for disease in diseases}
    models_to_config_dict = {'20210225-112632': 'Mean aggregate Noise Scaled Transformation',
                            '20210225-114534': 'Mean aggregate Negated Time-Flipped Transformation', 
                            '20210225-114535': 'Std aggregate Noise Scaled Transformation', 
                            '20210225-210908-testing-f-ws-512-bs-512-transformations-0-1-lr-01-agg-mean': 'Mean aggregate Noise Scaled Transformation',
                            '20210226-002515-testing-f-ws-512-bs-512-transformations-2-3-lr-01-agg-mean': 'Mean aggregated Negated Time-Flipped Transformation',
                            '20210226-002516-testing-f-ws-512-bs-512-transformations-0-1-lr-01-agg-std': 'Std aggregate Noise Scaled Transformation'}

    # models = next(os.walk(os.getcwd()))[1]
    models = ['20210225-210908-testing-f-ws-512-bs-512-transformations-0-1-lr-01-agg-mean',
            '20210226-002515-testing-f-ws-512-bs-512-transformations-2-3-lr-01-agg-mean',
            '20210226-002516-testing-f-ws-512-bs-512-transformations-0-1-lr-01-agg-std']
    for model in models:
        path_to_hchs_disease_predictions = os.path.join(os.getcwd(), model, 'disease_classifier_results.pickle')
        with open(path_to_hchs_disease_predictions, 'rb') as f:
            disease_classifier_results = pickle.load(f)

        for disease, classifier_results in disease_classifier_results.items():
            print(disease)
            classifier_metrics_data = []
            for results in classifier_results.values():
                accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = results[1], results[2], results[3], results[4], results[5], results[6], results[7]
                classifier_metrics_data.append(accuracy)
                classifier_metrics_data.append(precision_micro)
                classifier_metrics_data.append(recall_micro)
                classifier_metrics_data.append(f1_micro)
                classifier_metrics_data.append(precision_macro)
                classifier_metrics_data.append(recall_macro)
                classifier_metrics_data.append(f1_macro)

            np_classifier_metrics_data = np.array(classifier_metrics_data).reshape(1, len(classifier_metrics_data))
            classifier_metrics_dataframe = pd.DataFrame(np_classifier_metrics_data, columns=cols)
            disease_dataframe = disease_dataframe_dict[disease]
            disease_dataframe = disease_dataframe.append(classifier_metrics_dataframe)
            disease_dataframe_dict[disease] = disease_dataframe
        

    for disease, table in disease_dataframe_dict.items():
        path_to_excel = os.path.join(os.getcwd(), f'{disease}_classifier_results_lr_micro_macro.xlsx')
        model_to_config_index = [models_to_config_dict[model] for model in models]
        table.index = model_to_config_index
        table.to_excel(path_to_excel)


if __name__ == '__main__':
    view_results()
    