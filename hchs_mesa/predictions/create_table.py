import os
import pandas as pd
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
import numpy as np
import pickle
from tabulate import tabulate

def create_empty_table():
    classifiers = ['random_forest', 'logistic_regression', 'svc', 'bernoulli_nb', 'sgd', 'decision_tree', 'extra_tree', 'ada_boost', 'gradient_boosting']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision', 'recall', 'f1']])
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


    

        
if __name__ == '__main__':
    main()
    