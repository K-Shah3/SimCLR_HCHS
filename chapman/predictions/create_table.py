import os
import pandas as pd
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
import numpy as np
import pickle
from tabulate import tabulate


def create_empty_table_lr_micro_macro():
    classifiers = ['logistic_regression']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro', 'auc']])
    return pd.DataFrame([], columns=cols)


def main_micro_macro():
    classifiers = ['logistic_regression']
    cols = pd.MultiIndex.from_product([classifiers, ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro', 'auc']])
    table = create_empty_table_lr_micro_macro()
    models_to_config_dict = {'20210227-235159_testing-False-ws-200-bs-256-transformations-01-lr-00001-agg-mean': 'Mean aggregate Noise Scaled Transformation',
                            '20210227-235154_testing-False-ws-200-bs-256-transformations-01-lr-00001-agg-std': 'Std aggregate Noise Scaled Transformation',
                            '20210227-235154_testing-False-ws-200-bs-256-transformations-23-lr-00001-agg-mean': 'Mean aggregated Negated Time-Flipped Transformation'}

    # models = next(os.walk(os.getcwd()))[1]
    models = ['20210227-235159_testing-False-ws-200-bs-256-transformations-01-lr-00001-agg-mean',
            '20210227-235154_testing-False-ws-200-bs-256-transformations-01-lr-00001-agg-std',
            '20210227-235154_testing-False-ws-200-bs-256-transformations-23-lr-00001-agg-mean']
    for model in models:
        print(model)
        path_to_chapman_predictions = os.path.join(os.getcwd(), model, 'lr_results.pickle')
        with open(path_to_chapman_predictions, 'rb') as f:
            results = pickle.load(f)

        classifier_metrics_data = [results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]
        np_classifier_metrics_data = np.array(classifier_metrics_data).reshape(1, len(classifier_metrics_data))
        classifier_metrics_dataframe = pd.DataFrame(np_classifier_metrics_data, columns=cols)
        table = table.append(classifier_metrics_dataframe)
    
    path_to_excel = os.path.join(os.getcwd(), f'heart_rhythm_classifier_results_lr_micro_macro.xlsx')
    model_to_config_index = [models_to_config_dict[model] for model in models]
    table.index = model_to_config_index
    table.to_excel(path_to_excel)
    
        
if __name__ == '__main__':
    main_micro_macro()
    