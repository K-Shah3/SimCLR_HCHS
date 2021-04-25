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

import hchs_data_pre_processing
import hchs_transformations
import simclr_models
import simclr_utitlities
import simclr_predictions

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

    return testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature


def main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature):

    if testing_flag:
        working_directory = 'hchs_mesa_testing/'
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
    else:
        working_directory = 'hchs_mesa/'
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
    transformation_indicies_string = ''.join([str(i) for i in transformation_indices])
    lr_string = str(initial_learning_rate).replace('.','')
    name_of_run = f'testing-{testing_flag}-ws-{window_size}-bs-{batch_size}-transformations-{transformation_indicies_string}-lr-{lr_string}-agg-{aggregate}-temp-{temperature}'
    print(name_of_run)
    
    dataset_save_path = os.path.join(os.getcwd(), "PickledData", "hchs")

    if testing_flag:
        path_to_user_datasets = os.path.join(dataset_save_path, "100_users_dataset.pickle")
        path_to_test_train_split_dict = os.path.join(dataset_save_path, "100_users_reduced_test_train_users.pickle")
    else:
        path_to_user_datasets  = os.path.join(dataset_save_path, "user_dataset_resized.pickle")
        path_to_test_train_split_dict = os.path.join(dataset_save_path, "test_train_split_dict.pickle")

    path_to_baseline_sueno_merge_no_na = os.path.join(dataset_save_path, "baseline_sueno_merge_no_na.pickle")
    disease_labels = {'diabetes': {1: 'Non-diabetic', 2: 'Pre-diabetic', 3: 'Diabetic'}, 'sleep_apnea': {0: 'No', 1: 'Yes'}, 'hypertension': {0: 'No', 1: 'Yes'}, 'metabolic_syndrome': {0: 'No', 1: 'Yes'}, 'insomnia': {1: 'No clinically significant insomnia', 2: 'Subthreshold insomnia', 3: 'Clinical insomnia'}}


    with open(path_to_user_datasets, 'rb') as f:
        user_datasets = pickle.load(f)

    # Parameters
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

    # a fixed user-split

    with open(path_to_test_train_split_dict, 'rb') as f:
        test_train_user_dict = pickle.load(f)

    test_users = test_train_user_dict['test']
    train_users = test_train_user_dict['train']

    print(f'Test Numbers: {len(test_users)}, Train Numbers: {len(train_users)}')

    np_train, np_val, np_test = hchs_data_pre_processing.pre_process_dataset_composite(
        user_datasets=user_datasets, 
        label_map=label_map, 
        output_shape=output_shape, 
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

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    tf.keras.backend.set_floatx('float32')

    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    transformation_function = simclr_utitlities.generate_combined_transform_function(transform_funcs_vectorised, indices=transformation_indices)

    base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
    simclr_model = simclr_models.attach_simclr_head(base_model)
    simclr_model.summary()

    trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    print("end of simclr bit")

    simclr_model_save_path = f"{working_directory}{start_time_str}_simclr.hdf5"
    trained_simclr_model.save(simclr_model_save_path)

    print("got here 2")

    if testing_flag:
        total_epochs = 3
    else:
        total_epochs = non_testing_linear_eval_epochs

    tag = "linear_eval"

    simclr_model = tf.keras.models.load_model(simclr_model_save_path)
    linear_evaluation_model = simclr_models.create_linear_model_from_base_model(simclr_model, output_shape, intermediate_layer=7)

    best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.hdf5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_model_file_name,
        monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
    )

    if plot_flag:
        print('plotting epoch loss')
        plt.figure(figsize=(12,8))
        plt.plot(epoch_losses)
        plt.ylabel("Loss", fontsize=10)
        plt.xlabel("Epoch", fontsize=10)
        epoch_save_name = f"{start_time_str}_{name_of_run}_epoch_loss.png"
        plt_save_path = os.path.join(os.getcwd(), "plots", "epoch_loss", "hchs", epoch_save_name)
        plt.savefig(plt_save_path)
        plt.show()

    training_history = linear_evaluation_model.fit(
        x = np_train[0],
        y = np_train[1],
        batch_size=batch_size,
        shuffle=True,
        epochs=total_epochs,
        callbacks=[best_model_callback],
        validation_data=np_val
    )

    print("got here 3")

    best_model = tf.keras.models.load_model(best_model_file_name)

    print("Model with lowest validation Loss:")
    print(simclr_utitlities.evaluate_model_simple(best_model.predict(np_test[0]), np_test[1], return_dict=True))
    print("Model in last epoch")
    print(simclr_utitlities.evaluate_model_simple(linear_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))

    if plot_flag:
        print("starting tsne")

        target_model = simclr_model 
        perplexity = 30.0
        intermediate_model = simclr_models.extract_intermediate_model_from_base_model(target_model, intermediate_layer=7)
        intermediate_model.summary()

        embeddings = intermediate_model.predict(np_test[0], batch_size=batch_size)
        tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
        tsne_projections = tsne_model.fit_transform(embeddings)
        print("done projections")

        labels_argmax = np.argmax(np_test[1], axis=1)
        unique_labels = np.unique(labels_argmax)

        plt.figure(figsize=(16,8))
        graph = sns.scatterplot(
            x=tsne_projections[:,0], y=tsne_projections[:,1],
            hue=labels_argmax,
            palette=sns.color_palette("hsv", len(unique_labels)),
            s=50,
            alpha=1.0,
            rasterized=True
        )
        plt.xticks([], [])
        plt.yticks([], [])


        plt.legend(loc='lower left', bbox_to_anchor=(0.25, -0.3), ncol=2, fontsize='small')
        legend = graph.legend_
        for j, label in enumerate(unique_labels):
            legend.get_texts()[j].set_text(label_list_full_name[label]) 

        tsne_save_name = f"{start_time_str}_{name_of_run}_tsne.png"
        tsne_plt_save_path = os.path.join(os.getcwd(), "plots", "tsne", "hchs", tsne_save_name)
        plt.savefig(tsne_plt_save_path)
        plt.show()

    if predict_flag:
        print("starting classifier predictions")

        #### testing predictions #### 

        # lr classifier

        logistic_regression_clf = LogisticRegression(max_iter=10000)
        
        classifier_names = ['logistic_regression']
        classifiers = [logistic_regression_clf]

        # {disease_name:{classifier_name:[predictions, accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, confusion_matrix]}}
        disease_classifier_results = {}
        # {disease_name:{classifier_name:trained_classifier}}
        disease_trained_classifiers = {}

        scaler_fit_to_hchs_train_data = None
        for disease in disease_labels.keys():
            classifier_results = {}
            trained_classifiers_dictionary = {}
            for classifier_name, classifier in zip(classifier_names, classifiers):
                print(disease)
                print(classifier_name)
                
                trained_classifier, scaler, predictions, test_y, accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc_macro_ovr, auc_macro_ovo, confusion_mat = simclr_predictions.get_prediction_and_scores_from_model_and_classifier_disease_specific_f1_micro_f1_macro(disease, user_datasets, simclr_model, np_train, np_val, np_test, batch_size, train_users, test_users, window_size, path_to_baseline_sueno_merge_no_na, classifier, aggregate=aggregate)
                
                print(f'accuracy: {accuracy}, precision_micro: {precision_micro}, recall_micro: {recall_micro}, f1_micro: {f1_micro}, precision_macro: {precision_macro}, recall_macro: {recall_macro}, f1_macro: {f1_macro}')
                print(f'auc scores: macro_ovr - {auc_macro_ovr}, macro_ovo - {auc_macro_ovo}')
                print(f'confusion matrix: {confusion_mat}')
                print("---------------")
                classifier_results[classifier_name] = [predictions, test_y, accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auc_macro_ovr, auc_macro_ovo, confusion_mat]
                trained_classifiers_dictionary[classifier_name] = trained_classifier
                scaler_fit_to_hchs_train_data = scaler
            
            disease_classifier_results[disease] = classifier_results
            disease_trained_classifiers[disease] = trained_classifiers_dictionary

        print(disease_classifier_results.keys())

        disease_classifier_results_directory = os.path.join(working_directory, 'predictions', f'{start_time_str}_{name_of_run}')
        if not os.path.exists(disease_classifier_results_directory):
            os.makedirs(disease_classifier_results_directory)

        disease_classifier_results_save_path = os.path.join(disease_classifier_results_directory, "disease_classifier_results.pickle")
        with open(disease_classifier_results_save_path, 'wb') as f:
            pickle.dump(disease_classifier_results, f)

        if plot_flag:
            print("plotting the prediction metrics")
            disease_classifier_plots_save_path = os.path.join(disease_classifier_results_directory, 'plots')
            if not os.path.exists(disease_classifier_plots_save_path):
                os.makedirs(disease_classifier_plots_save_path)

            simclr_predictions.plot_prediction_metrics_f1_micro_macro(disease_classifier_results_save_path, disease_classifier_plots_save_path, show=False)

        print("predicting mesa")

        mesa_dataset_save_path = os.path.join(os.getcwd(), "PickledData", "mesa")

        if testing_flag:
            mesa_user_datasets_path = os.path.join(mesa_dataset_save_path, "100_users_dataset.pickle")
            mesa_path_to_test_train_split_dict = os.path.join(mesa_dataset_save_path, "100_users_reduced_test_train_users.pickle")
        else:
            # mesa file path 
            mesa_user_datasets_path = os.path.join(mesa_dataset_save_path, "mesaid_to_data_label_dict.pickle")
            mesa_path_to_test_train_split_dict = os.path.join(mesa_dataset_save_path, "test_train_split_dict.pickle")

        mesa_sample_key = 3950

        with open(mesa_user_datasets_path, 'rb') as f:
            mesa_user_datasets = pickle.load(f)

        with open(mesa_path_to_test_train_split_dict, 'rb') as f:
            mesa_test_train_user_dict = pickle.load(f)

        mesa_test_users = mesa_test_train_user_dict['test']
        mesa_train_users = mesa_test_train_user_dict['train']

        print(f'MESA Test Numbers: {len(mesa_test_users)}, MESA Train Numbers: {len(mesa_train_users)}')

        mesa_np_train, mesa_np_val, mesa_np_test = hchs_data_pre_processing.pre_process_dataset_composite(
            user_datasets=mesa_user_datasets, 
            label_map=label_map, 
            output_shape=output_shape, 
            train_users=mesa_train_users, 
            test_users=mesa_test_users, 
            window_size=window_size, 
            shift=window_size//2, 
            normalise_dataset=True, 
            verbose=1
        )

        # generate predictions for each disease and optimised for each metric
        mesa_disease_metric_optimised_predictions = simclr_predictions.mesa_predictions_from_user_datasets_f1_micro_macro(mesa_user_datasets, mesa_np_train, mesa_np_val, mesa_np_test, mesa_train_users, mesa_test_users, window_size, batch_size, simclr_model, disease_classifier_results, disease_trained_classifiers, scaler_fit_to_hchs_train_data, aggregate=aggregate, number_of_layers=7)

        # save predictions 

        mesa_disease_metric_optimised_predictions_save_path = os.path.join(disease_classifier_results_directory, "mesa_disease_metric_optimised_predictions.pickle")
        with open(mesa_disease_metric_optimised_predictions_save_path, 'wb') as f:
            pickle.dump(mesa_disease_metric_optimised_predictions, f)

        if plot_flag:
            print("plotting mesa predictions")
            # plot mesa predictions 
            simclr_predictions.mesa_plot_predictions_micro_macro(mesa_disease_metric_optimised_predictions, disease_classifier_plots_save_path, show=False)

if __name__ == '__main__':
    testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature = parse_arguments(sys.argv)
    print('args read successfully, starting main')
    main(testing_flag, window_size, batch_size, non_testing_simclr_epochs, transformation_indices, initial_learning_rate, non_testing_linear_eval_epochs, plot_flag, predict_flag, aggregate, temperature)



