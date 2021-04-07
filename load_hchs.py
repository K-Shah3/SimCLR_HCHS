import pandas as pd
import os
import numpy as np
import pickle
import pathlib
import random
import tqdm
random.seed(42)
# adding something for the sake of it

path_to_hchs_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\hchs")
path_to_actigraphy_data = os.path.join(path_to_hchs_dataset, "actigraphy")
path_to_baseline_dataset = os.path.join(path_to_hchs_dataset, "datasets", "hchs-sol-baseline-dataset-0.5.0.csv")
path_to_sueno_dataset = os.path.join(path_to_hchs_dataset, "datasets", "hchs-sol-sueno-ancillary-dataset-0.5.0.csv")
path_to_pickled_hchs_data = os.path.join(os.getcwd(), "PickledData", "hchs")
path_to_data_label_dict = os.path.join(path_to_pickled_hchs_data, "pid_to_data_label_dict.pickle")
path_to_data_label_dict_no_time_column = os.path.join(path_to_pickled_hchs_data, "user_dataset_resized.pickle")
path_to_pid_to_disease_array = os.path.join(path_to_pickled_hchs_data, "pid_to_disease_array.pickle")
path_to_pid_to_meta_data_array = os.path.join(path_to_pickled_hchs_data, "pid_to_meta_data_array.pickle")
path_to_labels_mapping_dict = os.path.join(path_to_pickled_hchs_data, "labels_mapping_dict.pickle")
path_to_diseases_to_labels = os.path.join(path_to_pickled_hchs_data, "pid_to_diseases_to_labels_dict.pickle")
condition_names_list = ["pid_to_diabetes", "pid_to_sleep_apnea", "pid_to_hypertension", "pid_to_mets", "pid_to_mets_ncep", "pid_to_mets_ncep2"]
path_to_test_train_split_dict = os.path.join(path_to_pickled_hchs_data, "test_train_split_dict.pickle")  
path_to_test_train_split_reduced_dict = os.path.join(path_to_pickled_hchs_data, "reduced_test_train_split_dict.pickle")
path_to_condition_to_label_dict = os.path.join(path_to_pickled_hchs_data, "condition_to_label_dict.pickle")

# use for training on smaller number 
path_to_users_dataset_100 = os.path.join(path_to_pickled_hchs_data, '100_users_dataset.pickle')
path_to_test_train_users_100 = os.path.join(path_to_pickled_hchs_data, '100_users_reduced_test_train_users.pickle')


def load_data_and_active_labels(path_to_actigraphy_data, path_to_pickled_hchs_data):
    """Iterate through the actigraphy dataset. Extract pid from each file. 
    Filter the actigraphy data to: activity, whitelight, redlight, greenlight, bluelight, time
    Labels are: ACTIVE, REST-S, REST
    Remove invalid rows
    """
    pid_to_data_label_dict = {}
    for file in pathlib.Path(path_to_actigraphy_data).iterdir():
        filename = os.path.basename(file)
        pid = get_pid_from_filename(filename)
        data = pd.read_csv(file)
        processed_data = process_data(data)
        if processed_data is None:
            continue
        else:
            actigraphy_data, actigraphy_label = processed_data[0], processed_data[1]
            pid_to_data_label_dict[pid] = [actigraphy_data, actigraphy_label]

    if not os.path.exists(path_to_pickled_hchs_data):
        os.makedirs(path_to_pickled_hchs_data)

    save_file = path_to_pickled_hchs_data + "\\pid_to_data_label_dict.pickle"
    with open(save_file, 'wb') as f:
        pickle.dump(pid_to_data_label_dict, f)

    return save_file

def process_data(data):
    columns_of_interest = ["activity", "whitelight", "bluelight", "greenlight", "redlight", "time", "interval"]
    data_label_dataset = data[[c for c in data.columns if c in columns_of_interest]]
    data_label_dataset = data_label_dataset[data_label_dataset["interval"] != "EXCLUD"]
    data_label_dataset = data_label_dataset.dropna()
    
    if data_label_dataset.empty:
        return 
    
    label_column = ["interval"]

    label_dataset = data_label_dataset[[c for c in data_label_dataset.columns if c in label_column]]
    data_dataset = data_label_dataset[[c for c in data_label_dataset.columns if c not in label_column]]

    return data_dataset, label_dataset

def load_disease_data(path_to_baseline_dataset, path_to_sueno_dataset, path_to_pickled_hchs_data, pid_to_data_label_dict):
    """Function to get disease data for each patient
    pid: [dibaetes sleep_apnea hypertension metabolic_syndrome insomnia]
    Data is pickled as well"""
    pid_list = pid_to_data_label_dict.keys()
    baseline_dataset = pd.read_csv(path_to_baseline_dataset, low_memory=False)
    baseline_dataset_filtered = baseline_dataset[baseline_dataset['pid'].isin(pid_list)]
    sueno_dataset = pd.read_csv(path_to_sueno_dataset, low_memory=False)
    sueno_dataset_filtered = sueno_dataset[sueno_dataset['PID'].isin(pid_list)]

    baseline_dataset_columns = ["pid", "DIABETES3", "AHI_GE15", "HYPERTENSION", "METS_NCEP"]
    sueno_dataset_columns = ["PID", "ISI_C4"]

    baseline_data_cols_of_interest = baseline_dataset_filtered[[c for c in baseline_dataset_filtered.columns if c in baseline_dataset_columns]]
    sueno_data_cols_of_interest = sueno_dataset_filtered[[c for c in sueno_dataset_filtered.columns if c in sueno_dataset_columns]]

    baseline_sueno_merged = baseline_data_cols_of_interest.set_index('pid').join(sueno_data_cols_of_interest.set_index('PID'))
    baseline_sueno_merged["PID"] = baseline_sueno_merged.index

    baseline_sueno_merged = baseline_sueno_merged.replace({"ISI_C4": {1.0:1.0, 2.0:2.0, 3.0:3.0, 4.0:3.0}})
    baseline_sueno_merged = baseline_sueno_merged.rename(columns={"DIABETES3": "diabetes",
                                                                    "HYPERTENSION" : "hypertension",
                                                                    "AHI_GE15": "sleep_apnea",
                                                                    "METS_NCEP": "metabolic_syndrome",
                                                                    "ISI_C4": "insomnia",
                                                                    "PID": "pid"})

    path_to_pid_to_disease_array = os.path.join(path_to_pickled_hchs_data, "pid_to_disease_array.pickle")

    baseline_sueno_merged.to_pickle(path_to_pid_to_disease_array)
    # with open(path_to_pid_to_disease_array, 'wb') as f:
    #     pickle.dump(baseline_sueno_merged, f)

def load_actigraphy_data(path_to_data, save_path_folder, pickle_file_name="actigraphy_data.pickle"):
    """Function to load the actigraphy data and save as a pickled file"""
    pid_to_data_dict = {}
    for file in pathlib.Path(path_to_data).iterdir():
        filename = os.path.basename(file)
        pid = get_pid_from_filename(filename)
        data = pd.read_csv(file)
        pid_to_data_dict[pid] = data

    print("done dictionary")

    if not os.path.exists(save_path_folder):
        os.makedirs(save_path_folder)

    save_path = os.path.join(save_path_folder, pickle_file_name)

    with open(save_path, 'wb') as f:
        pickle.dump(pid_to_data_dict, f)

    print("done dumping")

    return save_path, pid_to_data_dict

def load_baseline_data(path_to_baseline_dataset, pickle_save_path, pid_to_data_dict):
    """Function to create dictionaries that map pids to diabetes, sleep apnea, hypertension and metabolic syndrome
    Pickle each dictionary as well"""
    pid_list = pid_to_data_dict.keys()
    baseline_dataset = pd.read_csv(path_to_baseline_dataset, dtype={'AGE':float, 'GENDERNUM':float, 'BMI':float, 'HEIGHT':float}, low_memory=False)
    baseline_dataset_filtered = baseline_dataset[baseline_dataset['pid'].isin(pid_list)]
    diabetes3_dict = dict(zip(baseline_dataset_filtered["pid"], baseline_dataset_filtered["DIABETES3"]))
    sleep_apnea_dict = dict(zip(baseline_dataset_filtered["pid"], baseline_dataset_filtered["AHI_GE15"]))
    hypertension_dict = dict(zip(baseline_dataset_filtered["pid"], baseline_dataset_filtered["HYPERTENSION"]))
    mets_ncep_dict = dict(zip(baseline_dataset_filtered["pid"], baseline_dataset_filtered["METS_NCEP"]))

    condition_dicts = [diabetes3_dict, sleep_apnea_dict, hypertension_dict, mets_ncep_dict]
    condition_names_list = ["diabetes", "sleep_apnea", "hypertension", "metabolic_syndrome"]
    
    pid_to_diseases_dict = {}
    for condition_dict, condition_name in zip(condition_dicts, condition_names_list):
        pid_to_diseases_dict[condition_name] = condition_dict
    
    save_path = os.path.join(path_to_pickled_hchs_data, "pid_to_diseases_dict.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(pid_to_diseases_dict, f)

    return pid_to_diseases_dict

def load_sueno_data(path_to_sueno_dataset, pickle_save_path, pid_to_data_dict):
    """Function to create dictionaries that map pids to insomnia. Merge together moderate severity and severe severity insomnia together. 
    Pickle each dictionary as well"""
    pid_list = pid_to_data_dict.keys()
    sueno_dataset = pd.read_csv(path_to_sueno_dataset)
    sueno_dataset_filtered = sueno_dataset[sueno_dataset['PID'].isin(pid_list)]
    # map value 4 to 3

    insomnia_dict = dict(zip(sueno_dataset_filtered["PID"], sueno_dataset_filtered["ISI_C4"]))
    for pid, insomnia in insomnia_dict.items():
        if insomnia == 4:
            insomnia_dict[pid] = 3

    save_path = os.path.join(pickle_save_path, "pid_to_insomnia.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(insomnia_dict, f)

    return insomnia_dict, save_path

def get_pid_from_filename(filename):
    """Function to get the pid from the filename, which is in the format hchs-sol-sueno-pid.csv
    The pid is zero padded so need to get rid of the zero padding"""
    number_string = "".join(i for i in filename if i.isdigit())
    pid = int(number_string)
    return pid

def compare_mets_dicts(dict1, dict2):
    number_diff = 0
    number_dict1_yes = 0
    number_dict2_yes = 0
    for pid, value1 in dict1.items():
        if value1 == 1.0:
            number_dict1_yes += 1
        value2 = dict2[pid]
        if value2 == 1.0:
            number_dict2_yes += 1
        if value1 != value2:
            number_diff += 1
    return number_diff, number_dict1_yes, number_dict2_yes

def load_label_mappings_for_diseases_and_gender(path_to_pickled_hchs_data):
    diabetes_labels_dict = {1: "Non-diabetic", 2: "Pre-diabetic", 3: "Diabetic"}
    sleep_apnea_labels_dict = {0: "Non-apneaic", 1:"Mild-to-severe-apnea"}
    hyptertension_labels_dict = {0: "No", 1:"Yes"}
    metabolic_syndrome_labels_dict = {0: "No", 1:"Yes"}
    insomnia_labels_dict = {1: "No clinically significant insomnia",
                            2: "Subthreshold insomnia",
                            3: "Clinical insomnia"}
    gender_labels_dict = {0: "female", 1:"male"}
    
    labels_mapping_dict = {"diabetes": diabetes_labels_dict,
                                "sleep_apnea": sleep_apnea_labels_dict,
                                "hypertension": hyptertension_labels_dict,
                                "metabolic_syndrome": metabolic_syndrome_labels_dict,
                                "insomnia": insomnia_labels_dict,
                              "gender": gender_labels_dict}

    

    save_path = os.path.join(path_to_pickled_hchs_data, "labels_mapping_dict.pickle")
    with open(save_path, 'wb') as f:
        pickle.dump(labels_mapping_dict, f)

    return save_path

def load_meta_data_array(path_to_baseline_dataset, path_to_pickled_hchs_data, pid_to_disease_array):
    """Function to extract pid to meta_data array (age, gender, bmi and height)
    Pickle each dictionary as well"""
    pid_list = pid_to_disease_array.keys()
    baseline_dataset = pd.read_csv(path_to_baseline_dataset, low_memory=False)
    baseline_dataset_filtered = baseline_dataset[baseline_dataset['pid'].isin(pid_list)]
    
    columns_of_interest = ["pid", "AGE", "GENDERNUM", "BMI", "HEIGHT"]
    baseline_dataset_filtered_cols_of_interest = baseline_dataset_filtered[[c for c in baseline_dataset_filtered.columns if c in columns_of_interest]]

    path_to_meta_data_array = os.path.join(path_to_pickled_hchs_data, "pid_to_meta_data_array.pickle")

    baseline_dataset_filtered_cols_of_interest.to_pickle(path_to_meta_data_array)

def get_fixed_test_train_split_and_pickle(pid_to_data_label_dict, path_to_pickled_hchs_data, test_percentage=20):
    pids = pid_to_data_label_dict.keys()
    number_of_pids = len(pids)
    test_fraction = test_percentage / 100
    number_of_test_pids = int(test_fraction * number_of_pids)
    test_pids = random.sample(pids, number_of_test_pids)
    train_pids = [pid for pid in pids if pid not in test_pids]

    test_train_split_dict = {'test': test_pids,
                            'train': train_pids}

    
    save_path = os.path.join(path_to_pickled_hchs_data, "test_train_split_dict.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)
    
def get_fixed_test_train_split_reduced_and_pickle(pid_to_data_label_dict, path_to_pickled_hchs_data):
    pids = pid_to_data_label_dict.keys()
    reduced_pids = random.sample(pids, 10)
    number_of_test_pids = 2

    test_pids = random.sample(reduced_pids, number_of_test_pids)
    train_pids = [pid for pid in reduced_pids if pid not in test_pids]

    test_train_split_dict = {'test': test_pids,
                            'train': train_pids}

    save_path = os.path.join(path_to_pickled_hchs_data, "reduced_test_train_split_dict.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)

def get_fixed_test_train_users_and_user_datasets_and_pickle(pid_to_data_label_dict, path_to_pickled_hchs_data, total_number=100):
    pids = pid_to_data_label_dict.keys()
    reduced_pids = random.sample(pids, total_number)
    number_of_test_pids = int(0.2 * total_number)

    test_pids = random.sample(reduced_pids, number_of_test_pids)
    train_pids = [pid for pid in reduced_pids if pid not in test_pids]

    reduced_test_train_users_dict = {'test': test_pids, 'train': train_pids}
    reduced_pid_to_data_user_dataset = {}
    for pid in reduced_pids:
        reduced_pid_to_data_user_dataset[pid] = pid_to_data_label_dict[pid]

    save_path_users_dict = os.path.join(path_to_pickled_hchs_data, '100_users_reduced_test_train_users.pickle')
    save_path_users_dataset = os.path.join(path_to_pickled_hchs_data, '100_users_dataset.pickle')

    with open(save_path_users_dataset, 'wb') as f:
        pickle.dump(reduced_pid_to_data_user_dataset, f)

    with open(save_path_users_dict, 'wb') as g:
        pickle.dump(reduced_test_train_users_dict, g)

def check_if_data_is_all_zero(data):
    columns_zero = (data == 0).all()
    all_zero = (columns_zero == True).all()
    return all_zero

def load_disease_user_dataset(path_to_data, path_to_pid_to_disease_array, path_to_pickled_hchs_data, disease):
    with open(path_to_pid_to_disease_array, 'rb') as f:
        pid_to_disease_array = pickle.load(f)

    pid_to_data_dict = {}
    i = 0
    number_skipped = 0
    for file in tqdm.tqdm(pathlib.Path(path_to_data).iterdir()):
        filename = os.path.basename(file)
        pid = get_pid_from_filename(filename)
        data = pd.read_csv(file)
        processed_data = extract_columns_of_interest(data)
        if processed_data is None:
            number_skipped += 1
            print(f'{pid} was None')
            continue
        processed_data = processed_data.to_numpy()
        # check if data all zero
        if check_if_data_is_all_zero(processed_data):
            print(f'{pid} has all zero data')
            number_skipped += 1
            continue
        # check if data has NaN
        if np.isnan(processed_data).any():
            print(f'{pid} has NaN data')
            number_skipped += 1
            continue

        disease_label = pid_to_disease_array[pid_to_disease_array['pid'] == int(pid)][disease].values[0]

        pid_to_data_dict[pid] = [processed_data, disease_label]

    print(f'number of skipped files: {number_skipped}')
    save_name = f'{disease}_user_datasets.pickle'
    print(save_name)
    save_path = os.path.join(path_to_pickled_hchs_data, save_name)
    with open(save_path, 'wb') as f:
        pickle.dump(pid_to_data_dict, f)

    
    
def extract_columns_of_interest(data):
    columns_of_interest = ["activity", "whitelight", "bluelight", "greenlight", "redlight", "interval"]
    data_label_dataset = data[[c for c in data.columns if c in columns_of_interest]]
    data_label_dataset = data_label_dataset[data_label_dataset["interval"] != "EXCLUD"]
    data_label_dataset = data_label_dataset.dropna()
    
    if data_label_dataset.empty:
        return 
    
    label_column = ["interval"]

    return data_label_dataset[[c for c in columns_of_interest if c != 'interval']]

if __name__ == "__main__":
    # load_data_and_active_labels(path_to_actigraphy_data, path_to_pickled_hchs_data)
    # path_to_pid_to_data_label_dict = path_to_pickled_hchs_data + "\\pid_to_data_label_dict.pickle"
    # with open(path_to_data_label_dict_no_time_column, 'rb') as f:
    #     pid_to_data_label_dict = pickle.load(f)

    # # print(pid_to_data_label_dict)
    # print("done loading data_label dict")
    # # # load_disease_data(path_to_baseline_dataset, path_to_sueno_dataset, path_to_pickled_hchs_data, pid_to_data_label_dict)

    # # pid_to_disease_array = pd.read_pickle(path_to_pid_to_disease_array)
    # # print(pid_to_disease_array)
    # # # load_meta_data_array(path_to_baseline_dataset, path_to_pickled_hchs_data, pid_to_data_label_dict)

    # # pid_to_meta_data_array = pd.read_pickle(path_to_pid_to_meta_data_array)

    # # print(pid_to_meta_data_array)
    # # # load_label_mappings_for_diseases_and_gender(path_to_pickled_hchs_data)
    
    # # # with open(path_to_labels_mapping_dict, 'rb') as f:
    # # #     labels_mapping_dict = pickle.load(f)

    # # # print(labels_mapping_dict)

    # # # get_fixed_test_train_split_and_pickle(pid_to_data_label_dict, path_to_pickled_hchs_data)

    # # with open(path_to_test_train_split_dict, 'rb') as f:
    # #     test_train_split_dict = pickle.load(f)

    # # print(len(test_train_split_dict['test']))
    # # print(len(test_train_split_dict['train']))
    
    # get_fixed_test_train_split_reduced_and_pickle(pid_to_data_label_dict, path_to_pickled_hchs_data)

    # with open(path_to_test_train_split_reduced_dict, 'rb') as f:
    #     test_train_split_dict = pickle.load(f)

    # print(len(test_train_split_dict['test']))
    # print(len(test_train_split_dict['train']))


    # print("hello")
    # with open(path_to_condition_to_label_dict, 'rb') as f:
    #     condition_to_label_dict = pickle.load(f)

    # print(condition_to_label_dict)

    # get_fixed_test_train_users_and_user_datasets_and_pickle(pid_to_data_label_dict, path_to_pickled_hchs_data)

    # with open(path_to_users_dataset_100, 'rb') as f:
    #     user_dataset_100 = pickle.load(f)

    # print(len(user_dataset_100.keys()))

    # with open(path_to_test_train_users_100, 'rb') as f:
    #     test_train_users_100 = pickle.load(f)

    # print(len(test_train_users_100['train']))
    # print(len(test_train_users_100['test']))

    # for key in test_train_users_100['test']:
    #     print(user_dataset_100[key][0])

    # testing(path_to_actigraphy_data, path_to_pid_to_disease_array, 'diabetes')
    for disease in ['hypertension', 'diabetes', 'sleep_apnea', 'metabolic_syndrome', 'insomnia']:
        load_disease_user_dataset(path_to_actigraphy_data, path_to_pid_to_disease_array, path_to_pickled_hchs_data, disease)