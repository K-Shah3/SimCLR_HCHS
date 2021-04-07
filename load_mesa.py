import pandas as pd
import os
import numpy as np
import pickle
import pathlib
import random
import tqdm
random.seed(42)

path_to_mesa_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\MesaData\\mesa")
path_to_actigraphy_data = os.path.join(path_to_mesa_dataset, "actigraphy")
path_to_pickled_mesa_data = os.path.join(os.getcwd(), "PickledData", "mesa")
path_to_data_label_dict = os.path.join(path_to_pickled_mesa_data, "mesaid_to_data_label_dict.pickle")
path_to_pid_to_meta_data_array = os.path.join(path_to_pickled_mesa_data, "mesaid_to_meta_data_array.pickle")
path_to_test_train_split_dict = os.path.join(path_to_pickled_mesa_data, "test_train_split_dict.pickle")  
path_to_test_train_split_reduced_dict = os.path.join(path_to_pickled_mesa_data, "reduced_test_train_split_dict.pickle")
path_to_users_dataset_100 = os.path.join(path_to_pickled_mesa_data, '100_users_dataset.pickle')
path_to_test_train_users_100 = os.path.join(path_to_pickled_mesa_data, '100_users_reduced_test_train_users.pickle')
path_to_questionnarire_data = os.path.join(path_to_mesa_dataset, 'datasets', 'mesa-sleep-dataset-0.3.0.csv')
path_to_mesaid_to_disease_array = os.path.join(path_to_pickled_mesa_data, 'mesaid_to_disease_array.pickle')

def load_data_and_active_labels(path_to_actigraphy_data, path_to_pickled_mesa_data):
    mesaid_to_data_label_dict = {}
    for file in pathlib.Path(path_to_actigraphy_data).iterdir():
        filename = os.path.basename(file)
        mesaid = get_mesaid_from_filename(filename)
        data = pd.read_csv(file)
        processed_data = process_data(data)

        if process_data is None:
            continue
        else:
            actigraphy_data, actigraphy_label = processed_data[0], processed_data[1]
            mesaid_to_data_label_dict[mesaid] = [actigraphy_data, actigraphy_label]
        
    if not os.path.exists(path_to_pickled_mesa_data):
        os.makedirs(path_to_pickled_mesa_data)

    save_file = path_to_pickled_mesa_data + "\\mesaid_to_data_label_dict.pickle"
    with open(save_file, 'wb') as f:
        pickle.dump(mesaid_to_data_label_dict, f)

    return save_file

    
def get_mesaid_from_filename(filename):
    """Function to get the mesaid for the filename which is in the format mesa-sleep-mesaid.csv"""
    number_string = "".join(i for i in filename if i.isdigit())
    mesaid = int(number_string)
    return mesaid

def process_data(data):
    """Function to read the data file, split into data and labels """
    columns_of_interest = ["activity", "whitelight", "bluelight", "greenlight", "redlight", "interval"]

    data_label_dataset = data[[c for c in data.columns if c in columns_of_interest]]
    data_label_dataset = data_label_dataset[data_label_dataset["interval"] != "EXCLUDED"]
    data_label_dataset = data_label_dataset.dropna()

    if data_label_dataset.empty:
        return
    
    label_column = ["interval"]

    label_dataset = data_label_dataset[[c for c in data_label_dataset.columns if c in label_column]]
    data_dataset = data_label_dataset[[c for c in data_label_dataset.columns if c not in label_column]]

    new_data = np.array(data_dataset.values.tolist()).astype(np.float64)
    new_labels = np.array([label[0] for label in label_dataset.values.tolist()])

    return new_data, new_labels

def get_fixed_test_train_split_and_pickle(user_datasets, path_to_pickled_mesa_data, test_percentage=20):
    mesaids = user_datasets.keys()
    number_of_mesaids = len(mesaids)
    test_fraction = test_percentage / 100
    number_of_test_mesaids = int(test_fraction * number_of_mesaids)
    test_mesaids = random.sample(mesaids, number_of_test_mesaids)
    train_mesaids = [mesaid for mesaid in mesaids if mesaid not in test_mesaids]

    test_train_split_dict = {'test': test_mesaids, 'train': train_mesaids}

    save_path = os.path.join(path_to_pickled_mesa_data, "test_train_split_dict.pickle")
    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)
    
def get_fixed_test_train_split_reduced_and_pickle(user_datasets, path_to_pickled_mesa_data):
    mesaids = user_datasets.keys()
    reduced_mesaids = random.sample(mesaids, 100)
    number_of_test_mesaids = 20

    test_mesaids = random.sample(reduced_mesaids, number_of_test_mesaids)
    train_mesaids = [mesaid for mesaid in reduced_mesaids if mesaid not in test_mesaids]

    test_train_split_dict = {'test': test_mesaids,
                            'train': train_mesaids}

    save_path = os.path.join(path_to_pickled_mesa_data, "reduced_test_train_split_dict.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)


def get_fixed_test_train_users_and_user_datasets_and_pickle(mesaid_to_data_label_dict, path_to_pickled_mesa_data, total_number=100):
    mesaids = mesaid_to_data_label_dict.keys()
    reduced_mesaids = random.sample(mesaids, total_number)
    number_of_test_mesaids = int(0.2 * total_number)

    test_mesaids = random.sample(reduced_mesaids, number_of_test_mesaids)
    train_mesaids = [mesaid for mesaid in reduced_mesaids if mesaid not in test_mesaids]

    reduced_test_train_users_dict = {'test': test_mesaids, 'train': train_mesaids}
    reduced_mesaid_to_data_user_dataset = {}
    for mesaid in reduced_mesaids:
        reduced_mesaid_to_data_user_dataset[mesaid] = mesaid_to_data_label_dict[mesaid]

    save_path_users_dict = os.path.join(path_to_pickled_mesa_data, '100_users_reduced_test_train_users.pickle')
    save_path_users_dataset = os.path.join(path_to_pickled_mesa_data, '100_users_dataset.pickle')

    with open(save_path_users_dataset, 'wb') as f:
        pickle.dump(reduced_mesaid_to_data_user_dataset, f)

    with open(save_path_users_dict, 'wb') as g:
        pickle.dump(reduced_test_train_users_dict, g)

def get_mesaid_to_disease_array(path_to_questionnarire_data, path_to_pickled_mesa_data):
    questionnaire_data = pd.read_csv(path_to_questionnarire_data, low_memory=False)
    columns_of_interest = ['mesaid', 'slpapnea5', 'insmnia5']
    data = questionnaire_data[[c for c in questionnaire_data.columns if c in columns_of_interest]]
    data = data.rename(columns={'slpapnea5':'sleep_apnea', 'insmnia5':'insomnia'})
    save_name = 'mesaid_to_disease_array.pickle'
    save_path = os.path.join(path_to_pickled_mesa_data, save_name)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def extract_columns_of_interest(data):
    columns_of_interest = ["activity", "whitelight", "bluelight", "greenlight", "redlight", "interval"]
    data_label_dataset = data[[c for c in data.columns if c in columns_of_interest]]
    data_label_dataset = data_label_dataset[data_label_dataset["interval"] != "EXCLUD"]
    data_label_dataset = data_label_dataset.dropna()
    
    if data_label_dataset.empty:
        return 
    
    label_column = ["interval"]

    return data_label_dataset[[c for c in columns_of_interest if c != 'interval']]

def check_if_data_is_all_zero(data):
    columns_zero = (data == 0).all()
    all_zero = (columns_zero == True).all()
    return all_zero

def get_mesaid_list(path_to_actigraphy_data):
    mesaids = []
    for file in tqdm.tqdm(pathlib.Path(path_to_actigraphy_data).iterdir()):
        filename = os.path.basename(file)
        mesaid = get_mesaid_from_filename(filename)
        mesaids.append(mesaid)
    return mesaids

def load_disease_user_dataset(path_to_actigraphy_data, path_to_mesaid_to_disease_array, path_to_pickled_mesa_data, disease):
    print(disease)
    with open(path_to_mesaid_to_disease_array, 'rb') as f:
        mesaid_to_disease_array = pickle.load(f)

    mesaids_with_diseases = mesaid_to_disease_array['mesaid'].tolist()

    mesaid_to_data_dict = {}
    number_skipped = 0
    for file in tqdm.tqdm(pathlib.Path(path_to_actigraphy_data).iterdir()):
        filename = os.path.basename(file)
        mesaid = get_mesaid_from_filename(filename)
        # check if mesaid in disease table
        if mesaid not in mesaids_with_diseases:
            number_skipped += 1
            print(f'{mesaid} not in disease table')
            continue 
        
        data = pd.read_csv(file)
        processed_data = extract_columns_of_interest(data)
        
        if processed_data is None:
            number_skipped += 1
            print(f'{mesaid} was None')
            continue
        
        processed_data = processed_data.to_numpy()
        # check if data all zero
        if check_if_data_is_all_zero(processed_data):
            print(f'{mesaid} has all zero data')
            number_skipped += 1
            continue
        # check if data has NaN
        if np.isnan(processed_data).any():
            print(f'{mesaid} has NaN data')
            number_skipped += 1
            continue
        
        disease_label = mesaid_to_disease_array[mesaid_to_disease_array['mesaid'] == mesaid][disease].values[0]
        mesaid_to_data_dict[mesaid] = [processed_data, disease_label]

    print(f'number of skipped files: {number_skipped}')
    save_name = f'{disease}_user_datasets.pickle'
    print(save_name)
    save_path = os.path.join(path_to_pickled_mesa_data, save_name)
    with open(save_path, 'wb') as f:
        pickle.dump(mesaid_to_data_dict, f)

if __name__ == "__main__":
    with open(path_to_mesaid_to_disease_array, 'rb') as f:
        mesaid_to_disease_array = pickle.load(f)
        
    # diseases = ['sleep_apnea', 'insomnia']
    # for disease in diseases:
    #     load_disease_user_dataset(path_to_actigraphy_data, path_to_mesaid_to_disease_array, path_to_pickled_mesa_data, disease)
    
    with open(os.path.join(path_to_pickled_mesa_data, 'sleep_apnea_user_datasets.pickle'), 'rb') as f:
        sleep_apnea_user_datasets = pickle.load(f)

    with open(os.path.join(path_to_pickled_mesa_data, 'insomnia_user_datasets.pickle'), 'rb') as f:
        insomnia_user_datasets = pickle.load(f)

    get_fixed_test_train_split_reduced_and_pickle(sleep_apnea_user_datasets, path_to_pickled_mesa_data)

    with open(path_to_test_train_split_reduced_dict, 'rb') as f:
        test_train_split_dict = pickle.load(f)

    total = len(test_train_split_dict['train']) + len(test_train_split_dict['test'])
    print(total)
    print(len(sleep_apnea_user_datasets))
    print(len(insomnia_user_datasets))

    

    
