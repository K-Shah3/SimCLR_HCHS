import pandas as pd
import os
import numpy as np
import pickle
import pathlib
import random
random.seed(42)

ppath_to_mesa_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\MesaData\\mesa")
path_to_actigraphy_data = os.path.join(ppath_to_mesa_dataset, "actigraphy")
path_to_pickled_mesa_data = os.path.join(os.getcwd(), "PickledData", "mesa")
path_to_data_label_dict = os.path.join(path_to_pickled_mesa_data, "mesaid_to_data_label_dict.pickle")
path_to_pid_to_meta_data_array = os.path.join(path_to_pickled_mesa_data, "mesaid_to_meta_data_array.pickle")
path_to_test_train_split_dict = os.path.join(path_to_pickled_mesa_data, "test_train_split_dict.pickle")  
path_to_test_train_split_reduced_dict = os.path.join(path_to_pickled_mesa_data, "reduced_test_train_split_dict.pickle")
path_to_users_dataset_100 = os.path.join(path_to_pickled_mesa_data, '100_users_dataset.pickle')
path_to_test_train_users_100 = os.path.join(path_to_pickled_mesa_data, '100_users_reduced_test_train_users.pickle')

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

def get_fixed_test_train_split_and_pickle(mesaid_to_data_label_dict, path_to_pickled_mesa_data, test_percentage=20):
    mesaids = mesaid_to_data_label_dict.keys()
    number_of_mesaids = len(mesaids)
    test_fraction = test_percentage / 100
    number_of_test_mesaids = int(test_fraction * number_of_mesaids)
    test_mesaids = random.sample(mesaids, number_of_test_mesaids)
    train_mesaids = [mesaid for mesaid in mesaids if mesaid not in test_mesaids]

    test_train_split_dict = {'test': test_mesaids, 'train': train_mesaids}

    save_path = os.path.join(path_to_pickled_mesa_data, "test_train_split_dict.pickle")
    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)
    
def get_fixed_test_train_split_reduced_and_pickle(mesaid_to_data_label_dict, path_to_pickled_mesa_data):
    mesaids = mesaid_to_data_label_dict.keys()
    reduced_mesaids = random.sample(mesaids, 10)
    number_of_test_mesaids = 2

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

    


    
if __name__ == "__main__":
    
    with open(path_to_data_label_dict, 'rb') as f:
        mesaid_to_data_label_dict = pickle.load(f)

    get_fixed_test_train_users_and_user_datasets_and_pickle(mesaid_to_data_label_dict, path_to_pickled_mesa_data)

    with open(path_to_users_dataset_100, 'rb') as f:
        user_dataset_100 = pickle.load(f)

    print(len(user_dataset_100.keys()))

    with open(path_to_test_train_users_100, 'rb') as f:
        test_train_users_100 = pickle.load(f)

    print(len(test_train_users_100['train']))
    print(len(test_train_users_100['test']))

    for key in test_train_users_100['test']:
        print(user_dataset_100[key][0].shape)

    
    # 