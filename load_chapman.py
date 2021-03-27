import pandas as pd
import os
import numpy as np
import pickle
import pathlib
import random
import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
enc = LabelEncoder()

random.seed(42)

path_to_chapman_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\Chapman")
path_to_ecg_data = os.path.join(path_to_chapman_dataset, "ECGDataDenoised")
path_to_pickled_chapman_data = os.path.join(os.getcwd(), "PickledData", "chapman")
path_to_patient_to_data_label_dict = os.path.join(path_to_pickled_chapman_data, "patient_to_data_label_dict.pickle")
path_to_patient_to_data_per_row_label_dict = os.path.join(path_to_pickled_chapman_data, "patient_to_data_label_per_row_dict.pickle")
path_to_test_train_split_dict = os.path.join(path_to_pickled_chapman_data, "test_train_split_dict.pickle")
path_to_test_train_split_dict_no_nan = os.path.join(path_to_pickled_chapman_data, "test_train_split_dict_no_nan.pickle")  
path_to_four_lead_user_datasets = os.path.join(path_to_pickled_chapman_data, 'four_lead_user_datasets.pickle')

# for testing 
path_to_test_train_split_reduced_dict = os.path.join(path_to_pickled_chapman_data, '100_users_reduced_test_train_split_dict.pickle')
path_to_reduced_four_lead_user_datasets = os.path.join(path_to_pickled_chapman_data, '100_users_datasets.pickle')
path_to_pickled_patient_to_rhythm_dict = os.path.join(path_to_pickled_chapman_data, "patient_to_rhythm_dict.pickle")

# no nan paths
path_to_four_lead_user_datasets_no_nan = os.path.join(path_to_pickled_chapman_data, 'four_lead_user_datasets_no_nan.pickle')
path_to_test_train_split_dict_no_nan = os.path.join(path_to_pickled_chapman_data, "test_train_split_dict_no_nan.pickle")  

path_to_reduced_four_lead_user_datasets_no_nan = os.path.join(path_to_pickled_chapman_data, 'reduced_four_lead_user_datasets_no_nan.pickle')
path_to_reduced_test_train_split_dict_no_nan = os.path.join(path_to_pickled_chapman_data, "reduced_test_train_split_dict_no_nan.pickle")  



# # define file path
# attributes_file_path = path_to_chapman_dataset + "\\AttributesDictionary.xlsx"
# condition_names_file_path = path_to_chapman_dataset + "\\ConditionNames.xlsx"
# diagnostics_file_path = path_to_chapman_dataset + "\\Diagnostics.xlsx"
# rhythm_names_file_path = path_to_chapman_dataset + "\\RhythmNames.xlsx"

# # read xlsx files 
# attributes_dictionary = pd.read_excel(attributes_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
# condition_names = pd.read_excel(condition_names_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
# diagnostics = pd.read_excel(diagnostics_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
# rhythm_names = pd.read_excel(rhythm_names_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)

# # 
# dates = diagnostics['FileName'].str.split('_',expand=True).iloc[:,1]
# dates.name = 'Dates'
# dates = pd.to_datetime(dates)
# diagnostics_with_dates = pd.concat((diagnostics,dates),1)

# """ Combine Rhythm Labels - map to only 4 rhythms """
# old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
# new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
# diagnostics_with_dates['Rhythm'] = diagnostics_with_dates['Rhythm'].replace(old_rhythms,new_rhythms)
# unique_labels = diagnostics_with_dates['Rhythm'].value_counts().index.tolist()
# print(unique_labels)
# label_enc = enc.fit(unique_labels)
# print(label_enc)

# sampling_rate = 500
# resampling_length = 5000
# # resampling_length = 2500

# leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
# desired_leads = leads # ['II','V2','aVL','aVR'] #['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
def load_patient_to_rhythm_mapping(path_to_chapman_dataset, user_datasets, path_to_pickled_chapman_data):
    path_to_chapman_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\Chapman")
    path_to_diagnostics = os.path.join(path_to_chapman_dataset, "Diagnostics.xlsx")
    diagnostics = pd.read_excel(path_to_diagnostics, 'Sheet1', dtype=str, index_col=None)
    old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
    new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
    diagnostics['Rhythm'] = diagnostics['Rhythm'].replace(old_rhythms,new_rhythms)
    unique_labels = diagnostics['Rhythm'].value_counts().index.tolist()
    print(unique_labels)
    patient_rhythm_dict = dict(zip(diagnostics['FileName'], diagnostics['Rhythm']))
    patient_rhythm_dict_keep = {}
    for p, r in patient_rhythm_dict.items():
        if p in user_datasets:
            patient_rhythm_dict_keep[p] = r

    print(f'user datasets length {len(user_datasets)}')
    print(f'mapping datasets length {len(patient_rhythm_dict_keep)}')

    path_to_pickled_patient_to_rhythm_dict = os.path.join(path_to_pickled_chapman_data, "patient_to_rhythm_dict.pickle")
    with open(path_to_pickled_patient_to_rhythm_dict, 'wb') as f:
        pickle.dump(patient_rhythm_dict_keep, f)


def load_patient_to_data_labels(path_to_ecg_data, path_to_pickled_chapman_data, diagnostics, enc,
                            sampling_rate = 500,
                            resampling_length = 5000,
                            leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],
                            desired_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']):
    patient_to_data_label_dict = {}
    lead_indices = np.where(np.in1d(leads,desired_leads))[0]
    for file in tqdm(pathlib.Path(path_to_ecg_data).iterdir()):
        patient = os.path.basename(file)[:-4]
        data = pd.read_csv(file, header=None)
        data_resampled = resample(data, resampling_length)
        data_resampled = data_resampled[:,lead_indices] #2500x12
        # data_resampled_float = np.array(data_resampled.values.to_list()).astype(np.float64)
        label = diagnostics['Rhythm'][diagnostics['FileName']==patient]
        encoded_label = enc.transform(label).item()

        patient_to_data_label_dict[patient] = [data_resampled, encoded_label]
    
    if not os.path.exists(path_to_pickled_chapman_data):
        os.makedirs(path_to_pickled_chapman_data)
        
    save_file = os.path.join(path_to_pickled_chapman_data, "patient_to_data_label_dict.pickle")
    with open(save_file, 'wb') as f:
        pickle.dump(patient_to_data_label_dict, f)
    return save_file

def load_patient_to_data_label_per_row(path_to_ecg_data, path_to_pickled_chapman_data, diagnostics, enc,
                            sampling_rate = 500,
                            resampling_length = 5000,
                            leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],
                            desired_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']):
    patient_to_data_label_dict = {}
    lead_indices = np.where(np.in1d(leads,desired_leads))[0]
    for file in tqdm(pathlib.Path(path_to_ecg_data).iterdir()):
        patient = os.path.basename(file)[:-4]
        data = pd.read_csv(file, header=None)
        data_resampled = resample(data, resampling_length)
        data_resampled = data_resampled[:,lead_indices] #5000x12
        # data_resampled_float = np.array(data_resampled.values.to_list()).astype(np.float64)
        label = diagnostics['Rhythm'][diagnostics['FileName']==patient]
        encoded_label = enc.transform(label).item()
        label_array = np.array([encoded_label for _ in range(data_resampled.shape[0])])

        patient_to_data_label_dict[patient] = [data_resampled, label_array]
    
    if not os.path.exists(path_to_pickled_chapman_data):
        os.makedirs(path_to_pickled_chapman_data)
        
    save_file = os.path.join(path_to_pickled_chapman_data, "patient_to_data_label_per_row_dict.pickle")
    with open(save_file, 'wb') as f:
        pickle.dump(patient_to_data_label_dict, f)
    return save_file

def get_fixed_test_train_split_and_pickle(patient_to_data_label_dict, path_to_pickled_chapman_data, test_percentage=20):
    patients = patient_to_data_label_dict.keys()
    number_of_patients = len(patients)
    test_fraction = test_percentage / 100
    number_of_test_patients = int(test_fraction * number_of_patients)
    test_patients = random.sample(patients, number_of_test_patients)
    train_patients = [patient for patient in patients if patient not in test_patients]

    test_train_split_dict = {'test': test_patients,
                            'train': train_patients}

    save_path = os.path.join(path_to_pickled_chapman_data, "test_train_split_dict.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)

def get_fixed_test_train_split_reduced_and_pickle(patient_to_data_label_dict, path_to_pickled_chapman_data):
    patients = patient_to_data_label_dict.keys()
    reduced_patients = random.sample(patients, 10)
    number_of_test_patients = 2

    test_patients = random.sample(reduced_patients, number_of_test_patients)
    train_patients = [patient for patient in reduced_patients if patient not in test_patients]

    test_train_split_dict = {'test': test_patients,
                            'train': train_patients}

    save_path = os.path.join(path_to_pickled_chapman_data, "reduced_test_train_split_dict.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)  
 

def load_four_lead_patient_data(path_to_ecg_data, path_to_pickled_chapman_data):
    user_datasets = {}
    lead_indicies = [1,3,4,7]
    number_of_all_zero_files = 0
    for file in tqdm(pathlib.Path(path_to_ecg_data).iterdir()):
        patient = os.path.basename(file)[:-4]
        data = pd.read_csv(file, header=None)
        if check_if_data_is_all_zero(data):
            number_of_all_zero_files += 1
            print(patient)
            continue
        
        data_resampled = resample(data, 2500)
        # data_normalised = (data_resampled - data_resampled.mean())/data_resampled.std()
        four_lead_data = data_resampled[:, lead_indicies]
        labels = np.array([0,1,2,3])
        user_datasets[patient] = (four_lead_data, labels)

    save_file_name = os.path.join(path_to_pickled_chapman_data, 'four_lead_user_datasets.pickle')
    with open(save_file_name, 'wb') as f:
        pickle.dump(user_datasets, f)

    print(f'number of all zero files: {number_of_all_zero_files}')

def load_four_lead_reduced_train_test_users_and_dataset(four_lead_user_datasets, path_to_pickled_chapman_data):
    patients = four_lead_user_datasets.keys()
    reduced_patients = random.sample(patients, 100)

    test_patients = random.sample(reduced_patients, 20)
    train_patients = [patient for patient in reduced_patients if patient not in test_patients]

    test_train_split_dict = {'train': train_patients, 'test': test_patients}

    reduced_user_dataset = {}
    for test_patient in test_patients:
        reduced_user_dataset[test_patient] = four_lead_user_datasets[test_patient]
    for train_patient in train_patients:
        reduced_user_dataset[train_patient] = four_lead_user_datasets[train_patient]

    train_test_file_path = os.path.join(path_to_pickled_chapman_data, '100_users_reduced_test_train_split_dict.pickle')
    user_datasets_file_path = os.path.join(path_to_pickled_chapman_data, '100_users_datasets.pickle')

    with open(train_test_file_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)

    with open(user_datasets_file_path, 'wb') as f:
        pickle.dump(reduced_user_dataset, f)

def check_if_data_is_all_zero(data):
    columns_zero = (data == 0).all()
    all_zero = (columns_zero == True).all()
    return all_zero


def no_nan_user_datasets_full_reduced_split_dicts(path_to_four_lead_user_datasets, path_to_pickled_chapman_data, test_percentage=20):
    '''
    From the original user datasets, create user dataset that has no nan, a reduce version for testing, and corresponding
    test_train split dicts
    '''
    with open(path_to_four_lead_user_datasets, 'rb') as f:
        user_datasets = pickle.load(f)
    
    # remove users that have nan values 
    delete_users = []
    for user, data in user_datasets.items():
        a = data[0]
        if np.isnan(a).any():
            print(user)
            delete_users.append(user)

    for user in delete_users:
        print(user)
        del user_datasets[user]

    user_datasets_save_path = os.path.join(path_to_pickled_chapman_data, 'four_lead_user_datasets_no_nan.pickle')
    with open(user_datasets_save_path, 'wb') as f:
        pickle.dump(user_datasets, f)

    patients = user_datasets.keys()
    number_of_patients = len(patients)
    test_fraction = test_percentage / 100
    number_of_test_patients = int(test_fraction * number_of_patients)
    test_patients = random.sample(patients, number_of_test_patients)
    train_patients = [patient for patient in patients if patient not in test_patients]

    test_train_split_dict = {'test': test_patients,
                            'train': train_patients}

    save_path = os.path.join(path_to_pickled_chapman_data, "test_train_split_dict_no_nan.pickle")

    with open(save_path, 'wb') as f:
        pickle.dump(test_train_split_dict, f)

    
    reduced_patients = random.sample(patients, 100)
    number_of_reduced_test_patients = 20
    reduced_test_patients = random.sample(reduced_patients, number_of_reduced_test_patients)
    reduced_train_patients = [patient for patient in reduced_patients if patient not in reduced_test_patients]

    reduced_test_train_split_dict = {'test': reduced_test_patients, 
                                    'train': reduced_train_patients}

    reduced_save_path = os.path.join(path_to_pickled_chapman_data, "reduced_test_train_split_dict_no_nan.pickle")

    with open(reduced_save_path, 'wb') as f:
        pickle.dump(reduced_test_train_split_dict, f)

    reduced_user_datasets = {}
    for patient in reduced_test_patients:
        reduced_user_datasets[patient] = user_datasets[patient]

    for patient in reduced_train_patients:
        reduced_user_datasets[patient] = user_datasets[patient]

    reduced_user_datasets_save_path = os.path.join(path_to_pickled_chapman_data, 'reduced_four_lead_user_datasets_no_nan.pickle')
    with open(reduced_user_datasets_save_path, 'wb') as f:
        pickle.dump(reduced_user_datasets, f)



def testing():
    path_to_chapman_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\Chapman")
    path_to_diagnostics = os.path.join(path_to_chapman_dataset, "Diagnostics.xlsx")
    print(path_to_diagnostics)
    diagnostics = pd.read_excel(path_to_diagnostics, 'Sheet1', dtype=str, index_col=None)
    print(diagnostics.head())

def main():
    # load_four_lead_patient_data_no_nan(path_to_ecg_data, path_to_pickled_chapman_data)
    # with open(path_to_four_lead_user_datasets, 'rb') as f:
    #     user_datasets = pickle.load(f)

    # # print('done 1')
    # # get_fixed_test_train_split_and_pickle(user_datasets, path_to_pickled_chapman_data)

    # # print('done 2')
    # # load_four_lead_reduced_train_test_users_and_dataset(user_datasets, path_to_pickled_chapman_data)
    # load_patient_to_rhythm_mapping(path_to_chapman_dataset, user_datasets, path_to_pickled_chapman_data)

    no_nan_user_datasets_full_reduced_split_dicts(path_to_four_lead_user_datasets, path_to_pickled_chapman_data, test_percentage=20)
    # no nan paths
    path_to_four_lead_user_datasets_no_nan = os.path.join(path_to_pickled_chapman_data, 'four_lead_user_datasets_no_nan.pickle')
    path_to_test_train_split_dict_no_nan = os.path.join(path_to_pickled_chapman_data, "test_train_split_dict_no_nan.pickle")  

    path_to_reduced_four_lead_user_datasets_no_nan = os.path.join(path_to_pickled_chapman_data, 'reduced_four_lead_user_datasets_no_nan.pickle')
    path_to_reduced_test_train_split_dict_no_nan = os.path.join(path_to_pickled_chapman_data, "reduced_test_train_split_dict_no_nan.pickle")

    with open(path_to_four_lead_user_datasets_no_nan, 'rb') as f:
        user_datasets_no_nan = pickle.load(f)

    print(len(user_datasets_no_nan.keys()))

    with open(path_to_reduced_four_lead_user_datasets, 'rb') as f:
        reduced_user_datasets_no_nan = pickle.load(f)

    print(len(reduced_user_datasets_no_nan.keys()))

    with open(path_to_reduced_test_train_split_dict_no_nan, 'rb') as f:
        reduced_test_train_split_dict_no_nan = pickle.load(f)

    print(len(reduced_test_train_split_dict_no_nan['train']))
    print(len(reduced_test_train_split_dict_no_nan['test']))  

    with open(path_to_test_train_split_dict_no_nan, 'rb') as f:
        test_train_split_dict_no_nan = pickle.load(f)

    print(len(test_train_split_dict_no_nan['train']))
    print(len(test_train_split_dict_no_nan['test']))  

    
    
if __name__ == "__main__":
    
    main()

