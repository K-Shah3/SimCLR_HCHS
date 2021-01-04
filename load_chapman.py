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
from tqdm import tqdm
enc = LabelEncoder()

random.seed(42)

path_to_chapman_dataset = os.path.join(os.path.dirname(os.getcwd()), "Datasets\\Chapman")
path_to_ecg_data = os.path.join(path_to_chapman_dataset, "ECGDataDenoised")
path_to_pickled_chapman_data = os.path.join(os.getcwd(), "PickledData", "chapman")
path_to_patient_to_data_label_dict = os.path.join(path_to_pickled_chapman_data, "patient_to_data_label_dict.pickle")
# define file path
attributes_file_path = path_to_chapman_dataset + "\\AttributesDictionary.xlsx"
condition_names_file_path = path_to_chapman_dataset + "\\ConditionNames.xlsx"
diagnostics_file_path = path_to_chapman_dataset + "\\Diagnostics.xlsx"
rhythm_names_file_path = path_to_chapman_dataset + "\\RhythmNames.xlsx"

# read xlsx files 
attributes_dictionary = pd.read_excel(attributes_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
condition_names = pd.read_excel(condition_names_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
diagnostics = pd.read_excel(diagnostics_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)
rhythm_names = pd.read_excel(rhythm_names_file_path, 'Sheet1', dtype=str, index_col=None).applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 
dates = diagnostics['FileName'].str.split('_',expand=True).iloc[:,1]
dates.name = 'Dates'
dates = pd.to_datetime(dates)
diagnostics_with_dates = pd.concat((diagnostics,dates),1)

""" Combine Rhythm Labels - map to only 4 rhythms """
old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
diagnostics_with_dates['Rhythm'] = diagnostics_with_dates['Rhythm'].replace(old_rhythms,new_rhythms)
unique_labels = diagnostics_with_dates['Rhythm'].value_counts().index.tolist()
print(unique_labels)
label_enc = enc.fit(unique_labels)
print(enc)

sampling_rate = 500
resampling_length = 5000
# resampling_length = 2500

leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
desired_leads = leads # ['II','V2','aVL','aVR'] #['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


def load_patient_to_data_labels(path_to_ecg_data, path_to_pickled_chapman_data, diagnostics, enc,
                            sampling_rate = 500,
                            resampling_length = 5000,
                            leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],
                            desired_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']):
    patient_to_data_label_dict = {}
    lead_indices = np.where(np.in1d(leads,desired_leads))[0]
    for file in pathlib.Path(path_to_ecg_data).iterdir():
        patient = os.path.basename(file)[:-4]
        data = pd.read_csv(file)
        data_resampled = resample(data, resampling_length)
        data_resampled = data_resampled[lead_indices,:] #2500x12
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

if __name__ == "__main__":
    load_patient_to_data_labels(path_to_ecg_data, path_to_pickled_chapman_data, diagnostics_with_dates, label_enc)
    with open(path_to_patient_to_data_label_dict, 'rb') as f:
        path_to_patient_to_data_label_dict = pickle.load(f)

    print(len(path_to_patient_to_data_label_dict.keys()))