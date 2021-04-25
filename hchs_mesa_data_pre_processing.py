import numpy as np
import scipy.stats
import pickle
import scipy
import datetime
import tensorflow as tf
import sklearn
import numpy
import os
import random
random.seed(7) 
import sklearn.model_selection
import tensorflow as tf

def pre_process_dataset_composite(user_datasets, train_users, test_users, window_size=200, shift=100, normalise_dataset=True, verbose=0):
    """
    A composite function to process a dataset
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        2: Split the dataset into training and test set (see combine_windowed_dataset)
        3: Normalise the datasets (see get_mean_std_from_user_list_format)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(data, labels)]}

        train_users
            list or set of users (corresponding to the user_id) to be used as training data

        test_users
            list or set of users (corresponding to the user_id) to be used as testing data

        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)

        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)

        normalise_dataset = True
            applies Z-normalisation if True

        verbose = 0
            debug messages are printed if > 0

    
    Return:
        (np_train, np_val, windowed_user_list_train, windowed_user_list_test)
            windowed set of data points
    """
    
    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)
    print("step 1 done")
    # Step 2
    train_x, test_x, train_user_window_list, test_user_window_list = combine_windowed_dataset(user_datasets_windowed, train_users, test_users)
    print("step 2 done")
    # Step 3
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)
        train_x = normalise(train_x, means, stds)
        test_x = normalise(test_x, means, stds)
    print("step 3 done")

    return (train_x, test_x, train_user_window_list, test_user_window_list)

def get_windows_dataset_from_user_list_format(user_datasets, window_size=200, shift=100, stride=1, verbose=0):
    """
    Create windows dataset in 'user-list' format using sliding windows

    Parameters:

        user_datasets
            dataset in the 'user-list' format {user_id: [(data, labels)]}
        
        window_size = 400
            size of the window (output)

        shift = 200
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)

        stride = 1
            stride of the window (dilation)

        verbose = 0
            debug messages are printed if > 0

    Return:
        user_dataset_windowed
            Windowed version of the user_data without labels 
            Windows from different trials are combined into one array
            type: {user_id: windowed_data}
            windowed_sensor_values have shape (num_window, window_size, channels)
            windowed_activity_labels have shape (num_window)
            Labels are decided by majority vote
    """
    user_dataset_windowed = {}

    for user_id in user_datasets:
        if verbose > 0:
            print(f"Processing {user_id}")
        x = []
        v, l = user_datasets[user_id]
        v_windowed = sliding_window_np(v, window_size, shift, stride)
        if len(v_windowed) > 0:
            x.append(v_windowed)
        if verbose > 0:
            print(f"Data: {v_windowed.shape}")

        # combine all trials
        user_dataset_windowed[user_id] = np.concatenate(x)
        
    return user_dataset_windowed

def sliding_window_np(X, window_size, shift, stride, offset=0, flatten=None):
    """
    Create sliding windows from an ndarray

    Parameters:
    
        X (numpy-array)
            The numpy array to be windowed
        
        shift (int)
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)

        stride (int)
            stride of the window (dilation)

        offset (int)
            starting index of the first window
        
        flatten (function (array) -> (value or array) )
            the function to be applied to a window after it is extracted
            can be used with get_mode (see above) for extracting the label by majority voting
            ignored if is None

    Return:

        Windowed ndarray
            shape[0] is the number of windows
    """

    overall_window_size = (window_size - 1) * stride + 1
    num_windows = (X.shape[0] - offset - (overall_window_size)) // shift + 1
    windows = []
    for i in range(num_windows):
        start_index = i * shift + offset
        this_window = X[start_index : start_index + overall_window_size : stride]
        if flatten is not None:
            this_window = flatten(this_window)
        windows.append(this_window)
    return np.array(windows)

def combine_windowed_dataset(user_datasets_windowed, train_users, test_users, verbose=0):
    """
    Combine a windowed 'user-list' dataset into training and test sets

    Parameters:

        user_dataset_windowed
            dataset in the windowed 'user-list' format {user_id: windowed_data}
        
        train_users
            list or set of users (corresponding to the user_id) to be used as training data

        test_users = None
            list or set of users (corresponding to the user_id) to be used as testing data
            if is None, then all users not in train_users will be treated as test users 

        verbose = 0
            debug messages are printed if > 0

    Return:
        (train_x, test_x, train_user_window_list, test_user_window_list)
            train_x, test_x
                the resulting training/test input values as a single numpy array

    """
    
    train_x = []
    train_user_window_list = []
    test_x = []
    test_user_window_list = []

    for user_id in user_datasets_windowed:
        data = user_datasets_windowed[user_id]
        user_to_add = [user_id] * data.shape[0]
        if user_id in train_users:
            train_x.append(data)
            train_user_window_list += user_to_add
        if user_id in test_users:
            test_x.append(data)
            test_user_window_list += user_to_add
    
    train_x = np.concatenate(train_x)
    test_x = np.concatenate(test_x)
    
    return train_x, test_x, train_user_window_list, test_user_window_list

def get_mean_std_from_user_list_format(user_datasets, train_users):
    """
    Obtain and means and standard deviations from a 'user-list' dataset from training users only
    Take the mean and standard deviation for activity, white, blue, green and red light
    

    Parameters:

        user_datasets
            dataset in the 'user-list' format {user_id: [data, label]}
        
        train_users
            list or set of users (corresponding to the user_ids) from which the mean and std are extracted

    Return:
        (means, stds)
            means and stds of the particular users
            shape: (num_channels)

    """
    all_data = []
    for user in user_datasets.keys():
        if user in train_users:
            user_data = user_datasets[user][0]
            all_data.append(user_data)

    mean_std_data_combined = np.concatenate(all_data)
    
    means = np.mean(mean_std_data_combined, axis=0)
    stds = np.std(mean_std_data_combined, axis=0)
    
    return (means, stds)

def normalise(data, means, stds):
    """
    Normalise along the column for each of the leads, based on the means and stds given
    """
    for index, values in enumerate(zip(means, stds)):
        mean = means[index]
        std = stds[index]
        data[:, :,index] = (data[:, :,index] - mean) / std
    
    return data