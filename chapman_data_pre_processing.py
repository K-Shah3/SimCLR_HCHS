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
        4: Apply the label map and filter labels (see apply_label_map, filter_none_label)
        5: One-hot encode the labels (see tf.keras.utils.to_categorical)
        6: Split the training set into training and validation sets (see sklearn.model_selection.train_test_split)
    
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
        (np_train, np_val, np_test)
            windowed set of data points
    """
    
    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)
    
    print("step 1 done")
    # Step 2
    train_x, test_x = combine_windowed_dataset(user_datasets_windowed, train_users)
    print("step 2 done")
    # Step 3
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)
        train_x = normalise(train_x, means, stds)
        test_x = normalise(test_x, means, stds)
    print("step 3 done")
    


    print("step 4 done")
    # Step 5
    
    print("step 5 done")
    
    # Step 6
    # train_x_split, val_x_split, _, _ = sklearn.model_selection.train_test_split(train_x, train_x, test_size=0.25, random_state=42)
    train_val_x_split = np.split(train_x, [int(0.75*train_x.shape[0])])
    train_x_split, val_x_split = train_val_x_split[0], train_val_x_split[1]

    if verbose > 0:
        print(train_x_split.shape)
        print(val_x_split.shape)
        print(test_x.shape)

    print("step 6 done")

    return (train_x_split, val_x_split, test_x)

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
            Windowed version of the user_data
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

def combine_windowed_dataset(user_datasets_windowed, train_users, test_users=None, verbose=0):
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
        (train_x, test_x)
            train_x, test_x
                the resulting training/test input values as a single numpy array

    """
    
    train_x = []
    test_x = []
    for user_id in user_datasets_windowed:
        
        v = user_datasets_windowed[user_id]
        if user_id in train_users:
            if verbose > 0:
                print(f"{user_id} Train")
            train_x.append(v)
        elif test_users is None or user_id in test_users:
            if verbose > 0:
                print(f"{user_id} Test")
            test_x.append(v)
    

    if len(train_x) == 0:
        train_x = np.array([])
    else:
        train_x = np.concatenate(train_x)
    
    if len(test_x) == 0:
        test_x = np.array([])
    else:
        test_x = np.concatenate(test_x)

    return train_x, test_x

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

def get_window_to_user_mapping(user_datasets, train_users, test_users, window_size):
    """
    Return a list the same length as the size of the np_train, np_val and np_test datasets which have been windowed that
    map the row to which user it came from 
    
    Returns:
        [train_user_window_list, val_user_window_list, test_user_window_list]
    """
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size, shift=window_size//2)

    test_user_window_list = []
    train_user_window_list = []
    for user in user_datasets_windowed.keys():
        user_to_add = [user] * user_datasets_windowed[user].shape[0]
        if user in test_users:
            test_user_window_list += user_to_add

        if user in train_users:
            train_user_window_list += user_to_add

    train_val_user_window_list = np.split(train_user_window_list, [int(0.75*len(train_user_window_list))])
    train_user_window_list, val_user_window_list = list(train_val_user_window_list[0]), list(train_val_user_window_list[1])

    return [train_user_window_list, val_user_window_list, test_user_window_list]    