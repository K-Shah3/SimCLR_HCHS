# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import scipy.constants
import datetime
import tensorflow as tf
import numpy as np
import tqdm
import os
import pandas as pd


# %%
import hchs_data_pre_processing
import hchs_transformations
import simclr_models
import simclr_utitlities


# %%
working_directory = 'test_run_hchs'
dataset_save_path = os.path.join(os.getcwd(), "PickledData", "hchs")
user_datasets_path = os.path.join(dataset_save_path, "pid_to_data_label_dict.pickle")
user_dataset_resized_path = os.path.join(dataset_save_path, "user_dataset_resized.pickle")
path_to_test_train_split_dict = os.path.join(dataset_save_path, "test_train_split_dict.pickle")
path_to_reduced_test_train_split_dict = os.path.join(dataset_save_path, "reduced_test_train_split_dict.pickle")
sample_key = 163225
path_to_np_train = os.path.join(dataset_save_path, "np_train.pickle")
path_to_np_test = os.path.join(dataset_save_path, "np_test.pickle")
path_to_np_val = os.path.join(dataset_save_path, "np_val.pickle")

# %% [markdown]
# # Load hchs data from pickle path

# %%
# with open(user_datasets_path, 'rb') as f:
#     user_datasets = pickle.load(f)


# %%
# user_datasets_resized = {}

# for user, user_data_labels in user_datasets.items():
#     user_data = user_data_labels[0]
#     user_labels = user_data_labels[1]
#     print(user_data.shape, user_labels.shape, user)
#     new_data = np.array(user_data.values.tolist())[:, :-1].astype(np.float64)
#     new_labels = np.array([label[0] for label in user_labels.values.tolist()])
#     user_datasets_resized[user] = [new_data, new_labels]


# %%
# with open(user_dataset_resized_path, 'wb') as f:
#     pickle.dump(user_datasets_resized, f)


# %%
with open(user_dataset_resized_path, 'rb') as f:
    user_datasets = pickle.load(f)


# %%
user_datasets[sample_key]

# %% [markdown]
# # Pre Processing

# %%
# Parameters
window_size = 500
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

with open(path_to_reduced_test_train_split_dict, 'rb') as f:
    test_train_user_dict = pickle.load(f)

test_users = test_train_user_dict['test']
train_users = test_train_user_dict['train']

print(f'Test Numbers: {len(test_users)}, Train Numbers: {len(train_users)}')


# %%
# np_train, np_val, np_test = hchs_data_pre_processing.pre_process_dataset_composite(
#     user_datasets=user_datasets, 
#     label_map=label_map, 
#     output_shape=output_shape, 
#     train_users=train_users, 
#     test_users=test_users, 
#     window_size=window_size, 
#     shift=window_size//2, 
#     normalise_dataset=True, 
#     verbose=1
# )


# %%
# with open(path_to_np_train, 'wb') as f:
#     pickle.dump(np_train, f)
# with open(path_to_np_test, 'wb') as f:
#     pickle.dump(np_test, f)
# with open(path_to_np_val, 'wb') as f:
#     pickle.dump(np_val, f)


with open(path_to_np_train, 'rb') as f:
    np_train = pickle.load(f)
with open(path_to_np_test, 'rb') as f:
    np_test = pickle.load(f)
with open(path_to_np_val, 'rb') as f:
    np_val = pickle.load(f)

print("got here 1")
# %% [markdown]
# ## SimCLR Training

# %%
batch_size = 512
decay_steps = 1000
epochs = 200
# epochs = 3
temperature = 0.1
trasnformation_indices = [1, 2] # Use Scaling and rotation trasnformation

transform_funcs_vectorised = [
    hchs_transformations.noise_transform_vectorized, 
    hchs_transformations.scaling_transform_vectorized, 
    # transformations.rotation_transform_vectorized, 
    hchs_transformations.negate_transform_vectorized, 
    hchs_transformations.time_flip_transform_vectorized, 
    hchs_transformations.time_segment_permutation_transform_improved, 
    hchs_transformations.time_warp_transform_low_cost, 
    hchs_transformations.channel_shuffle_transform_vectorized
]
# transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']
transform_funcs_names = ['noised', 'scaled', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']


# %%
start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
transformation_function = simclr_utitlities.generate_combined_transform_function(transform_funcs_vectorised, indices=trasnformation_indices)

base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
simclr_model = simclr_models.attach_simclr_head(base_model)
simclr_model.summary()

trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

simclr_model_save_path = f"{working_directory}{start_time_str}_simclr.hdf5"
trained_simclr_model.save(simclr_model_save_path)

print("got here 2")

# %% [markdown]
# ## Linear Model

# %%



# %%

total_epochs = 50
batch_size = 200
tag = "linear_eval"

simclr_model = tf.keras.models.load_model(simclr_model_save_path)
linear_evaluation_model = simclr_models.create_linear_model_from_base_model(simclr_model, output_shape, intermediate_layer=7)

best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

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


# %%
# adding something for the sake of it


