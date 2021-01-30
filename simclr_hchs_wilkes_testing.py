import pickle 
import scipy.constants
import datetime
import tensorflow as tf
import numpy as np
import tqdm
import os
import pandas as pd

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

import hchs_data_pre_processing
import hchs_transformations
import simclr_models
import simclr_utitlities

working_directory = 'test_run_hchs_small_size/'

if not os.path.exists(working_directory):
    os.makedirs(working_directory)

dataset_save_path = os.path.join(os.getcwd(), "PickledData", "hchs")
user_dataset_resized_path = os.path.join(dataset_save_path, "user_dataset_resized.pickle")
path_to_test_train_split_dict = os.path.join(dataset_save_path, "reduced_test_train_split_dict.pickle")
sample_key = 163225

with open(user_dataset_resized_path, 'rb') as f:
    user_datasets = pickle.load(f)

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

batch_size = 512
decay_steps = 1000
epochs = 3
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

total_epochs = 3
batch_size = 200
tag = "linear_eval"

simclr_model = tf.keras.models.load_model(simclr_model_save_path)
linear_evaluation_model = simclr_models.create_linear_model_from_base_model(simclr_model, output_shape, intermediate_layer=7)

best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

plt.figure(figsize=(12,8))
plt.plot(epoch_losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
epoch_save_name = f"{start_time_str}_epoch_loss_testing.png"
epoch_save_dir = os.path.join(os.getcwd(), "testing","plots", "epoch_loss", "hchs")
if not os.path.exists(epoch_save_dir):
    os.makedirs(epoch_save_dir)
plt_save_path = os.path.join(epoch_save_dir, epoch_save_name)
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

target_model = simclr_model 
perplexity = 30.0
intermediate_model = simclr_models.extract_intermediate_model_from_base_model(target_model, intermediate_layer=7)
intermediate_model.summary()

print("starting tsne")
embeddings = intermediate_model.predict(np_test[0], batch_size=600)
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
tsne_projections = tsne_model.fit_transform(embeddings)

embeddings = intermediate_model.predict(np_test[0], batch_size=600)
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
tsne_projections = tsne_model.fit_transform(embeddings)

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


plt.legend(loc='lower left', bbox_to_anchor=(0.25, -0.3), ncol=2)
legend = graph.legend_
for j, label in enumerate(unique_labels):
    legend.get_texts()[j].set_text(label_list_full_name[label]) 

tsne_save_name = f"{start_time_str}_tsne_testing.png"
tsne_save_dir = os.path.join(os.getcwd(), "testing","plots", "tsne", "hchs")
if not os.path.exists(tsne_save_dir):
    os.makedirs(tsne_save_dir)
tsne_plt_save_path = os.path.join(tsne_save_dir, tsne_save_name)
plt.savefig(tsne_plt_save_path)
plt.show()