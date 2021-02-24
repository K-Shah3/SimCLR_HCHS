import os
import pickle5 as pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import tqdm
from matplotlib.lines import Line2D
import random
import sklearn.linear_model
import scipy.optimize
import sklearn.decomposition
import sklearn.model_selection
import scipy.constants
import ast
import sys

# Libraries for plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
sns.set_context('poster')

path_to_simclr_hchs = os.path.dirname(os.path.dirname(os.getcwd()))
path_to_hchs_dataset = os.path.join(os.path.dirname(path_to_simclr_hchs), "Datasets", "hchs")
path_to_actigraphy_data = os.path.join(path_to_hchs_dataset, "actigraphy")
path_to_baseline_dataset = os.path.join(path_to_hchs_dataset, "datasets", "hchs-sol-baseline-dataset-0.5.0.csv")
path_to_sueno_dataset = os.path.join(path_to_hchs_dataset, "datasets", "hchs-sol-sueno-ancillary-dataset-0.5.0.csv")
dataset_save_path = os.path.join(path_to_simclr_hchs, "PickledData", "hchs")
user_dataset_path = os.path.join(dataset_save_path, "user_dataset_resized.pickle")

path_to_all_data_one_array = os.path.join(os.getcwd(), 'all_data_hchs.pickle')
path_to_all_labels_one_array = os.path.join(os.getcwd(), 'all_labels_hchs.pickle')

disease_labels = {'diabetes': {1: 'Non-diabetic', 2: 'Pre-diabetic', 3: 'Diabetic'}, 'sleep_apnea': {0: 'No', 1: 'Yes'}, 'hypertension': {0: 'No', 1: 'Yes'}, 'metabolic_syndrome': {0: 'No', 1: 'Yes'}, 'insomnia': {1: 'No clinically significant insomnia', 2: 'Subthreshold insomnia', 3: 'Clinical insomnia'}, 'gender': {0: 'female', 1: 'male'}}

def main():
    with open(path_to_all_data_one_array, 'rb') as f:
        x = pickle.load(f)

    with open(path_to_all_labels_one_array, 'rb') as f:
        y = pickle.load(f)

    perplexity = 30.0
    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
    tsne_projections = tsne_model.fit_transform(x)

    unique_labels = ["ACTIVE", "REST", "REST-S"]

    print('starting the graph part')
    plt.figure(figsize=(16,8))
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=y,
        palette=sns.color_palette("hsv", len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True
    )
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title('Tsne for all actigraphy hchs data')

    plt.legend(loc='lower left', fontsize='small')
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(label)
    

    plt.savefig('tsne for all data and active/rest labels.png')

if __name__ == '__main__':
    main()
    