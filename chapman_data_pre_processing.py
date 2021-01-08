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

def get_mode(np_array):
    """
    Get the mode (majority/most frequent value) from a 1D array
    """
    return scipy.stats.mode(np_array)[0]

    