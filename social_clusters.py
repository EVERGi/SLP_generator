import os
import glob
import pandas as pd
import numpy as np
import random
import pickle
# K-mean clustering libraries
from kmodes.kprototypes import KPrototypes
# import minmax scaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils.functions import validation

random.seed(123)
model_dir = 'models/'

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cpadapter
from cpadapter.utils import train_cal_test_split
from cpadapter.visualization import conditional_band_interval_plot

input_dir = './data/'

# class Profile():
#     """
#     A class to create synthetic load profiles
#     """
#     def __init__(
#         self,
#         params: dict,
#     ):

#     self.params = params

@st.cache
def load_data():
    features = pd.read_csv(input_dir+'features.csv', index_col=0)
    metadata = pd.read_csv('data/EANLIJST_METADATA.csv', index_col=0, sep   = ';')
    # ADD the functietype column to the features
    features['function'] = metadata['Patrimonium Functietype']
    # read more metrics from csv
    metrics = pd.read_csv('data/ts_metrics.csv', usecols = ['ID', 'mean', 'std'], index_col='ID')
    # add the metrics to the features
    features = features.join(metrics)
    features.isnull().sum()
    features.dropna(inplace=True)
    features['ID'] = features.index
    return features

def band_interval_plot(x, y: np.ndarray, lower: np.ndarray, upper: np.ndarray, conf_percentage: float, sort: bool) -> None:
    r"""Function used to plot the data in `y` and it's confidence interval
    This function plots `y`, with a line plot, and the interval defined by the
    `lower` and `upper` bounds, with a band plot.
    Parameters
    ----------
    y: numpy.ndarray
    Array of observation or predictions we want to plot
    lower: numpy.ndarray
    Array of lower bound predictions for the confidence interval
    upper: numpy.ndarray
    Array of upper bound predictions for the confidence interval
    conf_percetage: float
    The desired confidence level of the predicted confidente interval
    sort: bool
    Boolean variable that indicates if the data, and the respective lower
    and upper bound values should be sorted ascendingly.
    Returns
    -------
    None
    Notes
    -----
    This function must only be used for regression cases
    """
    if sort:
        idx = np.argsort(y)
        y = y[idx]
        lower = lower[idx]
        upper = upper[idx]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x, y.reshape(-1), label='data')
    conf = str(conf_percentage*100) + '%'
    ax.fill_between(x, lower, upper, label=conf, alpha=0.3)
    ax.legend()
    return fig
if __name__ == "__main__":
    

