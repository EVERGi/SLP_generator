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
    st_p = pd.read_csv(input_dir+'st_p_kproto10.csv', index_col=0)
    return st_p

@st.cache
def load_model():
    with open(model_dir+'kproto10.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

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
    title_alignment=
    """
    <style>
    Profile Generator {
    text-align: center
    }
    </style>
    """
    st.markdown(title_alignment, unsafe_allow_html=True)

    profiles = load_data()
    model = load_model()
    # double-ended slider morning/evening
    st.slider('Use of building in morning/evening:', 0.0, 1.0, 0.01)

