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

@st.cache
def load_metadata():
    metadata = pd.read_csv('data/EANLIJST_METADATA.csv', index_col=0, sep   = ';')
    return metadata

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
    st.title("Profile Clustering")
    profiles = load_data()
    model = load_model()
    # double-ended slider morning/evening
    evening = st.slider('Use of building in the evening:', 0.0, 1.0, 0.01)
    if 0 <= evening < 0.25:
        st.markdown("The building is **_barely_ used** in the evening")
    elif 0.25 <= evening < 0.5:
        st.markdown("The building is **_sometimes_ used** in the evening")
    elif 0.5 <= evening < 0.75:
        st.markdown("The building is **_often_ used** in the evening")
    elif 0.75 <= evening <= 1.0:
        st.markdown("The building is **_mostly_ used** in the evening")
    # double-ended slider morning/evening
    weekend = st.slider('Use of building in the weekends:', 0.0, 1.0, 0.01)
    if 0 <= weekend < 0.25:
        st.markdown("The building is **_barely_ used** in the weekends")
    elif 0.25 <= weekend < 0.5:
        st.markdown("The building is **_sometimes_ used** in the weekends")
    elif 0.5 <= weekend < 0.75:
        st.markdown("The building is **_often_ used** in the weekends")
    elif 0.75 <= weekend <= 1.0:
        st.markdown("The building is **_mostly_ used** in the weekends")
    # Enter yearly consumption in float
    yearly_consumption = st.number_input('Yearly consumption:', min_value=0, max_value=10000, value=0, step=10)
    types = load_metadata()['Patrimonium Functietype'].unique()
    # Dropdown list for the type of building
    building_type = st.selectbox('Type of building:', types)
    # predict cluster
    cluster = model.predict([[evening, weekend, yearly_consumption, building_type]])
