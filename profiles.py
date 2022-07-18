from locale import D_FMT
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import dill as pickle

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
import datetime
from datetime import date
from scipy.stats.stats import pearsonr
from src.group_ts_split import PurgedGroupTimeSeriesSplit

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

import cpadapter
from cpadapter.utils import train_cal_test_split
from cpadapter.visualization import conditional_band_interval_plot

features = ['month', 'weekday', 'hour', 'yearly_cons', 'function']
categoricals = ['month', 'weekday', 'hour', 'function']
target = ['Target']
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
    data = pd.read_csv('data/EANLIJST_METADATA.csv', index_col=0, sep   = ';')
    return data

def load_params(data):
    params = {}
    params['s'] = st.date_input(
         "Start date",
         datetime.date(2019, 7, 1))

    params['e'] = st.date_input(
         "End date",
         datetime.date(2019, 7, 6))

    #params['show_yearly_cons']  = not st.checkbox('Known yearly consumption')
    #params['yearly_cons'] = st.sidebar.selectbox("What is a yearly consumption",yearly_cons_list)
    params['yearly_cons'] = st.slider('Yearly Consumption (kWh)', 0, 413982, 500)
    params['function'] = st.selectbox("Function", data['Patrimonium Functietype'].unique(), index=3)
    return params

def make_test_df(params):
    df = pd.DataFrame()
    df['datetime'] = pd.date_range(start=params['s'], end=params['e'], freq='1H')
    df['month'] = pd.DatetimeIndex(df['datetime']).month
    df['weekday'] = pd.DatetimeIndex(df['datetime']).weekday
    df['hour'] = pd.DatetimeIndex(df['datetime']).hour
    df['function'] = params['function']
    df['yearly_cons'] = params['yearly_cons']

    labelencoder = preprocessing.LabelEncoder()

    for col in categoricals:
        df[col] = labelencoder.fit_transform(df[col])
        df[col] = df[col].astype('int')
    return df

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

def profile_plot(x, y: np.ndarray, sort: bool) -> None:
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
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x, y.reshape(-1), label='data')
    ax.legend()
    return fig

if __name__ == "__main__":
    st.header("Synthetic Load Profile Generator")
    confidence = 0.8
    params = load_params(load_data())
    df = make_test_df(params)
    X_test = df[features]
    y_test = df['yearly_cons']
    test_dataset = lgb.Dataset(X_test,
                                y_test,
                                feature_name = features,
                                categorical_feature = categoricals,
                                free_raw_data=False).construct()

    # cp_model = pickle.load(open('train_prob_vvsg.p', 'rb'))
    # lower, pred_target, upper = cp_model.predict(test_dataset.get_data().to_numpy(), confidence)
    # lower, pred_target, upper = lower.clip(0), pred_target.clip(0), upper.clip(0)
    # lower, pred_target, upper = lower *  params['yearly_cons'], pred_target *  params['yearly_cons'], upper *  params['yearly_cons']
    # fig = band_interval_plot(df['datetime'], pred_target, lower, upper, 0.8, sort=False)

    model = pickle.load(open('train_vvsg.p', 'rb'))
    pred_target = model.predict(test_dataset.get_data().to_numpy())
    pred_target  = pred_target.clip(0)
    pred_target  = pred_target *  params['yearly_cons']
    fig = profile_plot(df['datetime'], pred_target, sort=False)


    st.write(fig)
