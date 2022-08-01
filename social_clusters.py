import os
import glob
import pandas as pd
import numpy as np
import random
import pickle
import joblib
# K-mean clustering libraries
from kmodes.kprototypes import KPrototypes
# import minmax scaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.utils.functions import validation

random.seed(123)
model_dir = 'models/'
scaler_dir = 'scalers/'
input_dir = 'data/'

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    st_p = pd.read_csv(input_dir+'st_p_kproto10.csv', index_col=0, parse_dates=[0])
    return st_p

def load_model():
    model = pickle.load(open(model_dir+'kproto10.pkl',  "rb"))
    return model

@st.cache
def load_metadata():
    metadata = pd.read_csv('data/EANLIJST_METADATA.csv', index_col=0, sep   = ';')
    return metadata

if __name__ == "__main__":
    st.title("Profile Clustering")
    profiles = load_data()
    kproto = load_model()
    scaler = joblib.load(scaler_dir+'scaler.gz')
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
    yearly_consumption = st.number_input('Yearly consumption:', min_value=0, max_value=10000, value=5000, step=10)
    types = load_metadata()['Patrimonium Functietype'].unique()
    # Dropdown list for the type of building
    building_type = st.selectbox('Type of building:', types,  index=1)
    #st.write(kproto)
    row = np.array([yearly_consumption, weekend, evening, building_type])
    #st.write(np.shape(row.reshape(1,-1)))
    cluster = kproto.predict(row.reshape(1,-1), categorical=[3])
    st.write(cluster[0])
    #st.write(profiles.columns)
    ts = profiles[str(cluster[0])] * yearly_consumption
    day_p = ts.groupby(ts.index.hour).mean()
    fig, ax = plt.subplots()
    day_p.plot(ax = ax)
    plt.title('Average Day Profile', fontsize=30)
    st.plotly_chart(fig)