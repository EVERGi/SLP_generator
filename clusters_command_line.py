import os
import glob
import pandas as pd
import numpy as np
import pickle
import joblib
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler
from src.utils.functions import validation
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model_dir = 'models/'
scaler_dir = 'scalers/'
input_dir = 'data/'

# Assuming 'types.txt' contains the types separated by commas, as in the original code.
with open(input_dir+'types.txt', 'r') as f:
    types = f.read().split(',')


def load_data():
    st_p = pd.read_csv(input_dir+'st_p_kproto10.csv', index_col=0, parse_dates=[0])
    st_p.dropna(inplace=True)
    st_p.drop(st_p[st_p.values == np.inf].index, inplace=True)
    st_p.drop(st_p[st_p.index.duplicated()].index, inplace=True)
    return st_p


def load_model():
    model = pickle.load(open(model_dir+'kproto10.pkl',  "rb"))
    return model


def main(evening, weekend, yearly_consumption, building_type):
    profiles = load_data()
    kproto = load_model()
    scaler = joblib.load(scaler_dir+'scaler.gz')
    
    scaled_consumption = scaler.transform([[yearly_consumption]])[0][0]

    if building_type not in types:
        raise ValueError(f"Invalid building type. Please choose from {types}")

    building_type_index = types.index(building_type)
    row = np.array([scaled_consumption, weekend, evening, building_type_index])

    cluster = kproto.predict(row.reshape(1,-1), categorical=[3])
    print("## Predicted cluster:", cluster[0])

    ts = profiles[str(cluster[0])] * yearly_consumption
    day_p = ts.groupby(ts.index.hour).mean()
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    day_p.plot(ax=ax)
    ax.set_xlabel('Hour')
    ax.grid(True)
    plt.title('Average Day Profile', fontsize=18, loc='left')
    ax.set_ylabel('kWh', fontsize=15)
    plt.show()

    week_dist = ts.groupby(ts.index.weekday).sum()
    week_dist = week_dist / len(ts.resample('W').count())
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    week_dist.plot(ax=ax, kind='bar')
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Weekday distribution', fontsize=18, loc='left')
    ax.set_ylabel('kWh', fontsize=18)
    ax.set_xlabel('Weekday')
    plt.show()


def get_cluster_ts(evening, weekend, yearly_consumption, building_type):
    profiles = load_data()
    kproto = load_model()
    scaler = joblib.load(scaler_dir+'scaler.gz')
    
    scaled_consumption = scaler.transform([[yearly_consumption]])[0][0]

    if building_type not in types:
        raise ValueError(f"Invalid building type. Please choose from {types}")

    building_type_index = types.index(building_type)
    row = np.array([scaled_consumption, weekend, evening, building_type_index])

    cluster = kproto.predict(row.reshape(1,-1), categorical=[3])
    ts = profiles[str(cluster[0])] * yearly_consumption
    return ts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Clustering Command Line Tool")
    parser.add_argument("--evening", type=float, required=True, help="Use of building in the evening (0.0 to 1.0)")
    parser.add_argument("--weekend", type=float, required=True, help="Use of building in the weekends (0.0 to 1.0)")
    parser.add_argument("--yearly-consumption", type=int, required=True, help="Yearly consumption in kWh")
    parser.add_argument("--building-type", type=str, required=True, help="Type of building")

    args = parser.parse_args()

    main(args.evening, args.weekend, args.yearly_consumption, args.building_type)
