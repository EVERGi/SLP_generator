# -*- coding: utf-8 -*-
"""
Created on Fri May 15 2024
@author: evgeny_genov
"""
# Imports
import pandas as pd
import numpy as np
import pickle
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler

from .tool import Tool


cl_A = ["Sporthal", "Sportcomplex", "Stadion", "Voetbalveld"]
cl_B = [
    "Administratief centrum",
    "Stadhuis/Gemeentehuis",
    "OCMW Administratief centrum",
]
cl_C = [
    "Lagere school",
    "School",
    "Kinderdagverblijf/BKO/IBO",
    "Algemene middelbare school",
    "Technische middelbare school",
    "Buitengewoon lager onderwijs (MPI)",
    "Buitengewoon middelbaar onderwijs (BUSO)",
    "Kleuterschool",
]
cl_D_1 = ["Containerpark", "Parking"]
cl_D_2 = ["Fontein"]
cl_D_3 = ["Kerk"]
cl_D_4 = ["Park"]
cl_D_5 = ["Pomp"]
cl_D_6 = ["Straatverlichting"]
cl_D_7 = ["Ziekenhuis"]
cl_E = [
    "Cultureel centrum",
    "Ontmoetingscentrum",
    "Bibliotheek",
    "Academie",
    "Museum",
    "Jeugdhuis",
]
cl_G = ["RVT/WZC/revalidatiecentrum", "Dienstencentrum/CAW/dagverblijf"]
cl_H = ["Werkplaats"]
cl_I = ["Zwembad"]
cl_K = ["Brandweerkazerne", "Politiegebouw"]
cl_F = ["OCMW Woningen"]


class Generator(Tool):
    """
    A class that represents a generator.

    Attributes:
        data_path (str): The path to the data.
        scaler (MinMaxScaler): The scaler object for scaling values.

    Methods:
        __init__(self, data_path): Initializes the Generator object.
        configure(self): Executes multiple functionalities before the main tool.
        get_cluster_by_type(self, building_type): Gets the cluster of a building type using expert knowledge.
        load_model(self, name): Loads a model from a pickle file for clustering categorical and numerical data.
        adjust_day(self, year, month, day): Adjusts the day for February 29 dates in non-leap years.
        load_data(self, file): Loads data from a CSV file.
        get_profile(self, scaled_cons, type, evening=None, weekend=None): Gets the temperature profile and changes its format.
    """

    type_to_cluster_map = {
        **{t: "A" for t in cl_A},
        **{t: "B" for t in cl_B},
        **{t: "C" for t in cl_C},
        **{t: "D_1" for t in cl_D_1},
        **{t: "D_2" for t in cl_D_2},
        **{t: "D_3" for t in cl_D_3},
        **{t: "D_4" for t in cl_D_4},
        **{t: "D_5" for t in cl_D_5},
        **{t: "D_6" for t in cl_D_6},
        **{t: "D_7" for t in cl_D_7},
        **{t: "E" for t in cl_E},
        **{t: "G" for t in cl_G},
        **{t: "H" for t in cl_H},
        **{t: "I" for t in cl_I},
        **{t: "K" for t in cl_K},
        **{t: "F" for t in cl_F},
    }

    def __init__(self, data_path):
        """
        Initializes the Generator object.

        Args:
            data_path (str): The path to the data.
        """
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit([[0.0], [7830212.5]])

    def configure(self):
        """
        Executes multiple functionalities before the main tool.
        """
        # Load data
        self.kris_profiles = self.load_data("Kris_profiles_reviewed")
        self.k_proto_profiles = self.load_data("st_p_kproto10")
        self.kproto = self.load_model("kproto10")

    def get_cluster_by_type(self, building_type):
        """
        Gets the cluster of a building type using expert knowledge.

        Args:
            building_type (str): The building type.

        Returns:
            str: The cluster of the building type.
        """
        return self.type_to_cluster_map.get(building_type)

    def load_model(self, name):
        """
        Loads a model from a pickle file for clustering categorical and numerical data.

        Args:
            name (str): The name of the model.

        Returns:
            object: The loaded model.
        """
        model = pickle.load(open(f"{self.static_data_path}/{name}.pkl", "rb"))
        return model

    def adjust_day(self, year, month, day):
        """
        Adjusts the day for February 29 dates in non-leap years.

        Args:
            year (int): The year.
            month (int): The month.
            day (int): The day.

        Returns:
            int: The adjusted day.
        """
        if (
            month == 2
            and day == 29
            and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        ):
            return 28  # Adjust to February 28
        return day

    def load_data(self, file):
        """
        Loads data from a CSV file.

        Args:
            file (str): The name of the CSV file.

        Returns:
            pd.DataFrame: The loaded data.
        """
        st_p = pd.read_csv(
            f"{self.static_data_path}/{file}.csv", index_col=0, parse_dates=[0]
        )
        # drop nan and inf values
        st_p.dropna(inplace=True)
        # drop inf values
        st_p.drop(st_p[st_p.values == np.inf].index, inplace=True)
        st_p.drop(st_p[st_p.index.duplicated()].index, inplace=True)
        return st_p

    def get_profile(self, scaled_cons, type, evening=None, weekend=None):
        """
        Gets the temperature profile and changes its format.

        Args:
            scaled_cons (float): The scaled consumption.
            type (str): The building type.
            evening (float, optional): The evening value. Defaults to None.
            weekend (float, optional): The weekend value. Defaults to None.

        Returns:
            pd.DataFrame: The temperature profile in the desired format.
        """
        if evening is not None:
            row = np.array([scaled_cons, weekend, evening, type])
            cluster = self.kproto.predict(row.reshape(1, -1), categorical=[3])[0]
            ts = self.k_proto_profiles[str(cluster)]
        else:
            cluster = self.get_cluster_by_type(type)
            ts = self.kris_profiles[str(cluster)]

        # Change format
        df = pd.DataFrame(ts.values, index=ts.index, columns=["Power (kW)"])
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        df.index = df.index.tz_localize("UTC").tz_convert(self.timezone)
        df.index = df.index.tz_convert("UTC")
        df.index = pd.to_datetime(
            {
                "year": self.simu_year,
                "month": df.index.month,
                "day": [
                    self.adjust_day(self.simu_year, m, d)
                    for m, d in zip(df.index.month, df.index.day)
                ],
                "hour": df.index.hour,
                "minute": df.index.minute,
            }
        )
        df.index.name = "date"
        if self.timestep == 1:
            df = df.resample("1h").mean()
        df.index = df.index.strftime("%d/%m/%Y %H:%M:%S")
        return df

