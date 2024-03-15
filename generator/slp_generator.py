# -*- coding: utf-8 -*-
"""
Created on Fri May 15 2024
@author: evgeny_genov
"""
# Imports
import pandas as pd
import numpy as np
import pickle
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
    # Map each type to its corresponding cluster name for quick lookup
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

    def __init__(self, config, node, **kwargs):
        """
        Generator constructor (mainly from Tool class)
        """
        super().__init__(config, node, **kwargs)

        # Set min and max values for the scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit([[0.0], [7830212.5]])

    def configure(self):
        """
        Multiple functionnalities to execute before the main tool
        """
        # Load data
        
        self.kris_profiles = self.load_data("Kris_profiles_reviewed")
        self.k_proto_profiles = self.load_data("st_p_kproto10")
        self.kproto = self.load_model("kproto10")

    def get_cluster_by_type(self, building_type):
        """
        Function to get the cluster of a building type using expert knowledge
        """
        return self.type_to_cluster_map.get(building_type)

    def load_model(self, name):
        """
        Load model k-prototypes from pickle file for clustering categorical and numerical data
        """
        model = pickle.load(open(f"{self.static_data_path}/{name}.pkl", "rb"))
        return model

    def adjust_day(self, year, month, day):
        """
        Adjust the day for February 29 dates in non-leap years
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
        Load data from csv file
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
        Get the csv with temperature profile and change its format
        """
        # ?? @evgenii
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

    def generate(self):
        """
        Main function of the tool
        """
        # Get normalized profile and scale it
        new_cons = self.config["nodes"][self.node]["load"]["yearly_consumption"]
        type_cons = self.config["nodes"][self.node]["load"]["type"]
        # if self.config["nodes"][self.node]["load"]["evening"] does not exist, it will be None
        # if it exists, it will be the value from the config
        evening = self.config["nodes"][self.node]["load"].get("evening")
        weekend = self.config["nodes"][self.node]["load"].get("weekend")
        # if one of the values is None, then both will be None
        if evening is None or weekend is None:
            evening = None
            weekend = None
        scaled_consumption = self.scaler.transform([[new_cons]])[0][0]
        base_load = self.get_profile(scaled_consumption, type_cons, evening, weekend)
        new_load = base_load * new_cons

        # Change name of load
        new_load = new_load.rename(
            columns={"Power (kW)": "Customer_" + str(self.node) + "_electric"}
        )

        # Save results
        name = self.intermediate_path + "/Consumer_" + str(self.node) + ".csv"
        self.results.append({"path": name, "res": new_load})

        # Debug message
        if self.debug:
            print("\n", "Load profiles have been configured.", "\n")

    def add_to_config(self, config):
        """
        Add LOADs to config file
        """
        # Add load
        load = {"name": "Consumer_" + str(self.node), "node_number": str(self.node)}
        config["loads"].append(load)

        # Add environment
        config["environments"].append(
            [
                "Consumer_" + str(self.node) + "_electric",
                "Consumer_" + str(self.node) + ".csv",
            ]
        )

        return config