#!/usr/bin/env python3.11
# coding=utf-8

import io
import zipfile
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from zip file into one pandas dataframe.

    :param filename: path to zip file
    """

    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10",
               "p11", "p12", "p13a", "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19",
               "p20", "p21", "p22", "p23", "p24", "p27", "p28", "p34", "p35", "p39", "p44", "p45a",
               "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a", "p57", "p58", "a",
               "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t",
               "p5a"]

    # Region codes (switched keys and values for easier user)
    regions = {
        "00": "PHA",
        "01": "STC",
        "02": "JHC",
        "03": "PLK",
        "04": "ULK",
        "05": "HKK",
        "06": "JHM",
        "07": "MSK",
        "14": "OLK",
        "15": "ZLK",
        "16": "VYS",
        "17": "PAK",
        "18": "LBK",
        "19": "KVK",
    }

    dataframes = []

    # Load zip file
    with zipfile.ZipFile(filename, "r") as master_zf:
        # List all files in zip
        file_names = master_zf.namelist()

        # Load all zip files in zip
        for file_name in file_names:
            # Read zip file
            data = io.BytesIO(master_zf.read(file_name))
            # Load inner zip file
            with zipfile.ZipFile(data, "r") as zf:
                csv_file_names = zf.namelist()
                # Load all csv files in zip
                for csv_file_name in csv_file_names:
                    # Get region from file name
                    region = regions.get(csv_file_name[:2])
                    if region is None:
                        # Skip if region is not in dictionary
                        continue
                    # Read csv file
                    csv_data = io.BytesIO(zf.read(csv_file_name))
                    # Load csv file to dataframe (low memory=False because of mixed types)
                    df = pd.read_csv(csv_data, sep=";", names=headers, encoding="cp1250",
                                     low_memory=False)
                    # Add region column
                    df["region"] = region
                    # Add to list of dataframes
                    dataframes.append(df)

    # Concatenate all dataframes and return as one dataframe
    return pd.concat(dataframes, ignore_index=True)

def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Creates new dataframe from original dataframe and cleans it

    :param df: original dataframe
    """

    def print_size(d: pd.DataFrame, name: str):
        if verbose:
            usage = d.memory_usage(deep=True).sum() / 10**6
            print(f"{name}={usage:.1f} MB")

    print_size(df, "orig_size")
    df = df.copy()

    # Set p2a as datetime
    df["p2a"] = pd.to_datetime(df["p2a"], format="%Y-%m-%d")
    # Make some columns categorical
    cols = ["p47", "h", "i", "j", "k", "p", "q", "t"]
    for col in cols:
        df[col] = df[col].astype("category")
    # Make some columns numeric
    cols = ["a", "b", "d", "e", "f", "g", "l", "n", "o"]
    for col in cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    # Drop duplicates
    df = df.drop_duplicates(subset=["p1"])

    print_size(df, "new_size")
    return df

def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Plots number of accidents by state of driver

    :param df: dataframe
    :param fig_location: location to save figure
    :param show_figure: whether to show figure
    """

    #TODO
    sns.countplot(x="region", hue="p53", data=df)

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plots number of accidents by hour of day in 4 regions

    :param df: dataframe
    :param fig_location: location to save figure
    :param show_figure: whether to show figure
    """

    #TODO
    sns.countplot(x="region", hue="p53", data=df)

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Plot different types of fault in different regions

    :param df: dataframe
    :param fig_location: location to save figure
    :param show_figure: whether to show figure
    """

    #TODO
    sns.countplot(x="region", hue="p53", data=df)

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    raw_df = load_data("data/data.zip")
    clean_df = parse_data(raw_df, True)
    plot_state(clean_df, "01_state.png")
    plot_alcohol(clean_df, "02_alcohol.png", True)
    plot_fault(clean_df, "03_fault.png")
