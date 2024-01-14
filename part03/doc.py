#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess data """

    df["fatal"] = df["p13a"] > 0
    df["p2a"] = pd.to_datetime(df["p2a"])
    return df


def create_graph(df: pd.DataFrame):
    """ Create graph """

    # Set style
    sns.set_style("darkgrid")
    # Set figure size
    plt.figure(figsize=(6, 3))
    # Filter out overtaking
    overtaking = df[(df["p12"] >= 301) & (df["p12"] <= 400)]
    # Aggregate fatal accidents by date (p2a)
    fatal_by_date = overtaking.groupby("p2a")["fatal"].sum()
    # Convert index to DatetimeIndex
    fatal_by_date.index = pd.to_datetime(fatal_by_date.index)
    # Resample
    fatal_by_date = fatal_by_date.resample("Y").sum()
    # Plot with seaborn
    sns.lineplot(data=fatal_by_date, color="#690e1e")
    # Set title and labels
    plt.title("Fatální nehody při nesprávném předjíždění")
    plt.xlabel("Rok")
    plt.ylabel("Počet fatalních nehod")
    # Set ticks for every year
    plt.xticks(fatal_by_date.index, fatal_by_date.index.year)
    # Save figure as png
    plt.savefig("fig.png", bbox_inches="tight", dpi=300)


def print_stats(df: pd.DataFrame):
    """ Print stats used in text """

    overtaking = df[(df["p12"] >= 301) & (df["p12"] <= 400)]
    # Aggregate fatal accidents by date (p2a)
    fatal_by_date = overtaking.groupby("p2a")["fatal"].sum()
    # Convert index to DatetimeIndex
    fatal_by_date.index = pd.to_datetime(fatal_by_date.index)
    # Resample
    fatal_by_date = fatal_by_date.resample("Y").sum()
    # Calculate average
    avg = fatal_by_date.mean()
    print(
        "Prumerny pocet fatalnich nehod pri nespravnem predjizdeni za rok: {:.1f}".format(avg))
    fatal_rate = overtaking["fatal"].mean() * 100
    print("Procento fatalnich nehod pri nespravnem predjizdeni: {:.1f}%".format(
        fatal_rate))
    # Aggregate all accidents by date (p2a) - size
    all_by_date = overtaking.groupby("p2a").size()
    # Convert index to DatetimeIndex
    all_by_date.index = pd.to_datetime(all_by_date.index)
    # Resample
    all_by_date = all_by_date.resample("Y").sum()
    # Calculate average
    avg = all_by_date.mean()
    print(
        "Prumerny pocet nehod pri nespravnem predjizdeni za rok: {:.1f}".format(avg))


def print_table(df: pd.DataFrame):
    """ Print table used in text """

    overtaking = df[(df["p12"] >= 301) & (df["p12"] <= 400)]
    # Agregate fatal and non-fatal accidents by date (p2a)
    fatal_by_date = overtaking.groupby("p2a")["fatal"].sum()
    nonfatal_by_date = overtaking.groupby(
        "p2a")["fatal"].count() - fatal_by_date
    # Convert index to DatetimeIndex
    fatal_by_date.index = pd.to_datetime(fatal_by_date.index)
    nonfatal_by_date.index = pd.to_datetime(nonfatal_by_date.index)
    # Resample
    fatal_by_date = fatal_by_date.resample("Y").sum()
    nonfatal_by_date = nonfatal_by_date.resample("Y").sum()
    # Create dataframe
    stats_df = pd.DataFrame(
        {"fatal": fatal_by_date, "nonfatal": nonfatal_by_date})
    # Convert date to year
    stats_df.index = stats_df.index.year
    # Print as csv
    print(stats_df.to_csv())


if __name__ == "__main__":
    df = pd.read_pickle("accidents.pkl.gz")
    df = preprocess_data(df)
    create_graph(df)
    print("**Statistiky pro text**")
    print_stats(df)
    print("**Tabulka pro text**")
    print_table(df)
