#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np
# muzete pridat vlastni knihovny


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    # Filter out rows with missing coordinates
    df = df[df["d"].notna() & df["e"].notna()]
    # Convert to GeoDataFrame (EPSG:5514 is Krovak projection)
    geometry = geopandas.points_from_xy(df["d"], df["e"])
    return geopandas.GeoDataFrame(df, geometry=geometry, crs="EPSG:5514")

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami  """
    # Convert to mercator projection
    gdf = gdf.to_crs(epsg=3857)
    # Only rows with p10 == 4
    gdf = gdf[gdf["p10"] == 4]
    # Convert p2a to datetime
    gdf["p2a"] = pd.to_datetime(gdf["p2a"])
    # Settings
    years = [2021, 2022]
    region = "JHM" # Jihomoravský kraj
    # Create figure with subplots
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    # Plot each year
    for i, year in enumerate(years):
        # Filter out rows with different year
        gdf_year = gdf[gdf["p2a"].dt.year == year]
        # Filter out rows with different region
        gdf_year = gdf_year[gdf_year["region"] == region]

        # Plot
        gdf_year.plot(ax=axes[i], markersize=1, color="red")
        # Add basemap
        contextily.add_basemap(ax=axes[i], crs="EPSG:3857")
        # Set title
        axes[i].set_title(f"{region} kraj ({year})")
        # Remove axis
        axes[i].set_axis_off()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    # Convert to mercator projection
    gdf = gdf.to_crs(epsg=3857)
    region = "JHM" # Jihomoravský kraj
    # Only rows with p11 >= 4 and region
    gdf = gdf[(gdf["p11"] >= 4) & (gdf["region"] == region)]


    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    g = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(g, "geo1.png", True)
    plot_cluster(g, "geo2.png", True)
