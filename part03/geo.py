#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    region = "JHM"  # Jihomoravský kraj
    # Create figure with subplots
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    # Plot each year
    for i, year in enumerate(years):
        # Filter out rows with different year
        gdf_year = gdf[gdf["p2a"].dt.year == year]
        # Filter out rows with different region
        gdf_year = gdf_year[gdf_year["region"] == region]
        # Remove points outside czech republic (got from epsg.io)
        gdf_year = geopandas.clip(
            gdf_year, (1296371, 6163881, 2132898, 6655524))
        # Plot
        gdf_year.plot(ax=axes[i], markersize=1, color="red")
        # Add basemap
        contextily.add_basemap(
            ax=axes[i], crs="EPSG:3857", source=contextily.providers.OpenStreetMap.Mapnik)
        # Set title
        axes[i].set_title(f"{region} kraj ({year})")
        # Remove axis
        axes[i].set_axis_off()
        # Set xlim and ylim
        axes[i].set_xlim(1720138, 1986013)
        axes[i].set_ylim(6203881, 6391142)
    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    # Convert to mercator projection
    gdf = gdf.to_crs(epsg=3857)
    region = "JHM"  # Jihomoravský kraj
    # Only rows with p11 >= 4 and region
    gdf = gdf[(gdf["p11"] >= 4) & (gdf["region"] == region)]
    # Remove points outside czech republic (got from epsg.io)
    gdf = geopandas.clip(gdf, (1296371, 6163881, 2132898, 6655524))
    # Kmeans clustering (number of cluster is set to 12 based on assignment image)
    kmeans = sklearn.cluster.KMeans(n_clusters=12, n_init=3)
    # Fit
    coordinates = gdf["geometry"].apply(lambda x: x.coords[0])
    estimator = kmeans.fit(coordinates.tolist())
    # Predict
    gdf["cluster"] = estimator.predict(coordinates.tolist())
    groups = gdf.groupby("cluster")
    # Create figure
    _, ax = plt.subplots(figsize=(8, 8))
    # Add color map legend
    sm = plt.cm.ScalarMappable(
        cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max(groups.size())))
    # Plot each cluster
    for cluster, group in groups:
        # Get convex hull
        convex_hull = group["geometry"].unary_union.convex_hull
        # Plot
        geopandas.GeoSeries(convex_hull).plot(ax=ax, color="grey", alpha=0.5)
        # Plot individual accidents with color based on cluster size
        group.plot(ax=ax, markersize=1, color=sm.to_rgba(len(group)))
    # Add basemap
    contextily.add_basemap(ax=ax, crs="EPSG:3857",
                           source=contextily.providers.OpenStreetMap.Mapnik)
    # Set title
    ax.set_title(f"Nehody v {region} kraji s významnou měrou alkoholu")
    # Axins
    axins = inset_axes(ax, width="100%", height="5%",
                       loc="lower center", borderpad=-2)
    plt.colorbar(sm, cax=axins, orientation="horizontal",
                 label="Počet nehod v úseku")
    # Remove axis
    ax.set_axis_off()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    g = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(g, "geo1.png", False)
    plot_cluster(g, "geo2.png", False)
