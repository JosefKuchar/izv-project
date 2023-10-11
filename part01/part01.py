#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Josef Kuchař (xkucha28)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from typing import List, Callable, Dict, Any
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    """
    Integate using rectangle method :math:`x_i - x_{i-1} = f(\frac{x_{i-1} + x_i}{2})`

    :param f: function to integrate
    :param a: lower bound
    :param b: upper bound
    :param steps: number of steps
    :return: approximated integral
    """

    # Create space
    space = np.linspace(a, b, steps)

    # Rectangle width
    dx = (b - a) / steps

    # Calculate values
    values = f(space)

    # Calculate integral
    return np.sum(values * dx)



def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Generates a graph of :math:`f_a(x) = a^2 * x^3 * sin(x)` for each item in a

    :param a: list of a values
    :param show_figure: whether to show the figure
    :param save_path: path to save the figure to
    """

    def f(x):
        return np.array(a)[:, np.newaxis]**2 * (x**3 * np.sin(x))

    # Create space (200 so trapz values match the ones in assignment)
    space = np.linspace(-3, 3, 200)

    # Calculate values
    values = f(space)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(space, values.T)
    plt.xlabel("$x$")
    plt.ylabel("$f_a(x)$")
    plt.legend([f"$y_{{{a}}}(x)$" for a in a],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    plt.xlim(-3, 5)
    plt.ylim(0, 40)

    # Remove last two ticks
    plt.gca().xaxis.set_ticks(plt.gca().get_xticks()[:-2])

    # Fill under the curves and add annotations
    for arg, integral, vals in zip(a, np.trapz(f(space), space), values):
        plt.fill_between(space, vals, alpha=0.1)
        plt.annotate(f"$\\int f_{{{arg}}}(x)dx = {integral:.2f}$", xy=(3, vals[-1] - 0.5))

    # Save and show if specified
    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generates a graph of :math:`f_1(t) = 0.5 * cos(\\pi * t * 1 / 50)` and
    :math:`f_2(t) = 0.25 * (sin(\\pi * t) + sin(\\pi * t * 3 / 2))` and their sum

    :param show_figure: whether to show the figure
    :param save_path: path to save the figure to
    """

    def f1(t):
        return 0.5 * np.cos(np.pi * t * 1 / 50)
    def f2(t):
        return 0.25 * (np.sin(np.pi * t) + np.sin(np.pi * t * 3 / 2))

    # Create space
    space = np.linspace(0, 100, 10000)

    # Calculate values
    f1_values = f1(space)
    f2_values = f2(space)
    f1_f2_values = f1_values + f2_values

    # Create 3 subplots
    _, ax = plt.subplots(3, 1, figsize=(10, 12))

    # Plot
    ax[0].plot(space, f1_values)
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$f_1(t)$")

    ax[1].plot(space, f2_values)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$f_2(t)$")

    overlap = 0.01
    top_half = np.ma.masked_where(f1_f2_values + overlap <= f1_values, f1_f2_values)
    ax[2].plot(space, top_half, color="green")
    bottom_half = np.ma.masked_where(f1_f2_values - overlap >= f1_values, f1_f2_values)
    ax[2].plot(space, bottom_half, color="red")
    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("$f_1(t) + f_2(t)$")

    # Set limits for all subplots
    plt.setp(ax, xlim=(0, 100), ylim=(-0.8, 0.8))

    # Save and show if specified
    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()


def download_data() -> List[Dict[str, Any]]:
    """
    Downloads data from https://ehw.fit.vutbr.cz/izv/stanice.html and returns them as a list

    :return: list of stations
    """

    def get_float(s: str) -> float:
        """
        Helper function to clean the data
        """
        return float(s.strip().replace(",", ".").replace("°", ""))

    # Get the page
    # Iframe of https://ehw.fit.vutbr.cz/izv/stanice.html
    page = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html", timeout=5)

    # Parse the page
    soup = BeautifulSoup(page.content, "html.parser")

    # Get the table
    table = soup.find_all("table")[1]

    # Get the rows
    rows = table.find_all("tr", {"class": "nezvyraznit"})

    # Parse rows
    results = []
    for row in rows:
        columns = row.find_all("td")

        results.append({
            'position': columns[0].text,
            'lat': get_float(columns[2].text),
            'long': get_float(columns[4].text),
            'height': get_float(columns[6].text),
        })

    return results
