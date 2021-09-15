import geopandas as gpd
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    
def plot_residuals(df, ax):
    """Plots alph-residuals scattered over a world map to axes"""
    #worldmap.plot(color="lightgrey", ax=ax)
    ax.set_title("αδ-residual components")

    if not df is None:
         q = ax.quiver(df.LAT, df.LONG, df.dAlpha, df.dDelta, color="k", scale=2)
         ax.quiverkey(q, 0.9,1.05, 10**-1, "10 cm", color = "red")
    
    ax.grid()

