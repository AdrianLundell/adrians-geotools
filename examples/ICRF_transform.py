"""
A more involved example showing what data processing is possible, see: https://arxiv.org/pdf/2010.13625.pdf for the original transform

The script is possible to run as a python live script for example in Visual Studio Code.
"""

# %% Imports
from matplotlib import projections
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# %% Load data
df_2018 = pd.read_fwf("C:/Users/Adrian/Documents/NVI/IcrfTool/data/icrf3_src_posn", skiprows = 23, header = None)
columns_2018 = ["ICRF", "ICRF_Designation", "IERS_Destignation", "Defining_source", "Right_Ascension_h", "Right_Ascension_m", "Right_Ascension_s", "Declination_o", "Declination_prime", "Declination_bis", "Right_Ascention_Uncertainty_s", "Declination_Uncertainty_bis", "Correlation", "Mean_MJD", "First_MJD", "Last_MJD", "Nb_sess", "Nb_del", "Nb_rat"]
df_2018.columns = columns_2018
#df_2018 = df_2018[~df_2018.Defining_source.isna()]

columns_2009 = ["ICRF", "ICRF_Designation", "IERS_Destignation", "Defining_source", "Right_Ascension_h", "Right_Ascension_m", "Right_Ascension_s", "Declination_o", "Declination_prime", "Declination_bis", "Right_Ascention_Uncertainty_s", "Declination_Uncertainty_bis", "Correlation", "Mean_MJD", "First_MJD", "Last_MJD", "Nb_sess", "Nb_del"]
df_2009 = pd.read_fwf("C:/Users/Adrian/Documents/NVI/IcrfTool/data/icrf2_src_posn", skiprows = 23, header = None)
df_2009.columns = columns_2009
#df_2009 = df_2009[~df_2009.Defining_source.isna()]

# %% Transform data 
def transform(df):
    new_df = pd.DataFrame(
        data =  {"alpha" : df.Right_Ascension_h * 2*np.pi/24 + df.Right_Ascension_m * 2*np.pi/(24*60) + df.Right_Ascension_s * 2*np.pi/(24*60*60), #radians
                 "alpha_sigma" : df.Right_Ascention_Uncertainty_s * 360/(24*60*60),                                                                #degrees
                 "delta" : df.Declination_o * 2*np.pi/360 + df.Declination_prime * 2*np.pi/(360*60) + df.Declination_bis * 2*np.pi/(360*60*60),    #radians
                 "delta_sigma" : df.Declination_Uncertainty_bis * 1/(60*60),
                 "Correlation" : df.Correlation})                                                                      #degrees
    new_df.index = df.IERS_Destignation
    return new_df
    
df_2009_transformed = transform(df_2009) 
df_2018_transformed = transform(df_2018)
df = pd.merge(df_2009_transformed, df_2018_transformed, how = "left", left_index=True, right_index=True, suffixes = ("_2009", "_2018"))
print(f"Common: {len(df.index)} / 3414")


#%%
df["alpha_sigma"] = np.sqrt(df["alpha_sigma_2018"]**2 + df["alpha_sigma_2009"]**2)
df["delta_sigma"] = np.sqrt(df["delta_sigma_2018"]**2 + df["delta_sigma_2009"]**2)
df["alpha_residuals"] = df["alpha_2018"] - df["alpha_2009"]
df["delta_residuals"] = df["delta_2018"] - df["delta_2009"]

"""
As a criterion, a source was considered as an outlier if the angular
separation between the measured positions in the two catalogs
normalized by its formal uncertainty (a quantity hereafter
referred to as normalized separation) is larger than 5
"""
df = df[np.abs(df["alpha_residuals"])/np.radians(df["alpha_sigma"]) < 5]
df = df[np.abs(df["delta_residuals"])/np.radians(df["delta_sigma"]) < 5]

"""
Additionally,
all sources with an angular separation between catalogs larger than 5 millias
"""
df = df[np.abs(df["alpha_residuals"])*1000*180*60*60/np.pi < 5]
df = df[np.abs(df["delta_residuals"])*1000*180*60*60/np.pi < 5]

"""
an error ellipse in position with a semi-major axis larger
than 5 millias in either catalog were also discarded. 
"""

alpha_sigma_star2009 = df["alpha_sigma_2009"] * np.cos(df["delta_2009"])
semi_major_axis_2009 = 1/2 * (alpha_sigma_star2009**2 + df["delta_sigma_2009"]**2 + np.sqrt((alpha_sigma_star2009**2 - df["delta_sigma_2009"]**2)**2 + (2*alpha_sigma_star2009*df["delta_sigma_2009"]*df["Correlation_2009"])**2))
# df = df[np.sqrt(semi_major_axis_2009) * 1000*180*60*60/np.pi < 5]

alpha_sigma_star_2018 = df["alpha_sigma_2018"] * np.cos(df["delta_2018"])
semi_major_axis_2018 = 1/2 * (alpha_sigma_star_2018**2 + df["delta_sigma_2018"]**2 + np.sqrt((alpha_sigma_star_2018**2 - df["delta_sigma_2018"]**2)**2 + (2*alpha_sigma_star_2018*df["delta_sigma_2018"]*df["Correlation_2018"])**2))
# df = df[(np.sqrt(semi_major_axis_2018) * 1000*180*60*60/np.pi) < 5]

covariance_matrix_2009 = np.array([[df["alpha_sigma_2009"]**2, df["alpha_sigma_2009"]* df["delta_sigma_2009"]* df["Correlation_2009"]],
                             [df["alpha_sigma_2009"]* df["delta_sigma_2009"]* df["Correlation_2009"], df["delta_sigma_2009"]**2]]).T
eigenvalues_2009 = np.linalg.eigvals(covariance_matrix_2009)


# df = df[(np.sqrt(eigenvalues_2009[:,0])*1000*180*60*60/np.pi < 5) & (np.sqrt(eigenvalues_2009[:,1])*1000*180*60*60/np.pi < 5)]

covariance_matrix_2018 = np.array([[df["alpha_sigma_2018"]**2, df["alpha_sigma_2018"]* df["delta_sigma_2018"]* df["Correlation_2018"]],
                             [df["alpha_sigma_2018"]* df["delta_sigma_2018"]* df["Correlation_2018"], df["delta_sigma_2018"]**2]]).T
eigenvalues_2018 = np.linalg.eigvals(covariance_matrix_2018)

# df = df[(np.sqrt(eigenvalues_2018[:,0])*1000*180*60*60/np.pi < 5) & (np.sqrt(eigenvalues_2018[:,1])*1000*180*60*60/np.pi < 5)]



print(f"Used: {len(df.index)} / 2918")

# %% Create matrixes 
alpha = df["alpha_2018"]
delta = df["delta_2018"]


design_matrix1 = np.vstack([np.sin(delta) * np.cos(alpha),             #R1
                            np.sin(delta) * np.sin(alpha),             #R2
                            -np.cos(delta),                            #R3
                            -np.sin(alpha),                            #D1
                            np.cos(alpha),                             #D2
                            np.zeros(len(df.index)),                   #D3
                            np.sin(2*delta),                           #M20 
                            np.zeros(len(df.index)),                   #E20
                            np.sin(alpha)*np.sin(delta),               #E21 Re
                            np.cos(alpha)*np.sin(delta),               #E21 Im
                            -np.cos(alpha)*np.cos(2*delta),            #M21 RE
                            np.sin(alpha)*np.cos(2*delta),             #M21 Im
                            -2*np.sin(2*alpha)*np.cos(delta),          #E22 Re
                            -2*np.cos(2*alpha)*np.cos(delta),          #E22 Im
                            -np.cos(2*alpha)*np.sin(2*delta),          #M22 Re
                            np.sin(2*alpha)*np.sin(2*delta)            #M22 Im
                            ]).T
design_matrix2 = np.vstack([-np.sin(alpha),                            #R1 
                            np.cos(alpha),                             #R2
                            np.zeros(len(df.index)),                   #R3
                            -np.cos(alpha)*np.sin(delta),              #D1
                            -np.sin(alpha)*np.sin(delta),              #D2
                            np.cos(delta),                             #D3
                            np.zeros(len(df.index)),                   #M20
                            np.sin(2*delta),                           #E20
                            -np.cos(alpha)*np.cos(2*delta),             #E21 Re
                            np.sin(alpha)*np.cos(2*delta),              #E21 Im
                            -np.sin(alpha)*np.sin(delta),               #M21 Re
                            -np.cos(alpha)*np.sin(delta),               #M21 Im
                            -np.cos(2*alpha)*np.sin(2*delta),           #E22 Re
                            np.sin(2*alpha)*np.sin(2*delta),            #E22 Im
                            2*np.sin(2*alpha)*np.cos(delta),            #M22 Re
                            2*np.cos(2*alpha)*np.cos(delta)             #M22 Im
                            ]).T
design_matrix = np.vstack((design_matrix1, design_matrix2))

observation_vector1 = np.degrees(alpha - df["alpha_2009"]) * np.cos(delta)
observation_vector2 = np.degrees(delta - df["delta_2009"])
observation_vector = np.hstack((observation_vector1, observation_vector2))

observation_var_vector1 = df["alpha_sigma_2018"]**2 + df["alpha_sigma_2009"]**2
observation_var_vector2 = df["delta_sigma_2018"]**2 + df["delta_sigma_2009"]**2
observation_var_vector = np.hstack((observation_var_vector1, observation_var_vector2))
observation_var_matrix = np.diag(observation_var_vector)
# %% 
from HelmertTool.regression import *

params, sigmas = weighted_least_squares(design_matrix, observation_vector, observation_var_matrix)
params = params * 10**6 * 3600
sigmas = sigmas * 10**6 * 3600
true_params = [8, 15, 0, -22, -63, -90, 43, 5, -11, 3, -1, -5, 1, 3, -3, 2]

print(np.round(params).astype(int))
print(np.round(true_params))

#print(sigmas)

true_sigmas = [4, 4, 3, 4, 4, 4, 4, 4, 5, 5, 4, 4, 2, 2, 2, 2]
# %%
"""
Common 3414
Outliers 496

"""