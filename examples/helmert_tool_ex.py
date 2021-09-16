"""
The script is possible to run as a python live script for example in Visual Studio Code.
"""

#%%
import HelmertTool.calc as calc 
import HelmertTool.io as io

df_a = io.load_sta("data/2020d.sta")
df_b = io.load_sta("data/2020d_off_0_0_10p_rate_0_0_0.sta")

calc.calculate_parameters(df_a, df_b, weighted=False, selected_serie="UT1", t0 = 2000)
