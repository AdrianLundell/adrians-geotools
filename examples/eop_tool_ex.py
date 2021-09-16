"""
Workflow example of the EopTool.

The script is possible to run as a python live script for example in Visual Studio Code.
"""

#%%
import EopTool.calc as calc 
import EopTool.io as io

#Load
df_a = io.load("data/2020d.eob")
df_b = io.load("data/usno_finals.fil")

#Force same sampling rate
df_a = calc.change_epoch(df_a, df_b.index)

#Caclulate trend
calc.calculate_parameters(df_a, df_b, weighted=False, selected_serie="UT1", t0 = 2000)
