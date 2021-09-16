"""
The script is possible to run as a python live script for example in Visual Studio Code.
"""
#%%
import IcrfTool.calc as calc
import IcrfTool.io as io

df_a = io.load_src_posn("icrf2_src_posn")
df_b = io.load_src_posn("icrf3_src_posn")

params = calc.calculate_parameters(df_a, df_b, weighted = False, type = "A")

df_c = calc.icrf_transform(df_a, parameters, type = "A")

calc.calculate_residuals(df_b, df_c)

