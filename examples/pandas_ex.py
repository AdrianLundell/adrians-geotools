"""
All data is stored in pandas dataframes, which out of the box comes with a number of data transformation possibilities. 
The full documentation can be found at  https://pandas.pydata.org/docs/ but a quick lookup is provided here for convinience.

The script is possible to run as a python live script for example in Visual Studio Code.
"""
#%% Import
import pandas as pd

#%% Dataframe creation
"""
Dataframes can be created and read in number of ways, the two ways used in the scripts are showcaedbelow but
for example reaidng csv and excel files are possible as well.
"""

#From fixed column width files
df = pd.read_fwf("data/icrf3_src_posn", skiprows = 23, header = None)
print(df)

#From dictionary 
data = {"A" : ["a","b","c"], "B" : [1,2,3], "C" : [4., 5., 6.]}
df = pd.DataFrame(data)
print(df)


#%% Dataframe structure
"""
The standard dataframe is a table with a label for each column and one for each row (called index). One column of a dataframe is called a serie.
"""

#Print columns and select by column
print(df.columns)
print(df.A)
print(df["A"])

#Print indexes and select by index
print("")
print(df.index)
print(df.loc[0])

#Datatypes, strings are objects
print()
print(df.dtypes)