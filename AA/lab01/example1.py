import numpy as np
import pandas as pd
from string import Template


data = pd.read_csv('ex1data1.txt', sep=',',header=None)
data.columns = ['col1','col2']

#templ = Template("This is our data predictors:\n ${predictors}")
#output = templ.substitute(predictors=data)

print(data)