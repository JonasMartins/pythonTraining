
""" 
source: https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from string import Template

from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library 

# Print the dataset, only works because it's a sklearn's dataset
# print(data.DESCR)

# print the column names of the independent variables
# print(data.feature_names)
# dependent variables
# print(data.target)

# define the data/predictors as the pre-set feature names  
# df = independent variables
df = pd.DataFrame(data.data, columns=data.feature_names)

"""
source: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
Dataframe example:
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df
col1  col2
0     1     3
1     2     4
"""


# Put the target (housing value -- MEDV) in another DataFrame
# target = dependent variables
target = pd.DataFrame(data.target, columns=["MEDV"])

templ = Template("This is our data predictors:\n ${predictors}")
output = templ.substitute(predictors=target)

# print(output)

X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())
