from sklearn.datasets import fetch_california_housing 
from sklearn.datasets import load_boston

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import matplotlib as mpl

boston = load_boston()
california = fetch_california_housing()

dataset = pd.DataFrame(boston.data,columns=boston.feature_names)
dataset['target'] = boston.target

# probability density function

# media
# mean = dataset['target'].mean()

# via nympy:
# np.mean(dataset['target'])
# square_errors = pd.Series(mean - dataset['target'])**2

# correlação: o quanto duas variáveis estão
# relacionadas uma com a outra

def covariance(v1,v2,bias=0):
    observations = float(len(v1))
    return np.sum((v1 - np.mean(v1))*(v2 - np.mean(v2)))/(observations-min(bias,1))

# standardizing a variable
def standardize(x):
    return (x-np.mean(x)/np.std(x))


def correlation(v1,v2,bias=0):
    return covariance(standardize(v1),standardize(v2),bias)

"""

from scipy.stats.stats import pearsonr

print('Correlation estimate: %0.5f' %(correlation(dataset['RM'],dataset['target'])))


print('Correlation from pearsonr estimate: %0.5f' %(pearsonr(dataset['RM'],dataset['target']))[0])

"""

#print(dataset['RM'])
#print(dataset['target'])

x_range = [dataset['RM'].min(),dataset['RM'].max()]
y_range = [dataset['target'].min(),dataset['target'].max()]
scatter_plot = dataset.plot(kind='scatter',x='RM',y='target',xlim=x_range,ylim=y_range)
meanY = scatter_plot.plot(x_range,[dataset['target'].mean(),dataset['target'].mean()],'--',color='red',linewidth=1)

meanX = scatter_plot.plot([dataset['RM'].mean(),dataset['RM'].mean()], y_range ,'--',color='red',linewidth=1)
plt.show() # sempre lembrar de colocar essa linha