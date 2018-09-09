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