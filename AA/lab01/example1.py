import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from string import Template


data = pd.read_csv('ex1data1.txt', sep=',',header=None)
data.columns = ['col1','col2']

# print(data.__doc__)



# df = pd.DataFrame(data)

plt.plot(data['col1'],data['col2'],'c.')
plt.ylabel('column 2')
plt.xlabel('column 1')
plt.show()

