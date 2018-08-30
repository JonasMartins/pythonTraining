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

# formula:
# O objetivo é encontrar o w0 e w1 da reta que serviria 
# como base para representar a distribuição dos pontos,
# yi = w1xi + w0, onde yi é o y barra que é o valor
# a ser achado e xi o valor de col1, temos que encontrar
# os pesos mais adequados para essa reta melhor representar
# o alinhamento desses pontos.
# 
# Após o desenvolvimento da formula 
# j(w1,w0) = 1/(2*n) somatório de 1 até n (yi-wi * xi - w0)^2
# iniciar os pontos w0 e w1 de forma aleatória.