import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


my_data = pd.read_csv('example/home.txt',names=["size","bedroom","price"])

#we need to normalize the features using mean normalization
my_data = (my_data - my_data.mean())/my_data.std()

# print(my_data.head())

# ! Da linha 0 até a ultima, pegando da coluna 0 até a 2
# ?.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
X = my_data.iloc[:,0:2].values

ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to

# * Cria uma matriz de uma linha e 3 colunas com apenas zeros
theta = np.zeros([1,3])
# print( ('theta0 = %s theta1 = %s') %(theta0, theta1) )

# print('theta transposto shape:',theta.T.shape)
# print('X shape:',X.shape)
# print('y shape:',y.shape)
# print('X @ theta transposto shape:',(X @ theta.T).shape)

#computecost
# ! theta.T = theta transpost
# * @ é o produto entre matrizes
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost


#set hyper parameters
alpha = 0.01
iters = 1000

# print(X[:10,1:2])
fig,ax = plt.subplots()

ax.set_xlabel('x?')
ax.set_ylabel('y?')
ax.set_title('Exercicio 2')
#ax.set_xticks(index + bar_width / 2)
#ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))

fig.tight_layout()

custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='g', lw=4),
                Line2D([0], [0], color='b', lw=4)]


ax.plot(X[:,1:2],'.r',X[:,2:3],'.g',y,'.b')

ax.legend(custom_lines, ['Área', 'Quartos', 'Preço'])

plt.show()

#running the gd and cost function
# g,cost = gradientDescent(X,y,theta,iters,alpha)
# print(g)

# finalCost = computeCost(X,y,g)
# print(finalCost)

# plt.plot(np.arange(iters), cost, 'r')  
# plt.xlabel('Iterations')  
# plt.ylabel('Cost')  
# plt.title('Error vs. Training Epoch')  
# plt.show()