import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from string import Template


# * Método de regressão linear Univariada
 
def linear_regression_univariate(df,col1:str='col1',col2:str='col2',alpha:float=0.01,times:int=5):
    n = df.shape[0] # row number
    # Loop through times variable, it had default set as 5 
    w0 = np.random.random(df.shape[1])
    w1 = np.random.random(df.shape[1])
    ww1 = w1; ww0 = w0; sum_ei = 0; sum_eii = 0


    for i in range(times):
        # percorrendo o dataframe
        for index, row in df.iterrows():
            # print(row[col1],row[col2])            
            # row[col1] = xi
            yii = (w1 * row[col1]) + w0
            ei = row[col2] - yii
            #sum_ei += ei # normal
            #sum_eii += (ei*row[col1]) # normal
        
        # ww0 = w0 + alpha * (sum_ei/n) # normal
        # ww1 = w1 + alpha * (sum_eii/n) # normal
        
        ww0 = w0 + alpha * ei
        ww1 = w0

        # print("Run {0}: w0: {1} w1: {2}".format(i,w0,w1)) logs
        w0 = ww0
        w1 = ww1
        # sum_ei=sum_eii=0 # normal

        # updating weights    
        # w0 = w0 + (alpha * ei)
        # w1 = w1 + (alpha * (ei*row[col1]))

        # print("W0 {0}: W1: {1}".format(w0,w1)) logs
    return (ww0,ww1)


def main():

    data = pd.read_csv('ex1data1.txt', sep=',',header=None)
    data.columns = ['col1','col2']

    # print(data.__doc__)

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

    alpha = 0.001
    
    
    # atualmente sum retorna uma iteração, os primeiros w0 e w1 
    # print('Final Weights, w0,w1: {0}'.format(linear_regression_univariate(data,'col1','col2',w0,w1,alpha)))

    # y_barra = w1xi + w0 => ( (ww1*data['col1']) + ww0)
    # times = epocas
    times = 1000;
    #(ww0,ww1) = linear_regression_univariate(data,'col1','col2',alpha,times)
    
    w0 = np.random.random(data.shape[1])
    w1 = np.random.random(data.shape[1])

    print(w0,w1)

    # plt.plot(data['col1'],data['col2'],'c.', data['col1'], ( (ww1*data['col1']) + ww0)  )
    # plt.ylabel('column 2')
    # plt.xlabel('column 1')
    # plt.show()


if __name__ == "__main__":
    main()