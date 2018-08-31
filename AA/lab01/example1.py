import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from string import Template


# recebe um dataframe como argumento e percorre pela coluna indicada
# retornando o somatorio dos seus elementos
def sum(df,col1:str='col1',col2:str='col2',w0:float=0.1,w1:float=0.2,alpha:float=0.01):
        n = df.shape[0] # row number
        W1 = w1; W0 = w0; sum_ei = 0; sum_eii = 0

        # percorrendo o dataframe
        for index, row in df.iterrows():
            # print(row[col1],row[col2])            
            # row[col1] = xi
            yii = (w1 * row[col1]) + w0
            ei = row[col2] - yii
            sum_ei += ei
            sum_eii += (ei*row[col1])
        
        W0 = w0 + alpha * (sum_ei/n)
        W1 = w1 + alpha * (sum_eii)
        return (W0,W1)


def main():

    data = pd.read_csv('ex1data1.txt', sep=',',header=None)
    data.columns = ['col1','col2']

    # print(data.__doc__)



    # df = pd.DataFrame(data)

    # plt.plot(data['col1'],data['col2'],'c.')
    # plt.ylabel('column 2')
    # plt.xlabel('column 1')
    # plt.show()

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
    w0 = 1.1
    w1 = 0.1
    

    
    
    # atualmente sum retorna uma iteração, os primeiros w0 e w1 
    print( sum(data,'col1','col2',w0,w1,alpha) )

    # print(n)
    # print(data.iloc[0])

    # w0 = w0 + alfa*(1/n)



if __name__ == "__main__":
    main()