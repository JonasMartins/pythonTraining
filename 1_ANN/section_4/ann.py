# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
"""
  Tentar prever a saidas das variaveis dependentes, a partir das variaveis independentes
"""

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Pegando as colunas a partir de 3 ate 12, de CreditScore a EstimatedSalary 
X = dataset.iloc[:,3:13].values 
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# transformando os indices dos paises em valores numericos
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Transformando os indices de genero em valores numericos
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])




# criando dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Exluindo a primeira coluna de dummy variables resultando apenas em 2 colunas de dummy variables
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling: https://en.wikipedia.org/wiki/Feature_scaling
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 Now make an ann

# import keres
import keras

# Library to initialize the neural network
from keras.models import Sequential

# Library to create the layers on the artificial neural etwork
from keras.layers import Dense


# Initializing the ANN
classifier = Sequential()







#print(X)
#print(X_train)
#print(X_test)


"""

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""
