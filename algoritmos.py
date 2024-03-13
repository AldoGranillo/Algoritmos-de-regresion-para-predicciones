import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#ALMACENANDO LOS DATOS EN UN DATA FRAME 
data= pd.read_csv("01_Defunciones_1950_2070.csv")
#print(data)#Imprimiendo data

#Limpiando el data set
data_cleaned = data.dropna()  # Eliminar filas con datos faltantes
data_cleaned['SEXO'] = data_cleaned['SEXO'].map({'Hombres': 0, 'Mujeres': 1}) 
#Dividiendo el conjunto de datos 
x= data_cleaned[["AÑO","SEXO","EDAD"]].values
y=data_cleaned["DEFUNCIONES"].values
print(f"Valores de x : {x}")
print(f"Valores de y : {y}")
X_train, X_test,y_train, y_test= train_test_split(x, y, test_size=0.3 , random_state=42)

# Entrenamiento de los modelos
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# Evaluación de los modelos
linear_regression_predictions = linear_regression_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)

linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
knn_mse = mean_squared_error(y_test, knn_predictions)

linear_regression_mae = mean_absolute_error(y_test, linear_regression_predictions)
random_forest_mae = mean_absolute_error(y_test, random_forest_predictions)
knn_mae = mean_absolute_error(y_test, knn_predictions)

linear_regression_r2 = r2_score(y_test, linear_regression_predictions)
random_forest_r2 = r2_score(y_test, random_forest_predictions)
knn_r2 = r2_score(y_test, knn_predictions)

# Predicción de los datos de 
X_1951 = np.array([[2019, 0, i] for i in range(110)])
y_1951_linear_regression = linear_regression_model.predict(X_1951)
y_1951_random_forest = random_forest_model.predict(X_1951)
y_1951_knn = knn_model.predict(X_1951)

# Visualización de los resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(X_1951[:, 2], y_1951_linear_regression, label='Linear Regression')
plt.plot(X_1951[:, 2], y_1951_random_forest, label='Random Forest')
plt.plot(X_1951[:, 2], y_1951_knn, label='KNN')
plt.xlabel('Edad')
plt.ylabel('Defunciones')
plt.title('Predicción de Defunciones en 2019')
plt.legend()
plt.show()