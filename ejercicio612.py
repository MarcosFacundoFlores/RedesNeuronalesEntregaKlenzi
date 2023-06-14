# Se importan las bibliotecas necesarias: numpy para trabajar con matrices y arreglos, 
# las clases Sequential y Dense de Keras para crear y configurar el modelo de la red neuronal.

#pip install numpy
#pip install keras
#pip install tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Se definen los datos de entrada (X) y salida esperada (y) para la función XOR. 
# En este caso, X contiene las cuatro posibles combinaciones binarias de entrada 
# y contiene las salidas esperadas correspondientes.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Se crea una instancia del modelo de red neuronal utilizando la clase Sequential de Keras. 
# Luego se agregan capas al modelo utilizando el método add(). 
# En este caso, se agrega una capa oculta con cuatro neuronas y una capa de salida con una neurona. 
# La función de activación utilizada es la sigmoide (sigmoid), que permite obtener salidas en el rango de 0 a 1.
model = Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid'))  # Capa oculta con 4 neuronas y 2 input de entrada
model.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona

# Se compila el modelo utilizando el método compile(). 
# Se especifica la función de pérdida (binary_crossentropy) para problemas de clasificación binaria. 
# El optimizador utilizado es adam, que es un algoritmo popular de optimización. 
# También se define la métrica de evaluación de accuracy para medir el rendimiento del modelo.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Se entrena el modelo utilizando el método fit(). Los datos de entrada (X) y salida esperada (y) se pasan como argumentos. 
# El parámetro epochs indica el número de épocas de entrenamiento, y en este caso se establece en 5000. 
# El parámetro verbose se establece en 0 para no mostrar información detallada durante el entrenamiento.
model.fit(X, y, epochs=1000, verbose=0)


# Predecir la salida para los datos de entrada
predictions = model.predict(X)

# Imprimir las predicciones
print(predictions)

#Para correr el algoritmo: python ejercicio612.py
