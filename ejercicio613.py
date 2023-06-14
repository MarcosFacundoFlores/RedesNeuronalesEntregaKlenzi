from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

# Definir la arquitectura de la red neuronal
model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))
# Establecer los pesos
pesos_capa = np.array([[0.5], [0.4], [0.8]])
pesos_bias = np.array([0])
model.layers[0].set_weights([pesos_capa, pesos_bias])


# Compilar el modelo con tasa de aprendizaje de 0.3
learning_rate = 0.3
sgd = SGD(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer=sgd)

# Visualizar la arquitectura del model

# Datos de entrada
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# Salida esperada
y = np.array([[0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7]])

# Entrenar la red neuronal
model.fit(X, y, epochs=5, batch_size=1, verbose=0)

# Decodificar números binarios
resultados = model.predict(X)

# Imprimir resultados
for i in range(len(X)):
    print(f"Número binario {X[i]} decodificado a decimal: {resultados[i][0]}")
