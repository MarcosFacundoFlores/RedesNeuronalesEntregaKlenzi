import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Datos de entrada y salida XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid'))  # Capa oculta con 4 neuronas
model.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar la red neuronal
model.fit(X, y, epochs=5000, verbose=0)

# Predecir la salida para los datos de entrada
predictions = model.predict(X)

# Imprimir las predicciones
print(predictions)
