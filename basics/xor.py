import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# cargamos los 4 valores de la puerta xor
training_data = np.array([[0,0], [0,1], [1,0], [1,1]], "float32")

#y estos son los resultados que se obtienen, en el mismo orden
target_data = np.array([[0], [1], [1], [0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
 
model.fit(training_data, target_data, epochs=1000)
 
# evaluamos el modelo
scores = model.evaluate(training_data, target_data)
 
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict([[1,0], [0,0]]).round())