import numpy as np
import tensorflow as tf

# Set up data
celcius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100],  dtype=float)

# Enumerate adds counter to object, used here to index Fahrenheit
for i, c in enumerate(celcius_q):
    print(f"{c} degrees Celcius = {fahrenheit_a[i]} degrees Farenheit")

# Build model
# Single Dense layer
# input_shape = [1] = one dimensional array with one member
# units = number of neurons in the layer

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Once layers are defined they must be assembled
# Sequential() takes a list of layers as argument in calculation order from INPUT to OUTPUT
model = tf.keras.Sequential([l0])

# Compile the model
# Requires a LOSS FUNCTION and an OPTIMISER FUNCTION
# The 0.1 that follows Adam is the learning rate
# During training the optimiser function calculates adjustments to the model's internal variables
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model by calling the fit method
# First argument is INPUTS, second is OUTPUTS
history = model.fit(celcius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

# Use the model with model.predict()
print(model.predict([100.0]))

# You can print the layer weights
print(f"These are the layer variables: {format(l0.get_weights())}")