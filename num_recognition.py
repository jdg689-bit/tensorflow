import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import cv2

# LOAD DATA
dataset = tfds.load('mnist', as_supervised=True)    

train_data = dataset['train']
test_data = dataset['test']

num_train_examples=60000
num_test_examples=10000


# Normalise
# Normalising the data took accuracy from %10 to 57% (with 5 epochs)
def normalise(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_data = train_data.map(normalise)
test_data = test_data.map(normalise)

# Build model
model = tf.keras.Sequential([
    # INPUT LAYER (1D vector with 784 elements)
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # HIDDEN LAYER
    # No. of neurons in hidden layer between size of input (784) and size of output(10) layers
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    # OUTPUT LAYER
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    # Try changing this to tf.keras.metrics.Accuracy
    metrics=['accuracy'])

# Train model
BATCH_SIZE = 32
train_data = train_data.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

model.fit(train_data, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# CHECK MODEL WORKS USING TEST DATA
# Use model.evaluate
test_loss, test_accuracy = model.evaluate(test_data, steps=math.ceil(num_test_examples/BATCH_SIZE))

####################
# MAKING PREDICTIONS
####################

# Testing returns 97.62% accuracy. Time to make predictions
# take() is batch count, not example count
'''
for test_images, test_labels in test_data.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    for i in range(BATCH_SIZE):
        cv2.imshow('Image', test_images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows

        print(np.argmax(predictions[i]))
        # print(test_labels[0])
'''

#####################
# TESTING ON NEW DATA
#####################

my_img = cv2.imread('C:/Users/Jacob de Graaf/Documents/python/tensorflow/assets/seven.png', cv2.IMREAD_GRAYSCALE)
my_img = cv2.resize(my_img, (28, 28))
# my_img = my_img.astype('float32') / 255

# I don't know what these lines do, but it doesn't work without them. Something to do with reshaping to (1,28,28,1) -> (batch_size, height, width, channels)

my_img = np.expand_dims(my_img, axis=0)
my_img = np.expand_dims(my_img, axis=-1)


prediction = model.predict(my_img)

print(np.argmax(prediction))

