import tensorflow as tf
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
import numpy as np

# tfds returns a dict like object with two keys 'test' and 'train'
# with_info=True will also return a second value with info about the loaded dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Currently the labels are just integers 0 - 9
# We can use the metadata to assign their class name (trouser, dress, etc)
class_names = metadata.features['label'].names
print(f"Class names: {class_names}")

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# Pixel values range from 0 to 255, for the model to work these values need to be normalised to the range 0 to 1
# tf.cast casts a tensor (matrix) to a new type
def normalise(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# map is used when you need to apply a TRANSFORMATION FUNCTION to each item in an iterable
train_dataset = train_dataset.map(normalise)
test_dataset = test_dataset.map(normalise)

# The first time you use the dataset, the images will be loaded from disk
# Caching keeps them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

'''
# Check images are correct prior to testing

plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(train_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
'''

# Configure model layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# This line is what performs the training
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# What is the accuracy when using the test set?
# It is normal for this to be lower than the accuracy derived from the training set (after all, this is new data)
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print(f"Accuracy on test dataset: {test_accuracy}")

# Making predictions
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

    print(np.argmax(predictions[0]))
    print(test_labels[0])