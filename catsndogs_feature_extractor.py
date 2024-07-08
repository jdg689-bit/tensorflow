# TRANSFER LEARNING WITH TENSORFLOW HUB

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image 

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
IMAGE_SHAPE = 224

#########################################
# TRANSFERRING THE MODEL TO CATS AND DOGS
#########################################
# Recall dataset is a dict containing 'test' and 'train'
# Use split to assign 20% of 'train' for validation purposes
(train_examples, validation_examples), metadata = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]'])

examples_count = metadata.splits['train'].num_examples
class_count = metadata.features['label'].num_classes

# Reformat all images to be 224 x 224 pixels
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_SHAPE, IMAGE_SHAPE))/255.0
    return image, label

train_dataset = train_examples.shuffle(examples_count//4).map(format_image).batch(32).prefetch(1)
validation_dataset = validation_examples.map(format_image).batch(32).prefetch(1)

#################################
# MODIFYING THE FEATURE EXTRACTOR
#################################


feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))

# Freeze variables in the feature extractor layer
feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

EPOCHS = 6

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset
                    )







