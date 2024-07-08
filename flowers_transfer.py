import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import numpy as np
from matplotlib import pyplot as plt

import cv2

# DOWNLOAD FLOWERS DATASET
(train_ds, val_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True
)

print(metadata)

classes = metadata.features['label'].names

trng_example_count = 0
val_example_count = 0

for example in train_ds:
    trng_example_count += 1
for example in val_ds:
    val_example_count += 1

# Display test image with matplotlib -> syntax unclear
'''
image, label = next(iter(train_ds))

plt.imshow(image)
plt.title(classes[label])
plt.show()
'''
######################################################

# REFORMAT IMAGES FOR MOBILENET (224, 224)
def format_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image, label

train_ds = train_ds.map(format_image)
val_ds = val_ds.map(format_image)

# CREATE BATCHES -> Requirement for model training?
BATCH_SIZE = 32

train_ds = train_ds.shuffle(trng_example_count).batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# TRANSFER LEARNING
# Create FEATURE EXTRACTOR
IMAGE_SHAPE = 224
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))

# Freeze pre-trained model
feature_extractor.trainable = False

# Attach classification head
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_ds,
          epochs=1,
          validation_data=val_ds)

# MAKE PREDICTIONS
# Create Image batch and corresponding label batch
image_batch, label_batch = next(iter(train_ds))

image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = classes[predicted_ids]

print(predicted_class_names)

# Convert both to numpy arrays 

# Use .predict() method

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
