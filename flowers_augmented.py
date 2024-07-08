import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# Import TensorFlow and Keras layers for CNN
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D


###########
# LOAD DATA
###########
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_dir = tf.keras.utils.get_file(origin=_URL, extract=True)
# Only difference in base_dir seems to be absence of the .tgz file extension
base_dir = os.path.join(os.path.dirname(zip_dir), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# Seperate train and validation folders
# New folder for each, each containing 5 folders (for each flower)
# Move images from original folders
# 5 empty folders, train, val
for flower in classes:
    img_path = os.path.join(base_dir, flower)
    # glob.glob returns a list of all pathnames matching a pathname PATTERN
    # * is a wildcard, so gather any pathname matching the pattern img_path\FILENAME.jpg
    images = glob.glob(img_path + '/*.jpg')

    # Allocate 80% of images to training
    num_train = int(round(len(images) * 0.8))
    train, val = images[:num_train], images[num_train:]

    for image in train:
        # Check if folder exists, if not, create it
        if not os.path.exists(os.path.join(base_dir, 'train', flower)):
            os.makedirs(os.path.join(base_dir, 'train', flower))
        # Move images from original flower folder to flower subfolder within training folder
        # (source path, destination path), remember image is just a file path
        shutil.move(image, os.path.join(base_dir, 'train', flower))

    for image in val:
        if not os.path.exists(os.path.join(base_dir, 'val', flower)):
            os.makedirs(os.path.join(base_dir, 'val', flower))
        shutil.move(image, os.path.join(base_dir, 'val', flower))

    # Save train and val paths for convenience
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

################
# AUGMENT IMAGES
################
BATCH_SIZE = 100
IMG_SHAPE = 150

image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=45, zoom_range=0.5)

# Specify method to apply image_gen
# class_mode determines the type of label arrays that are returned
aug_train_data = image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)

# Generally augmentation only applied to training samples
# Validation images still need to be rescaled (pixel values between 0 and 1)
# Shuffle not necessary
val_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
aug_val_data = val_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)

################
# CREATE THE CNN
################

# Note input shape is only specified for the first layer
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Dropout AFTER Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax')        
])

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
epochs = 80

# Not sure why this is always called history
# Because .fit returns a history object (see documentation)
history = model.fit_generator(
    aug_train_data,
    steps_per_epoch=int(np.ceil(aug_train_data.n / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=aug_val_data,
    validation_steps=int(np.ceil(aug_val_data.n / float(BATCH_SIZE)))
)

# PLOT RESULTS (This step isn't always necessary)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



