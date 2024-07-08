# TRANSFER LEARNING WITH TENSORFLOW HUB

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image 

CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
])

print(model.summary())

# ImageNet dataset has been trained on 1000 different output classes. 
# Test on a single image containing a military uniform (one of the classes)
img = tf.keras.utils.get_file('Grace_Hopper.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')

# Resize image to match size of images model was trained on
img = Image.open(img).resize((IMAGE_SHAPE, IMAGE_SHAPE))

# Rescale pixel values so they range between 0 and 1
img = np.array(img)/255.0

# MODELS ALWAYS WANT TO PROCESS A BATCH
# ADD A BATCH DIMENSION
result = model.predict(img[np.newaxis, ...])

predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

# CLASS 653. Need ImageNet labels to know which class this is
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

# SYNTAX FOR OPENING TXT FILE AS LIST
labels = np.array(open(labels_path).read().splitlines())
print(labels[653])

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



