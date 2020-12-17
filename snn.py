import os
import sys
from random import randint

# Machine Learning Libs
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import initializers
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


def load(dir):
    """
    Loads all images from a specified directory
    Returns
        images {list of images}
        labels {list of labels 0-6}
        dict_keys {name : 0 - 6}
    """
    # Initalize return arrays
    images = []
    labels = []
    keys = {}

    # Iterate Dog and Cat directory
    count = 0
    for person in os.listdir(dir):
        
        # Find and read in every img for a person
        person_path = os.path.join(dir, person)
        for image in os.listdir(person_path):

            img_path = os.path.join(person_path, image)
            img = cv2.imread(img_path, 0)
            images.append(img)
            labels.append(count)

        # Store 0-n with corresponding person name
        keys[person] = count 
        count += 1
    
    return (images, labels, keys)

def getLayer(filters, kernel_size):
    return layers.Conv2D(
        filters=        filters, 
        kernel_size=    kernel_size, 
        activation=     "relu", 
    )

def getModel(input_shape):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Convolutional Neural Network
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(8, 8), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(getLayer(64, (4, 4)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(getLayer(128, (4, 4)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(getLayer(256, (4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation="relu"))
    


    # Define the tensors (vectors) for the two input images as input_x1 & x2. Next we want to pull all the feature vectors by 
    # putting these tensors through the CNN
    input_x1 = Input(input_shape)
    input_x2 = Input(input_shape)
    encoded_l = model(input_x1)
    encoded_r = model(input_x2)
    
    # Feed feature vectors into energy function. This is a custom function that calculates the absolute difference between either vector.
    # The Lambda (or custom) layer is a neat feature that allows us to do our own math operations. In this case we just subtract the vectors!
    # The first part is making the layer, and the second part actually executes it
    L1_layer = layers.Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = layers.Dense(1, activation='sigmoid')(L1_distance)
    
    # Make a single model, comprised of taking in two inputs, and spitting out one output
    siamese_net = tf.keras.Model(
        inputs = [input_x1, input_x2],
        outputs=prediction
    )
    
    # return the model
    siamese_net.summary()

    # Compile the model with weight optimization, loss algorithms and emphasize accuracy
    siamese_net.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return siamese_net


def getBatch(n, images, labels):

    # Convert to trains
    x = np.asarray(images)
    y = np.asarray(labels)

    # Reshape x from (27, 100, 100) to (27, 100, 100, 1)
    samples, w, h = x.shape
    x = np.reshape(images, (samples, w, h, 1))


    print()
    

def main():


    # Global vars
    n = 10
    EPOCHS = 50
    IMG_WIDTH = 100
    IMG_HEIGHT = 100
    TEST_SIZE = 0.4


    # Get image arrays and labels for all image files. Then make batches of mixed or matched pairs with corresponding labels
    images, labels, dict_keys = load(r"images/proc")
    getBatch(n, images, labels)


if __name__ == '__main__':
    main()