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


def generateBatch(n, images, labels, dict_keys):
    n_per_person = int(n / 2)
    people = dict_keys.keys()
    for person in people:
        
        person_index = dict_keys[person]
        person_images = narrowToPerson(person_index, dict_keys, images, labels) # Narrow down for ease

        matching_pairs = createPersonMatches(n_per_person, person_images)
        mismatching_pairs = createPersonMismatches(n_per_person, person, person_images, images, labels, dict_keys)

        return matching_pairs, mismatching_pairs




def createPersonMismatches(n_per_person, person, person_images, images, labels, dict_keys):
    '''
    n_per_person {int pairs to be made}
    person_images {sliced list to only include imgs from person}
    images {list of all images}
    dict_keys {person : 0 - 6}
    Returns hsape of (n, 2) {list of mismatching pairs}
    '''
    person_pairs = []
    for n in range(n_per_person):
        random_person_img_index = randint(0, len(person_images) - 1)
        random_person_img = person_images[random_person_img_index]

        other_person_index = getRandomIntExcluding(0, max(dict_keys.values()), dict_keys[person])
        other_person_imgs = narrowToPerson(other_person_index, dict_keys, images, labels)
        random_other_img_index = randint(0, len(other_person_imgs) - 1)
        random_other_person_img = other_person_imgs[random_other_img_index] 

        # Reshaping (100, 100) to (100, 100, 1)
        random_person_img = np.reshape(random_person_img, (100, 100, 1))
        random_other_person_img = np.reshape(random_person_img, (100, 100, 1))
        person_pairs.append([random_person_img, random_other_person_img])
    return person_pairs


def createPersonMatches(n_per_person, person_images):
    '''
    Returns shape of (n, 2) {list of matching pairs}
    '''
    person_pairs = []
    for n in range(n_per_person):
        random_img_index1 = randint(0, len(person_images) - 1)
        random_img_index2 = randint(0, len(person_images) - 1)
        random_img1 = person_images[random_img_index1]
        random_img2 = person_images[random_img_index2]

        # Reshaping (100, 100) to (100, 100, 1)
        random_img1 = np.reshape(random_img1, (100, 100, 1))
        random_img2 = np.reshape(random_img2, (100, 100, 1))
        person_pairs.append([random_img1, random_img2])
    return person_pairs

def narrowToPerson(person_index, dict_keys, images, labels):
        # Narrow down imgs to search through by slicing
        # Then finally create matching pairs for each person
        '''
        person_index {int of key of person}
        dict_keys {person : 0 - 6}
        labels {0 - 6}
        returns sliced list of images pertainin to a single person
        '''
        
        startIndex = labels.index(person_index)
        endIndex =  lastItem(labels, person_index) + 1
        person_images = images[startIndex:endIndex]
        return person_images

def getRandomIntExcluding(min, max, excluding_num):
    '''
    Return number within min and max, that is not equal to excluding_num
    '''
    rand = randint(0, max)
    while (rand == excluding_num):
        rand = randint(0, max)
    return rand

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def lastItem(list1, item):
    return len(list1) - 1 - list1[::-1].index(item)


def getBatch(n, images, labels):

    num_people, = np.asarray(images)
    np.zeros((n, ))

def train(model, x_train, y_train):

    x_shape = x_train.shape
    pairs = np.zeros(())


    # Train model on each pair
    for i in range(x_train.shape[0]):
        # do stuff
        pair = x_train[i]
        label = y_train[i]
        img0, img1 = pair

        probs = model.predict(pair)
        print("Prediction: " + str(np.argmax(probs)) )
        print("Label: " + str(np.argmax(label)) + "\n")

    return 0


def main():

    # Global vars
    n = 10
    EPOCHS = 50
    IMG_WIDTH = 100
    IMG_HEIGHT = 100
    TEST_SIZE = 0.4


    # Get image arrays and labels for all image files. Then make batches of mixed or matched pairs with corresponding labels
    images, labels, dict_keys = load(r"images/proc")
    matching_pairs, mismatching_pairs = generateBatch(n, images, labels, dict_keys)
    pairs = np.asarray(matching_pairs + mismatching_pairs)
    labels = np.zeros((n,))
    labels[n//2:] = 0

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(pairs), labels, test_size=TEST_SIZE
    )
    # x_train.shape = (6, 2, 100, 100) means 6 people, 2 images, each image is 100 rows, each row has 100 pixels
    # y_train.shape = (6, 2) means 6 people, 2
    x_train_shape = (100, 100, 1)
    getBatch(n, images, labels)

    model = getModel(x_train_shape)

    train(model, pairs, labels)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")



if __name__ == "__main__":
    main()