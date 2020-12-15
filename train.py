import os
import sys
from random import randint

# Machine Learning Libs
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load(dir):
    """
    Loads all images from a specified directory
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

        # Store 0-n with corresponding person name
        keys[count] = person
        count += 1
    
    return (images, labels, keys)

def get_model(input_shape):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Convolutional Neural Network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (10, 10), activation="relu", input_shape=(input_shape)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"),
        tf.keras.layers.Conv2D(64, (10, 10), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"),
        tf.keras.layers.Conv2D(128, (7,7), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding="same"),
        tf.keras.layers.Conv2D(200, (4,4), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation="sigmoid")
    ])
    


    # Define the tensors (vectors) for the two input images as input_x1 & x2. Next we want to pull all the feature vectors by 
    # putting these tensors through the CNN
    input_x1 = tf.keras.Input(input_shape)
    input_x2 = tf.keras.Input(input_shape)
    encoded_l = model(input_x1)
    encoded_r = model(input_x2)
    
    # Feed feature vectors into energy function. This is a custom function that calculates the absolute difference between either vector.
    # The Lambda (or custom) layer is a neat feature that allows us to do our own math operations. In this case we just subtract the vectors!
    # The first part is making the layer, and the second part actually executes it
    L1_layer = tf.keras.layers.Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)
    
    # Make a single model, comprised of taking in two inputs, and spitting out one output
    siamese_net = tf.keras.Model(
        inputs = [input_x1, input_x2],
        outputs=prediction
    )
    
    # return the model
    siamese_net.summary()

    # Compile the model with weight optimization, loss algorithms and emphasize accuracy
    siamese_net.compile(
        optimizer="RMSprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    #tf.keras.losses.

    return siamese_ne


def generateBatch(n, images, labels, dict_keys):
    n_per_person = n / 2
    for person in people:
        
        person_index = dict_keys[person]
        person_images = narrowToPerson(person_index, dict_keys, labels) # Narrow down for ease

        matching_pairs = createPersonMatches(n_per_person, person_images)
        mismatching_pairs = createPersonMismatches(n_per_person, person_images, images, dict_keys)

        return matching_pairs, mismatching_pairs




def createPersonMismatches(n_per_person, person_images, images, dict_keys):
    '''
    n_per_person {int pairs to be made}
    person_images {sliced list to only include imgs from person}
    images {list of all images}
    dict_keys {person : 0 - 6}
    Returns hsape of (n, 2) {list of mismatching pairs}
    '''
    person_pairs = []
    for n in range(n_per_person):
        random_person_img_index = randint(0, len(person_images))
        random_person_img = person_images[random_person_img_index]

        other_person_index = getRandomIntExcluding(0, max(keys), keys[person])
        other_person_imgs = narrowToPerson(other_person_index, dict_keys, labels)
        random_other_img_index = randint(0, len(other_person_imgs))
        random_other_person_img = random_other_person_imgs[random_other_img_index] 

        person_pairs.append([random_person_img, random_other_person_img])
    return person_pairs


def createPersonMatches(n_per_person, person_images):
    '''
    Returns shape of (n, 2) {list of matching pairs}
    '''
    person_pairs = []
    for n in range(n_per_person):
        random_img_index1 = randint(0, len(person_images))
        random_img_index2 = randint(0, len(person_images))
        random_img1 = person_images[random_img_index1]
        random_img2 = person_images[random_img_index2]
        person_pairs.append([random_img1, random_img2])
    return person_pairs

def narrowToPerson(person_index, dict_keys, labels):
        # Narrow down imgs to search through by slicing
        # Then finally create matching pairs for each person
        '''
        person_index {int of key of person}
        dict_keys {person : 0 - 6}
        labels {0 - 6}
        returns sliced list of images pertainin to a single person
        '''
        
        startIndex = labels.index(person_index)
        endIndex =  lastItem(labels, person_index)
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

def main():

    # Global vars
    EPOCHS = 50
    IMG_WIDTH = 100
    IMG_HEIGHT = 100
    TEST_SIZE = 0.4


    # Get image arrays and labels for all image files. Then make batches of mixed or matched pairs
    images, labels, keys = load(r"C:\users\xlqgi\dev\ai\deep\facerecog\siamese\images\proc")
    matching_pairs, mismatching_pairs = generateBatch(10, images, labels)



def __name__ == "__main__":
    main()