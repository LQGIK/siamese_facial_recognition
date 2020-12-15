import os
import sys
from random import randint

# Machine Learning Libs
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Global vars
EPOCHS = 50
IMG_WIDTH = 80
IMG_HEIGHT = 80
TEST_SIZE = 0.4

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
        

        # Retrieve path to folder
        pathFolder = os.path.join(dir, person)

        # Iterate every image in the directory
        for image in os.listdir(pathFolder):

            
            # Retrieve path to image
            img_path = os.path.join(pathFolder, image)

            # Pull the image and convert to grayscale
            img0 = Image.open(img_path)
            img1 = img0.convert(mode='L')

            # Resize to IMG_HEIGHT and IMG_WIDTH
            dim = (IMG_HEIGHT, IMG_WIDTH)
            img = img1.resize(dim)

            # Append image to images if correctly resized
            img = np.asarray(img)
            img = np.reshape(img, (IMG_WIDTH, IMG_HEIGHT, 1))
            if img.shape == (IMG_WIDTH, IMG_HEIGHT, 1):
                images.append(img)

                # Increment count
                labels.append(count)
        keys[count] = person
        count += 1
    
    return (images, labels, keys)
def getMatches(match_label, labels, images):
    """
    Returns range of matching indexes for a given image. Assuming all images are processed in order
    """

    matches = []
    for i in range(len(labels)):
        if match_label == labels[i]:
            matches.append(i)
    return (min(matches), max(matches))

def pair(batch_size, images, labels):
    """
    Organize randomized pairs for similarity comparison
    """

    num = len(labels)
    half = int(0.5 * batch_size)
    batch = []
    match = []

    # Organize match pairs for one half, and mismatched for another
    for i in range(batch_size):
        
        # Pick random image
        index1 = randint(0, num - 1)
        x1 = images[index1]
        label = labels[index1]
        low, high = getMatches(label, labels, images)

        if i < half:
            # From given label, find range of indexes of images with matching labels, and choose randomly from those
            x2 = images[randint(low, high)]
            batch.append([x1, x2])
            match.append(1)


        # Create mismatches
        else:

            # Randomly choose to pick from upper bound or lower bound
            bounds = randint(0, 1)
            if bounds == 0:
                x2 = images[randint(0, low)]
            else:
                x2 = images[randint(high, num - 1)]

            batch.append([x1, x2])
            match.append(0)
    
    return batch, match
def preprocess(item):
    x1_list =[]
    x2_list = []
    for pair in item:
        x1_list.append(pair[0])
        x2_list.append(pair[1])
    return [x1_list, x2_list]
    


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

    return siamese_net







def main():

    # Get image arrays and labels for all image files. Then make batches of mixed or matched pairs
    images, labels, keys = load("Images")
    batch, match = pair(200, images, labels)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(match)
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(batch), np.asarray(match), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model(x_train.shape[2:])

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    # Fit model on training data (16, 2, 200, 200, 1)
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")


if __name__ == "__main__":
    main()