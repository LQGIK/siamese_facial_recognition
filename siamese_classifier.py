import cv2
from PIL import Image
import os
import sys

import tensorflow as tf
import numpy as np

IMG_WIDTH = 80
IMG_HEIGHT = 80

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

        # Get first image
        image = os.listdir(pathFolder)[0]
        
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

        labels.append(count)
        keys[count] = person
        count += 1

    return images, labels, keys


def main():

    # Initialize the classifier
    cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    images, labels, keys = load("Images")

    # Load model and error checking
    if len(sys.argv) != 2:
        print("Usage: python classifier.py <model_name.h5>")
        return

    model_name = sys.argv[1]
    model = tf.keras.models.load_model(model_name)
    if not model:
        print("This isn't an acceptable model!")
        return


    # Capture video footage
    video_capture = cv2.VideoCapture(0)


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert to greyscale cv2 image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        # Identify faces in image (gray), 
        faces = faceCascade.detectMultiScale (
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        








        # Draw a rectangle around the faces
        # Sets x,y,w,h to dimensions of face, and draws green rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Video", frame)




        
        if cv2.waitKey(1) & 0xFF == ord("i"):

            # Reshape and predict
            dim = (IMG_WIDTH, IMG_HEIGHT)
            small_frame = cv2.resize(gray, (0, 0), fx=0.046875, fy=0.0625)
            image = small_frame.reshape(dim)
            image = [np.array(image).reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)]

            # Make predictions
            predictions = []
            for image in images:
                predictions.append(model.predict(image).argmax())
            max_val = max(predictions)
            final = predictions.index(max_val)
            print(keys[final])
            
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break


    # Release capture frames
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()