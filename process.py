from PIL import Image, ImageOps
import numpy as np
import os

def crop(img, dim):
    new_img = [] 
    for row in img[:dim]:
        new_img.append(row[:dim])
    return np.asarray(new_img)

def refit(path, dest_path):
    count = 0
    for filename in os.listdir(path):

        # Get paths
        filepath = os.path.join(path, filename)
        dest_filepath = os.path.join(dest_path, "proc" + str(count) + ".jpg")
        print("Opening " + filepath + ", Saving in " + dest_filepath)

        # Open File and Crop
        shape = (100, 100)
        img = Image.open(filepath)
        new_img = ImageOps.fit(img, shape, method=Image.NEAREST, bleed=0.5)

        # Save files and overwrite
        if os.path.isfile(dest_filepath):
            os.remove(filepath)
            os.remove(dest_filepath)
        new_img.save(dest_filepath)
        count += 1


def main():
    src_path = r"C:\Users\xlqgi\DEV\AI\DEEP\FaceRecog\siamese\images\unproc"
    dest_path = r"C:\Users\xlqgi\DEV\AI\DEEP\FaceRecog\siamese\images\proc"

    for folder_name in os.listdir(src_path):
        folder_path = os.path.join(src_path, folder_name)
        dest_folder_path = os.path.join(dest_path, folder_name)

        if not os.path.isdir(dest_folder_path):
            os.mkdir(dest_folder_path)
        refit(folder_path, dest_folder_path)
            


if __name__ == '__main__':
    main()