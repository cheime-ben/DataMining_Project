import os
from PIL import Image
import numpy as np
from tensorflow.keras import datasets

res_folder = "."+os.sep+"preprocessed"


def open_image(src: str):
    '''
    load an image as an numpy array
    '''
    img = Image.open(src)
    image_array = np.asarray(img)
    return image_array
def load_ardis_data(src):
    '''
    iterate the foler where the preprocessed images where stored and import them and split them into arrays
    '''
    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    for root, dirs, files in os.walk(src):
        if len(root.split(os.sep)) == 3:
            current_number = root.split(os.sep)[2]
            test_labels += [int(current_number) for _ in range(100)]
            train_labels += [int(current_number) for _ in range(660)]
            all_images = [open_image(os.path.join(root, x)) for x in files]
            test_images += all_images[:100]
            train_images += all_images[100:]

    return (train_images, train_labels), (test_images, test_labels)


def load_ardis_mnist():
    '''
    load ardis and mnist datasets and combine them together
    '''
    global res_folder
    (train_images, train_labels), (test_images,
                                   test_labels) = load_ardis_data(res_folder)
    # load ardis dataset
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    # load mnist dataset
    (train_images2, train_labels2), (test_images2,
                                     test_labels2) = datasets.mnist.load_data()

    # append them together
    train_images = np.append(train_images, train_images2, axis=0)
    train_labels = np.append(train_labels, train_labels2, axis=0)
    test_images = np.append(test_images, test_images2, axis=0)
    test_labels = np.append(test_labels, test_labels2, axis=0)

    return (train_images, train_labels), (test_images, test_labels)
