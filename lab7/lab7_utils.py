from enum import Enum
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

seed = 42

class Dataset_Select(Enum):
    MNIST = "mnist"
    F_MNIST = "fashion_mnist"
    K_MNIST = "kmnist"
    KUZ_49 = "Kuzushiji-49"

def fetch_from_hg(dataset_name):
    dataset = load_dataset(dataset_name)

    X_train = dataset['train']['image']
    X_test = dataset['test']['image']
    y_train = dataset['train']['label']
    y_test = dataset['test']['label']

    X_train = np.array([np.array(image) for image in X_train])
    X_test = np.array([np.array(image) for image in X_test])

    return X_train, X_test, y_train, y_test

def convert_tsfl_dataset_to_nparray(dataset):
    images = []
    labels = []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    
    return np.array(images), labels

def fetch_from_tsfl(dataset_name):
    dataset = tfds.load(dataset_name, split=["train", "test"], as_supervised=True)

    train_dataset, test_dataset = dataset

    X_train, y_train = convert_tsfl_dataset_to_nparray(train_dataset)
    X_test, y_test = convert_tsfl_dataset_to_nparray(test_dataset)

    return X_train, X_test, y_train, y_test

def fetch_from_openml(dataset_name):
    dataset = fetch_openml(name=dataset_name, version=1)

    X = dataset.data.to_numpy()
    X = X.reshape(X.shape[0], 28, 28)
    y = dataset.target.to_numpy()

    return train_test_split(X, y, test_size=1/(1+6), random_state=seed)

def select_fetch_fucntion(dataset_name):
    if dataset_name == Dataset_Select.KUZ_49.value:
        fetch_fun = fetch_from_openml
    elif dataset_name == Dataset_Select.K_MNIST.value:
        fetch_fun = fetch_from_tsfl
    else:
        fetch_fun = fetch_from_hg
    return fetch_fun

def print_samples(images, n):
    plt.figure(figsize=(n, 2))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def binarize(X_train, X_test):
    X_train = X_train.reshape((X_train.shape[0], 28, 28)) / 255.
    X_train = np.where(X_train > .5, 1.0, 0.0).astype(np.uint8)

    X_test = X_test.reshape((X_test.shape[0], 28, 28)) / 255.
    X_test = np.where(X_test > .5, 1.0, 0.0).astype(np.uint8)

    return X_train, X_test

def flatten_image(X_train, X_test):
    X_train = X_train.reshape((X_train.shape[0], 784))
    X_test = X_test.reshape((X_test.shape[0], 784))
    
    return X_train, X_test



def get_dataset(dataset_name, print_sample_number):
    fetch_fun = select_fetch_fucntion(dataset_name)

    X_train, X_test, y_train, y_test = fetch_fun(dataset_name)

    X_train, X_test = binarize(X_train, X_test)

    if print_sample_number > 0:
        print_samples(X_train, print_sample_number)

    X_train, X_test = flatten_image(X_train, X_test)

    return X_train, X_test, y_train, y_test
   
