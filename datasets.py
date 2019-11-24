import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.ndimage as img
import matplotlib.pyplot as plt

np.random.seed(0)


def mnist_linked_plus_minus_1(num_examples):
    
    
    def __inner__(y_train):
        hb_link = np.zeros([num_examples,num_examples])
        for i ,x in enumerate(y_train):
            for j,y in enumerate(y_train):
                if abs(x - y) == 1:
                    hb_link[i,j] = 1
        hb_link = np.reshape(hb_link, [1, -1])


        hb_pm1 =  np.zeros([10,10])
        for i in range(10):
            for j in range(10):
                if abs(i - j) == 1:
                    hb_pm1[i,j] = 1
        hb_pm1 = np.reshape(hb_pm1, [1, -1])

        y_train = np.eye(10)[y_train]
        hb_digit = np.reshape(y_train, [1, -1])

        hb_equal = np.reshape(np.eye(10), [1, -1])

        hb=np.concatenate([hb_digit,hb_link,hb_pm1,hb_equal], axis=1)
        return hb

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train[:num_examples]
    y_train = y_train[:num_examples]
    
    x_test = x_test[:num_examples]
    y_test = y_test[:num_examples]
    
    
    x_train = np.reshape(x_train, [-1, 784])
    hb_train = __inner__(y_train)
    
    x_test = np.reshape(x_test, [-1, 784])
    hb_test = __inner__(y_test)

    return (x_train, hb_train),(x_test, hb_test)


def mnist_equal(num_examples):
    def __inner__(y_train):

        y_train = np.eye(10)[y_train]
        hb_digit = np.reshape(y_train, [1, -1])

        hb_equal = np.reshape(np.eye(10), [1, -1])

        hb = np.concatenate([hb_digit, hb_equal], axis=1)
        return hb

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:num_examples]
    y_train = y_train[:num_examples]

    x_test = x_test[:num_examples]
    y_test = y_test[:num_examples]

    x_train = np.reshape(x_train, [-1, 784])
    hb_train = __inner__(y_train)

    x_test = np.reshape(x_test, [-1, 784])
    hb_test = __inner__(y_test)

    return (x_train, hb_train), (x_test, hb_test)


def mnist_follows(num_examples, seed = 0, perc_soft=0.1):
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(len(x_train))
    if num_examples > len(x_test):
        raise Exception("I cannot create a test of this size.")


    _,x_train, _, y_train = train_test_split(x_train, y_train, test_size=num_examples, stratify=y_train, random_state=0)
    _,x_test, _, y_test = train_test_split(x_test, y_test, test_size=num_examples, stratify=y_test, random_state=0)


    def _inner(x,y):

        x_new = []
        for I in x:
            I = I / 255
            # I = img.rotate(I, float(np.random.rand() * 90), reshape=False)
            # I = I + 0.3 * np.random.randn(28,28)
            x_new.append(I)
        x = np.reshape(x_new, [-1, 28*28])

        links = np.zeros([num_examples,num_examples])
        for i,y_i in enumerate(y):
            for j, y_j in enumerate(y):

                if y_i == y_j + 1:
                    # if np.random.rand() < 0.9:
                        links[i,j] = 1
                else:
                    r = np.random.rand()
                    if  r < perc_soft:
                        links[i,j] = 1
        links = np.reshape(links, [1, -1])

        follows = np.zeros([10,10])
        for i in range(10):
            for j in range(10):
                if i == j + 1:
                    follows[i,j] = 1
        follows = np.reshape(follows, [1, -1])

        y = np.eye(10)[y]
        digit = np.reshape(y, [1, -1])

        hb = np.concatenate((digit, links, follows), axis=1)

        return (x,hb)


    return _inner(x_train, y_train), _inner(x_test, y_test)



def citeseer(test_size = 0.5, valid_size=0.):
    documents = np.load("data/citeseer/words.npy")
    n = len(documents)
    documents = documents[:n]
    labels = np.load("data/citeseer/labels.npy")
    labels = labels[:n]
    labels = np.eye(6)[labels]
    citations = np.greater(np.load("data/citeseer/citations.npy"), 0).astype(np.float32)
    citations = citations[:n, :n]
    num_documents = len(documents)
    num_classes = 6


    def _inner_take_hb(idx):

        x = documents[idx]
        l = np.reshape(labels[idx].T, [1, -1])
        c = np.reshape(citations[idx][:,idx], [1, -1])

        hb = np.concatenate((l,c), axis=1)
        hb = hb.astype(np.float32)

        return x, hb



    trid, teid = train_test_split(np.arange(num_documents), test_size=test_size, random_state=0)
    trid, vaid = train_test_split(trid, test_size=valid_size, random_state=0) if valid_size > 0 else (trid, None)


    x_train, hb_train = _inner_take_hb(trid)
    x_valid, hb_valid = _inner_take_hb(vaid) if valid_size > 0 else (None, None)
    x_test,hb_test =  _inner_take_hb(teid)

    return (x_train, hb_train), (x_valid,hb_valid), (x_test, hb_test)




def citeseer_em(test_size, valid_size, seed):


    documents = np.load("data/citeseer/words.npy")
    n = len(documents)
    documents = documents[:n]
    labels = np.load("data/citeseer/labels.npy")
    labels = labels[:n]
    labels = np.eye(6)[labels]
    citations = np.greater(np.load("data/citeseer/citations.npy"), 0).astype(np.float32)
    citations = citations[:n, :n]
    num_documents = len(documents)

    def _inner_take_hb(idx):
        x = documents[idx]
        l = np.reshape(labels[idx].T, [1, -1])
        c = np.reshape(citations[idx][:, idx], [1, -1])

        hb = np.concatenate((l, c), axis=1)
        hb = hb.astype(np.float32)

        return x, hb

    trid, teid = train_test_split(np.arange(num_documents), test_size=test_size, random_state=seed)

    trid, vaid = train_test_split(trid, test_size=valid_size, random_state=seed) if valid_size>0 else (trid, None)

    mask_train_labels = np.zeros_like(labels)
    mask_train_labels[trid] = 1

    x_train, hb_train = _inner_take_hb(trid)
    x_valid, hb_valid = _inner_take_hb(vaid) if valid_size>0 else (None, None)
    x_test, hb_test = _inner_take_hb(teid)
    x_all, hb_all = _inner_take_hb(np.arange(n))

    return (x_train, hb_train), (x_valid, hb_valid), (x_test, hb_test), (x_all, hb_all), labels, mask_train_labels, trid, vaid, teid


