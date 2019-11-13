import numpy as np
import tensorflow as tf

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
