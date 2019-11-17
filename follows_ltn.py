import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')


def main(lr, seed, perc_soft, l2w=0.01, w_rule=0.01):
    num_examples = 200

    (x_train, hb_train), (x_test, hb_test) = datasets.mnist_follows(num_examples, seed=0, perc_soft=perc_soft)

    # I set the seed after since i want the dataset to be always the same
    np.random.seed(seed)
    tf.random.set_seed(seed)


    m_e = np.zeros_like(hb_train)
    m_e[:, num_examples*10:] = 1

    y_e_train = hb_train * m_e
    y_e_test = hb_test * m_e

    """Logic Program Definition"""
    o = mme.Ontology()

    #Domains
    images = mme.Domain("Images", data=x_train)
    numbers = mme.Domain("Numbers", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T)
    o.add_domain([images, numbers])

    # Predicates
    digit = mme.Predicate("digit", domains=[images, numbers])
    links = mme.Predicate("links", domains=[images, images], given=True)
    follows = mme.Predicate("follows", domains=[numbers, numbers], given=True)
    o.add_predicate([digit, links, follows])

    """MME definition"""

    #Supervision
    indices = np.reshape(np.arange(images.num_constants * numbers.num_constants),
                         [images.num_constants, numbers.num_constants])
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(784,)))
    nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(10, use_bias = False, activation=None))  # up to the last hidden layer


    adam = tf.keras.optimizers.Adam(lr=0.001)


    def make_hb_with_model(neural_digits, hb):
        digits = tf.reshape(neural_digits,[-1, num_examples*10])
        hb = tf.concat((digits, hb[:,num_examples*10:]), axis=1)
        return hb


    def training_step(logic=False):
        with tf.GradientTape() as tape:
            neural_logits = nn(x_train)


            total_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(hb_train[:, :num_examples * 10], [-1, 10]),
                                                        logits=neural_logits)) + tf.reduce_sum(nn.losses)

            if logic:
                neural_softmax = tf.nn.softmax(nn(x_train))

                hb_model_train = make_hb_with_model(neural_softmax, hb_train)

                c = mme.Formula(definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)", ontology=o)
                groundings = c.ground(herbrand_interpretation=hb_model_train)
                logical_loss = tf.reduce_sum(- c.compile(groundings, mme.logic.LukasiewiczLogic))
                total_loss += w_rule*logical_loss

        grads = tape.gradient(target=total_loss, sources=nn.variables)
        grad_vars = zip(grads, nn.variables)
        adam.apply_gradients(grad_vars)



    logic = False
    epochs = 150
    y_test = tf.reshape(hb_test[0, :num_examples * 10], [num_examples, 10])
    for e in range(epochs):
        training_step(logic)
        # if e%10==0:
        #     y_nn = tf.nn.softmax(nn(x_test))
        #     acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        #     print(acc_nn)
        # if mme.utils.heardEnter(): break
    y_nn = tf.nn.softmax(nn(x_test))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
    print(acc_nn)

    if logic: return acc_nn,acc_nn


    print("MAP Inference")
    steps_map = 500



    prior = tf.nn.softmax(nn(x_test))
    y_bb = tf.Variable(initial_value=prior)
    # prior = nn(x_test)
    # y_bb = tf.Variable(initial_value=0.5 * tf.ones_like(prior))
    c = mme.Formula(definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)", ontology=o)
    #
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     lr, decay_steps=steps_map, decay_rate=0.96, staircase=True)

    adam2 = tf.keras.optimizers.Adam(lr=lr)


    # Recreating an mme scenario
    mutual = mme.potentials.MutualExclusivityPotential(indices)


    # print(acc_map)
    def map_inference_step():

        with tf.GradientTape() as tape:
            y_map = tf.sigmoid(10 * (y_bb - 0.5))
            hb_model_test = make_hb_with_model(y_map, hb_test)
            groundings = c.ground(herbrand_interpretation=hb_model_test)
            inference_loss = w_rule * tf.reduce_sum(- c.compile(groundings, mme.logic.LukasiewiczLogic)) \
                             +  tf.reduce_sum(tf.square(prior - y_bb)) \
                             + 100 * tf.reduce_sum(- mutual.call_on_groundings(y_map))



        grads = tape.gradient(target=inference_loss, sources=y_bb)
        grad_vars = [(grads, y_bb)]
        adam2.apply_gradients(grad_vars)


    for e in range(steps_map):
        map_inference_step()
        # acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_bb, axis=1)), tf.float32))
        # print(acc_map)
        if mme.utils.heardEnter(): break
    acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_bb, axis=1)), tf.float32))

    return acc_nn, acc_map




if __name__ == "__main__":
    res = []
    seed = 0

    for lr, perc, w_rule,  in product([1, 0.3, 0.1, 0.06, 0.03]
                              ,[0., 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1]
                              ,[0.1, 1., 10., 100],):
        acc_nn, acc_map = main(lr=lr, seed=seed, perc_soft=perc, w_rule=w_rule)
        res.append("\t".join([str(a) for a in [lr, perc, w_rule, acc_nn.numpy(), str(acc_map.numpy())+"\n"]]))
        for i in res:
            print(i)


    with open("res_lyrics_cc", "a") as file:
        file.write("lr,perc,w_rule,acc_nn, acc_map\n")
        file.writelines(res)







