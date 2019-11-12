import unittest
from mme import Ontology, Domain, Predicate
import mme
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datasets


"""Loading Data"""
num_examples = 10


(x_train, hb), (x_test, hb_test) = datasets.mnist_linked_plus_minus_1(num_examples)

"""Logic description"""
o = Ontology()

images = mme.Domain("Images", data=x_train)
numbers = mme.Domain("Numbers", data=np.array([0,1,2,3,4,5,6,7,8,9]).T)
o.add_domain([images,numbers])

digit = mme.Predicate("digit", domains=[images, numbers])
link = mme.Predicate("link", domains=[images, images])
pm1 = mme.Predicate("plus_minus_one", domains=[numbers,numbers])
equal = mme.Predicate("equal", domains=[numbers,numbers])
o.add_predicate([digit,link,pm1,equal])
indices = np.reshape(np.arange(images.num_constants * numbers.num_constants), [images.num_constants,numbers.num_constants])

"""Defining a neural model on which to condition our distribution"""
nn = tf.keras.Sequential()
nn.add(tf.keras.layers.Input(shape=(784,)))
nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  # up to the last hidden layer
nn.add(tf.keras.layers.Dense(10, use_bias=False))  # up to the last hidden layer

"""Instantiating the supervised potential"""
p1 = mme.potentials.SupervisionLogicalPotential(model=nn, indices= indices)
p2 = mme.LogicPotential(constraint=mme.Constraint(formula="digit(x,d) and digit(y,q) and link(x,y) -> plus_minus_one(d,q)", ontology=o),
                        logic=mme.logic.BooleanLogic)
p3 = mme.LogicPotential(constraint=mme.Constraint(formula="(not equal(d,q)) <-> (digit(x,d) xor digit(x,q)) ", ontology=o),
                        logic=mme.logic.BooleanLogic)


P = mme.potentials.GlobalPotential([p1,p2,p3])

pwt = mme.PieceWiseTraining(global_potential=P, learning_rate=0.01, y=hb)
pwt.compute_beta_logical_potentials()

p2.beta = p3.beta = 1000


epochs = 20
for _ in range(epochs):
    pwt.maximize_likelihood_step(hb,x=x_train)


# """Inference"""
# evidence = np.zeros([1,len(hb[0])])
# evidence[0,num_examples*10:] = 1
# evidence_mask = np.array(evidence) > 0
#
# evidence = np.zeros([1,len(hb[0])])
# evidence[0,num_examples*10:] = hb[0,num_examples*10:]
# map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
#                                                 potential=P,
#                                                 logic=mme.logic.LukasiewiczLogic,
#                                                 evidence=evidence,
#                                                 evidence_mask=evidence_mask,
#                                                 learning_rate=0.001)
#
# hb=hb
# x=x_train
#
#
# for i in range(200):
#     map_inference.infer_step(x)
#     print(map_inference.map())
#
#
# y_test = tf.reshape(hb[0,:num_examples*10], [num_examples,10])
# y_map = tf.reshape(map_inference.map()[0,:num_examples*10], [num_examples,10])
# y_nn = p1.model(x)
#
# print("Test accuracy with map", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32)))
# print("Test accuracy with nn", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32)))
#


"""Inference with sampling"""
evidence = np.zeros([1,len(hb[0])])
evidence[0,num_examples*10:] = 1
evidence_mask = np.array(evidence) > 0

evidence = np.zeros([1,len(hb[0])], dtype=np.float32)
evidence[0,num_examples*10:] = hb[0,num_examples*10:]

sampler = mme.inference.GPUGibbsSampler(potential=P, num_variables=o.herbrand_base_size,
                                                num_chains=10, evidence=evidence, evidence_mask=evidence_mask)


print(sampler.sample(x_train).shape)