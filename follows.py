import mme
import tensorflow as tf
import datasets
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


np.random.seed = 0
tf.random.set_seed(0)

num_examples = 50

(x_train, hb_train), (x_test, hb_test) = datasets.mnist_follows(num_examples, seed=1)

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
nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  # up to the last hidden layer
# nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  # up to the last hidden layer
nn.add(tf.keras.layers.Dense(10,use_bias=False))
p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)


#Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it)
p2 = mme.potentials.MutualExclusivityPotential(indices=indices)

#Logical
c = mme.Formula(definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)", ontology=o)
p3 = mme.potentials.EvidenceLogicPotential(formula=c,
                                           logic=mme.logic.BooleanLogic,
                                           evidence=y_e_train,
                                           evidence_mask=m_e)

# p3 = mme.potentials.LogicPotential(formula=c,
#                                            logic=mme.logic.BooleanLogic)



P = mme.potentials.GlobalPotential([p1,p2,p3])


pwt = mme.PieceWiseTraining(global_potential=P, y=hb_train)
pwt.compute_beta_logical_potentials()


p2.beta = 19
print(p3.beta)

epochs = 500
for _ in range(epochs):
    pwt.maximize_likelihood_step(hb_train, x=x_train)

"""Inference"""
hb = hb_test
x = x_test
evidence = y_e_test
evidence_mask = m_e>0


map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                potential=P,
                                                logic=mme.logic.LukasiewiczLogic,
                                                evidence=evidence,
                                                evidence_mask=evidence_mask,
                                                learning_rate=0.06)



for i in range(1000):
    map_inference.infer_step(x)
    y_test = tf.reshape(hb[0, :num_examples * 10], [num_examples, 10])
    y_map = tf.reshape(map_inference.map()[0, :num_examples * 10], [num_examples, 10])
    y_nn = p1.model(x)

    acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
    print("Accuracy MAP", acc_map)
    print("Accuracy NN", acc_nn)

    if mme.utils.heardEnter():
        break

y_test = tf.reshape(hb[0, :num_examples * 10], [num_examples, 10])
y_map = tf.reshape(map_inference.map()[0, :num_examples * 10], [num_examples, 10])
y_nn = p1.model(x)

acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))

print("Accuracy MAP", acc_map)
print("Accuracy NN", acc_nn)











