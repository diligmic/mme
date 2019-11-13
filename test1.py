import unittest
from mme import Ontology, Domain, Predicate
import mme
from itertools import product
import numpy as np
import tensorflow as tf
import datasets



y = [[0, 1, 0, #friend of
             1, 0, 0,
             0, 0, 0,
             1, 1, 0]]#smokes

y_e = [[0, 1, 0, #friend of
               1, 0, 0,
               0, 0, 0,
               -1, -1, -1]]#smokes
m_e = [[1,1, 1, #friend of
        1, 1, 1,
        1, 1, 1,
        0, 0, 0]]#smokes


G_c1 = [[9, 0, 9], #smokes(Alice), friendOf(Alice,Alice), smokes(Alice)
        [9, 1, 10],
        [9, 2, 11],
        [10, 3, 9],
        [10, 4, 10],
        [10, 5, 11],
        [11, 6, 9],
        [11, 7, 10],
        [11, 8, 11]]

m_e = np.array(m_e)
G_c1 = np.array(G_c1)
n = len(G_c1[0])
N = len(G_c1)
m_e_G_c1 = np.take(m_e, G_c1, axis=1)
y_e_G_c1 = np.take(y_e, G_c1, axis=1)



k =  n - sum(m_e_G_c1[0,0])
num_example = 1


shape = [n, num_example, N, 2**k]

indices = tf.where(m_e_G_c1[0][0]>0)
given = tf.reshape(tf.gather(y_e_G_c1, tf.squeeze(indices), axis=-1), [1, num_example, -1, 1])
given = tf.cast(tf.tile(given, [1, 1, 1, 2**k]), tf.float32)
first = tf.scatter_nd(shape=shape, indices=indices, updates=given)


indices =tf.where(m_e_G_c1[0][0]<1)
l = list(product([False, True], repeat=k))
comb = np.stack(l, axis=1).astype(np.float32)
assignments = np.tile(np.reshape(comb, [-1,1,1,2**k]), [1,1,N,1])
second = tf.scatter_nd(shape=shape, indices=indices, updates=assignments)

final = tf.transpose(first+second, [1,2,3,0])

print(final)










# given = np.tile(np.expand_dims(y_e_G_c1, axis=-2), [1,2**k,1])
# mask_given = np.tile(np.expand_dims(m_e_G_c1, axis=-2), [1,2**k,1])
# print(given)
# print(mask_given)
# exit()
#
#
# fr = 0
# to = 0
# for r in range(len(n)):
#     if m_e_G_c1[0,r]
#
# assignments = np.tile(np.expand_dims(all_combinations(k), axis=0), [len(G_c1), 1 ,1])
# print(assignments)
