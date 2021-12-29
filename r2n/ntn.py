#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

class NeuralTensorLayer(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim, arity=2, name=None, reg_weight=0.0, **kwargs):
    super().__init__(name=name, **kwargs)

    self.output_dim = output_dim  #k
    self.input_dim = input_dim    #d
    self.arity = arity

    k = self.output_dim
    d = self.input_dim
    w_init = tf.random_normal_initializer()
    initial_W = w_init(shape=(k, d, d))
    self.W = tf.Variable(initial_value=initial_W, dtype=tf.float32,
                         trainable=True)
    initial_V = w_init(shape=(arity * d, k))
    self.V = tf.Variable(initial_value=initial_V, dtype=tf.float32,
                         trainable=True)
    self.b = tf.Variable(
        initial_value=tf.zeros((1, self.output_dim), dtype=tf.float32),
        trainable=True)

    self._trainable_weights = [self.W, self.V, self.b]

    if reg_weight > 0.0:
        self.add_loss(lambda: reg_weight * self.get_untracked_regularization_loss())


  # Input: inputs is a tensor (or vector of tensors):
  # (arity, n=num_patterns, d=feature_size)
  # Output: tensor with shape (n=num_patterns, d=feature_size)
  def call(self, inputs, **kwargs):
    e = []
    for i in range(self.arity):
      ei = inputs[i]  # n x d
      e.append(ei)

    # This is defined for relations of any arity.
    linear_product = tf.linalg.matmul(tf.concat(e, axis=-1), self.V)  # n x k
    embeddings = linear_product + self.b

    if self.arity == 2:
      # TODO: generalization to relations of any arity could be done.
      e1t = tf.transpose(e[0])  # d x n
      e2t = tf.transpose(e[1])  # d x n

      # W_e2t[k,d,n] = sum_i W[k,d,i] * e2t[i, n]
      # kdn = sum_i kdi * in
      # kdn = kdi, in
      # kdi, in->kdn
      W_e2t = tf.einsum('kdi,in->kdn', self.W, e2t) # k x d x n

      # e1t_W_e2t[k, n] = sum_d e1t[d, n] * W_e2t[k, d, n]
      # kn = sum_d dn * kdn
      # kn = dn, kdn
      # dn, kdn->kn
      e1t_W_e2t = tf.einsum('dn, kdn->kn', e1t, W_e2t)  # k x n
      bilinear_product = tf.transpose(e1t_W_e2t)  # n x k
      embeddings += bilinear_product

    embeddings = tf.nn.tanh(embeddings)
    return embeddings

  def get_untracked_regularization_loss(self):
      return tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.V) + tf.nn.l2_loss(self.b)

  def compute_output_shape(self, input_shape):
      return (self.output_dim, self.input_dim)


# Main acts as a test.
def main():
  num_inputs = 5
  constant_embedding_size = 3
  atom_embedding_size = 2
  ntn =  NeuralTensorLayer(constant_embedding_size, atom_embedding_size)

  inputs = []
  w_init = tf.random_normal_initializer()
  inputs.append(tf.constant(w_init(shape=(num_inputs,
                                          constant_embedding_size))))
  inputs.append(tf.constant(w_init(shape=(num_inputs,
                                          constant_embedding_size))))
  print('O', ntn(inputs))

if __name__ == '__main__':
    main()
