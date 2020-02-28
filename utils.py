import os, select, sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from sortedcontainers import SortedDict
import re



if os.name == 'nt':
    import msvcrt

def heardEnter():
    # Listen for the user pressing ENTER

    if os.name == 'nt':
        if msvcrt.kbhit():
            if msvcrt.getch() == b"q":
                print("Quit key pressed.")
                return True
        else:
            return False
    else:
        i, o, e = select.select([sys.stdin], [], [], 0.0001)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                return True
        return False

def save(path, o):
    with open(path, 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore(path):
    with open(path, 'rb') as handle:
        o = pickle.load(handle)
    return o


def ranking(relations_train, relations_test, relations_predicted):
    MRR = 0.0
    HITS1 = 0.0
    HITS3 = 0.0
    HITS5 = 0.0
    HITS10 = 0.0
    counter = 0.0

    for relation in relations_test.keys():
        r_test = relations_test[relation]
        r_predicted = relations_predicted[relation]
        r_train=relations_train[relation]
        n = np.shape(r_test)[0]
        for i in range(n):
            for j in range(n):
                if r_test[i, j] == 1:
                    for s, k in enumerate((i, j)):

                        # s k
                        # 0 i
                        # 1 j

                        predicted_score = r_predicted[i, j]

                        # we multiply for 1 - r_train in such a way to eliminate the scores coming from train data
                        if s == 0:
                            mask = (1 - r_train[i]) # ones row with 0 when the data is in the training data
                            mask[j] = 1
                            all_scores = sorted(r_predicted[i]*mask, reverse=True)
                        else:
                            mask = 1 - r_train[:,j]
                            mask[i] = 1
                            all_scores = sorted(r_predicted[:, j]*mask, reverse=True)
                        rank = all_scores.index(predicted_score) + 1

                        # if k == i:
                        #     all_scores = np.argsort(r_predicted[i])[::-1]
                        #     rank = list(all_scores).index(j) + 1
                        # else:
                        #     all_scores = np.argsort(r_predicted[:, j])[::-1]
                        #     rank = list(all_scores).index(i) + 1

                        counter += 1.0
                        if rank <= 1:
                            HITS1 += 1.0
                        if rank <= 3:
                            HITS3 += 1.0
                        if rank <= 5:
                            HITS5 += 1.0
                        if rank <= 10:
                            HITS10 += 1.0

                        MRR += 1.0 / rank

    MRR /= counter
    HITS1 /= counter
    HITS3 /= counter
    HITS5 /= counter
    HITS10 /= counter

    return (MRR, HITS1, HITS3, HITS5, HITS10)




class CustomDense(tf.keras.layers.Layer):


  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(CustomDense, self).__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = tf.dtypes.as_dtype(self.dtype)
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = tf.cast(inputs, self._compute_dtype)
      if tf.is_sparse(inputs):
        outputs = tf.sparse_tensor_dense_matmul(inputs, self.kernel)
      else:
        outputs = tf.math.mat_mul(inputs, self.kernel)
    if self.use_bias:
      outputs = outputs + self.bias
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs


def read_file_fixed_world(file, constants, predicates):


    t_constants = constants
    constants = SortedDict()
    for a in sorted(t_constants):
        constants[a] = len(constants)
    data = SortedDict()
    for p in predicates.keys():
        data[p] = []
    with open(file) as f:
        for line in f:
            p,rest = line.split("(")
            rest = re.sub(r"\)[\\*]*\s*", "", rest)
            args = rest.split(",")
            for i in range(len(args)):
                arg = args[i].replace(" ","")
                args[i] = arg
            try:
                if len(data[p])>0: assert(len(data[p][0]) == len(args))
            except:
                print(file)
            data[p].append(args)
    m_data = SortedDict()
    for key in sorted(data.keys()):
        k = key
        v = data[k]
        m_data[k] = np.zeros(shape=[len(constants) for _ in range(predicates[k])])
        for args in v:
            id = [constants[a] for a in args]
            np.put(m_data[k], np.ravel_multi_index(id, m_data[k].shape), 1)

    return constants,m_data,{}