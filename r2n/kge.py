import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.regularizers import l2
from r2n.utils import StringEqualsCaseInsensitive, Params
import r2n.ntn as ntn

###############################
# Layers embedding Constants.
# Get the embedding size for a given domain.
# Returns:
# - if embedding is enabled for the domain
# - the embedding size, or the input size when embedding is not enabled and the input space has features.
def _get_constant_embedding_size(params, inputs, domain):
    if domain is not None and domain in params.per_domain_constant_embedding_sizes:
        return True, params.per_domain_constant_embedding_sizes[domain]

    elif (params.constant_embedding_sizes and
          params.constant_embedding_sizes[-1] > 0):
        # Global constant.
        return True, params.constant_embedding_sizes[-1]

    elif (domain is not None and inputs is not None and
          # Annoying but in some mono-domain setups inputs is directly a tensor.
          isinstance(inputs, dict) and
          domain in inputs):
        # Feature based, no embedding.
        return False, inputs[domain].shape[-1]

    return False, 0

# Get the embedding size for all domains.
# Returns:
def _get_constant_embedding_size_by_domain(params, inputs, domain_names):
    embedding_enabled = False
    domain2constant_size = {}
    domain2constant_enabled = {}
    for domain in domain_names:
        enable, size = _get_constant_embedding_size(params, inputs, domain)
        domain2constant_enabled[domain] = enable
        domain2constant_size[domain] = size
        if enable:
            embedding_enabled = True

    return embedding_enabled, domain2constant_size, domain2constant_enabled


class DomainConstantEmbedding(tf.keras.layers.Layer):
    """Takes an hidden representation of constants for a single domain and creates
    representation. Optionally, embeds the constants before concatenations (if
    len(embedding_sizes) > 0)."""
    def __init__(self, o, embedding_size, params):
        super().__init__()
        activation = params.constant_embedder_activation
        reg_weight = params.constant_embedder_regularization
        normalize = params.constant_embedding_normalization
        assert embedding_size > 0

        self.o = o
        self.normalize = normalize
        self.embedder = None

        regularizer = None
        if reg_weight > 0.0:
            regularizer = tf.keras.regularizers.l2(reg_weight)
        self.embedder = Dense(embedding_size, use_bias=False, activation=activation,
                              kernel_regularizer=regularizer)

    def call(self, inputs, **kwargs):
        features = self.embedder(inputs)
        if self.normalize:
            features = tf.math.l2_normalize(features, axis=-1)
        return features


class ConstantEmbedding(tf.keras.layers.Layer):
    """Calls the constant embedders, differenciating the behavior of the single domains."""
    def __init__(self, o, inputs, params):
        super().__init__()
        enabled, domain2size, domain2enabled = _get_constant_embedding_size_by_domain(
            params, inputs, o.domains.keys())
        assert enabled, 'Initializing a constant embedder that is not required for any domain.'
        self.domain2size, self.domain2enabled = domain2size, domain2enabled

        self.embedder = {}
        for name, d in o.domains.items():
            assert name in domain2enabled
            if not domain2enabled[name]:
                continue
            self.embedder[name] = DomainConstantEmbedding(o, domain2size[name], params)

    def call(self, domain_inputs, **kwargs):
        domain_features = {}
        for name, d in domain_inputs.items():
            if self.domain2enabled[name]:
                domain_features[name] = self.embedder[name](d)
            else:
                domain_features[name] = d
        return domain_features


###############################
def activation_factory(str):
    if not str or StringEqualsCaseInsensitive(str, 'linear'):
        return tf.identity
    elif StringEqualsCaseInsensitive(str, 'relu'):
        return tf.nn.relu
    elif StringEqualsCaseInsensitive(str, 'elu'):
        return tf.nn.elu
    elif StringEqualsCaseInsensitive(str, 'sigmoid'):
        return tf.nn.sigmoid
    elif StringEqualsCaseInsensitive(str, 'tanh'):
        return tf.nn.tahn
    elif StringEqualsCaseInsensitive(str, 'softmax'):
        return tf.nn.softmax
    else:
        assert False, 'Unknown activation %s' % str

###############################
# Layers embedding Atoms.
# All layers assume to have as input
# tuple_features = self._build_tuples(inputs)
# and return a tensor
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim

########################################
# Common functionalities to all embedding layers.
class AtomEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, o, inputs, params):
        super().__init__()
        self.atom_embedding_sizes = params.atom_embedding_sizes
        if not isinstance(self.atom_embedding_sizes, list):
            self.atom_embedding_sizes = [self.atom_embedding_sizes]
        assert self.atom_embedding_sizes[-1] > 0

        self.o = o
        self.normalize = params.atom_embedding_normalization
        self.reg_weight = params.atom_embedder_regularization
        # self.constant_embedding_size = params.constant_embedding_sizes[-1]

        self.drop_relational_dim = False

        self.embedders = self._atom_embedder_factory(inputs, params)

    def _build_tuples(self, inputs):
        tuple_features = {}
        for k,ids in self.o.tuple_indices.items():
            # k is domain
            # Cartesian product of the domains.
            X = []
            # ids.shape[1] is num dim in the tuple.
            for i in range(ids.shape[1]):
                features = inputs[k[i]]
                indices = ids[:, i]
                x_i_th_element_of_tuple = tf.gather(features, indices, axis=1)
                X.append(x_i_th_element_of_tuple)
                tuple_features[k] = X
        return tuple_features

    def _atom_embedder_factory(self, inputs, params):
        embedders = {}
        if not params.atom_embedding_sizes or params.atom_embedding_sizes[-1] <= 0:
            return embedders

        atom_embedding_size = params.atom_embedding_sizes[-1]

        for i, (name, p) in enumerate(self.o.predicates.items()):
            constant_embedding_size = None
            for domain in p.domains:
                _, es = _get_constant_embedding_size(params, inputs, domain)
                if constant_embedding_size is not None:
                    assert constant_embedding_size == es
                else:
                    constant_embedding_size = es

            atom_embedder = (params.atom_embedder
                             if p.name not in params.per_predicate_atom_embedder
                             else params.per_predicate_atom_embedder[p.name])

            if StringEqualsCaseInsensitive(atom_embedder, 'TransE'):
                embedders[p.name] = TransEEmbedder(
                    p, atom_embedding_size, activation=None,
                    reg_weight=self.reg_weight)

            elif StringEqualsCaseInsensitive(atom_embedder, 'RotateE'):
                assert atom_embedding_size == constant_embedding_size, (
                    'RotateE: constant and atom embedding space must be the same.')
                embedders[p.name] = RotateEEmbedder(atom_embedding_size,
                                                    reg_weight=self.reg_weight)

            elif StringEqualsCaseInsensitive(atom_embedder, 'DistMult'):
                assert atom_embedding_size == constant_embedding_size, (
                    'DistMult: constant and atom embedding space must be the same.')
                embedders[p.name] = DistMultEmbedder(
                    atom_embedding_size,
                    reg_weight=self.reg_weight)

            elif StringEqualsCaseInsensitive(atom_embedder, 'ComplEx'):
                assert 2 * atom_embedding_size == constant_embedding_size, (
                    'ComplEx: constant embedding size must be 2x atom embedding one. %d-%d' % (
                        constant_embedding_size, atom_embedding_size))
                embedders[p.name] = ComplexEmbedder(atom_embedding_size,
                                                    reg_weight=self.reg_weight)

            elif StringEqualsCaseInsensitive(atom_embedder, 'NTN'):
                self.drop_relational_dim = True
                assert constant_embedding_size > 0
                embedders[p.name] = ntn.NeuralTensorLayer(
                    constant_embedding_size,
                    atom_embedding_size,
                    p.arity,
                    reg_weight=self.reg_weight)

            elif StringEqualsCaseInsensitive(atom_embedder, 'MLP'):
                embedders[p.name] = MLPEmbedder(atom_embedding_size,
                                                reg_weight=self.reg_weight)

            elif StringEqualsCaseInsensitive(atom_embedder, 'MLP_cossim'):
                embedders[p.name] = MLPCossimEmbedder(p.arity, atom_embedding_size,
                                                      reg_weight=self.reg_weight)

            else:
                assert False, 'Unknown embedder:%s' % atom_embedder

        return embedders

    def call(self, inputs, **kwargs):
        tuple_features = self._build_tuples(inputs)

        if self.drop_relational_dim:
            # Remove the relational dimension if not assumed by the embedder.
            for k in tuple_features.keys():
                for i in range(len(tuple_features[k])):
                    tuple_features[k][i] = tf.squeeze(tuple_features[k][i], 0)

        predicate_atoms2embeddings = {}
        for name, p in self.o.predicates.items():
            k = tuple([d.name for d in p.domains])
            X = tuple_features[k]
            embeddings = self.embedders[p.name](X)
            if self.drop_relational_dim:
                # Add relational dim as assumed to be present by later layers.
                embeddings = tf.expand_dims(embeddings, 0)

            if self.normalize:
                embeddings = tf.math.l2_normalize(embeddings, axis=-1)

            predicate_atoms2embeddings[p.name] = embeddings

        atom_embeddings = self.o.fol_dictionary_to_linear_tf(
            predicate_atoms2embeddings)
        return atom_embeddings

########################################
class TransEEmbedder(tf.keras.layers.Layer):
    def __init__(self, p, embedding_sizes,
                 activation=None, reg_weight=0.0,
                 **kwargs):
        layer_name = "TransEEmbedder_%s" % p.name
        super().__init__()
        if not isinstance(embedding_sizes, list):
            embedding_sizes = [embedding_sizes]
        self.embedding_sizes = embedding_sizes
        if activation is None:
            assert len(embedding_sizes) == 1, (
                'Multiple linear layers, reduce this to a single layer')
            self.activation = activation

        self.regularizer = None
        if reg_weight > 0.0:
            self.regularizer = tf.keras.regularizers.l2(reg_weight)

        self.M = tf.Variable(#init(shape=[embedding_size]))
            initial_value=tf.random.normal(self.embedding_sizes, stddev=1.0))


    def call(self, inputs, **kwargs):
        X = inputs
        if len(X) == 2:
            embeddings = self.M + X[0] - X[1]
        else:
            embeddings = self.M - X[0]
        return embeddings


# Input
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim
# Output
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class TransEAtomOutputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    # Inputs is (num_atoms_in_formula, o.linear_size. atom_embedding_size)
    def call(self, inputs, **kwargs):
        # inputs becomes shaped (num_atoms_in_formula, o.linear_size)
        outputs =  tf.norm(inputs, ord=1, axis=-1)

        # outputs is also shaped (num_atoms_in_formula, o.linear_size)
        # High norm bring output to zero while a small norm returns 1.
        outputs = tf.ones_like(outputs) / (tf.ones_like(outputs) + outputs)
        return outputs

########################################
class DistMultEmbedder(tf.keras.layers.Layer):
    def __init__(self, embedding_size, reg_weight, use_prod=True):
        super().__init__()
        self.use_prod = use_prod  # if False, it implements DistAdd
        # Globrot does not work well for distmult, unclear why.
        #init = tf.initializers.GlorotUniform()
        self.M = tf.Variable(#init(shape=[embedding_size]))
            initial_value=tf.random.normal([embedding_size], stddev=1.0))
        if reg_weight > 0.0:
            regularizer = tf.keras.regularizers.l2(reg_weight)
            self.add_loss(lambda: regularizer(self.M))


    def call(self, inputs, **kwargs):
        if self.use_prod:
            # embeddings = self.M * inputs[0] * inputs[1]
            embeddings = self.M * tf.math.reduce_prod(tf.stack(inputs), axis=0)
        else:
            embeddings = self.M * tf.math.reduce_sum(tf.stack(inputs), axis=0)
        return embeddings


# Input
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim
# Output
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class DistMultAtomOutputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    # Inputs is (batch_dim, o.linear_size. atom_embedding_size)
    def call(self, inputs, **kwargs):
        outputs = tf.reduce_sum(inputs, axis=-1)
        outputs = tf.nn.sigmoid(outputs)
        return outputs

########################################
class RotateEEmbedder(tf.keras.layers.Layer):
    def __init__(self, embedding_size, reg_weight):
        super().__init__()
        self.M = tf.Variable(
            initial_value=tf.random.normal([embedding_size], stddev=0.1))

        if reg_weight > 0.0:
            regularizer = tf.keras.regularizers.l2(reg_weight)
            self.add_loss(lambda: regularizer(self.M))

    def call(self, inputs, **kwargs):
        if len(inputs) == 2:
            embeddings = self.M * inputs[0] - tf.reduce_sum(
                tf.stack(inputs[1:]), axis=0)
        else:
            embeddings = self.M * inputs[0]
        return embeddings


# Input
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim
# Output
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class RotateEAtomOutputLayer(TransEAtomOutputLayer):
    def __init__(self):
        super().__init__()

########################################
class ComplexEmbedder(tf.keras.layers.Layer):
    def __init__(self, embedding_size, reg_weight):
        super().__init__()
        self.embedding_size = embedding_size
        init = tf.initializers.GlorotUniform()
        # Relation "real" embedder.
        self.Rr = tf.Variable(init(shape=[embedding_size]))
        # Relation "imm" embedder.
        self.Ri = tf.Variable(init(shape=[embedding_size]))

        if reg_weight > 0.0:
            self.add_loss(lambda: reg_weight *
                          self.get_regularization_loss())

    def call(self, inputs, **kwargs):
        # assert len(inputs) == 2, 'Complex defined for binary relations, while arity:%d' % len(inputs)
        h_r, h_i = (inputs[0][:,:,:self.embedding_size],
                    inputs[0][:,:,self.embedding_size:])
        if len(inputs) == 1:
            embeddings = (h_r * self.Rr +
                          h_i * self.Rr +
                          h_r * self.Ri -
                          h_i * self.Ri)
        else:
            t_r, t_i = (inputs[1][:,:,:self.embedding_size],
                        inputs[1][:,:,self.embedding_size:])
            embeddings = (h_r * t_r * self.Rr +
                          h_i * t_i * self.Rr +
                          h_r * t_i * self.Ri -
                          h_i * t_r * self.Ri)
        return embeddings

    def get_regularization_loss(self):
        return (tf.nn.l2_loss(self.Rr) + tf.nn.l2_loss(self.Ri))


# Input
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim
# Output
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class ComplexAtomOutputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        outputs = tf.reduce_sum(inputs, axis=-1)
        outputs = tf.nn.sigmoid(outputs)
        return outputs


# Input: tensor with shape
# (batch_dim=1,
#  o.linear_size=num_atoms_predicate[0]+num_atoms_predicate[1]+...,
#  embedding_dim)
# Returns: tensor with shape
# (batch_dim=1, o.linear_size=num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class NeuralTensorNetworkAtomOutputLayer(tf.keras.layers.Layer):
    def __init__(self, o, input_embedding_size,
                 layer_name='NeuralTensorNetworkAtomOutput', activation=None):
        super().__init__()
        self.o = o
        self.activation = activation_factory(activation)
        self.layer_name = layer_name
        self.input_embedding_size = input_embedding_size

    def build(self, input_shape):  # Create the state of the layer (weights)
        u_init = tf.random_normal_initializer()
        self.U = {}
        for name, p in self.o.predicates.items():
            self.U[p.name] = self.add_weight(
                shape=(self.input_embedding_size, 1), initializer=u_init,
                dtype=tf.float32)
            super().build(input_shape)


    def call(self, inputs):
        predicates_outputs = []
        for name, p in self.o.predicates.items():
            a, b = self.o.id_range_by_predicate(p.name)
            predicate_inputs = inputs[:, a:b, :]
            # Multiplication:
            # (batch_dim, num_inputs, self.input_embedding_size) x
            # (self.input_embedding_size, 1) = (batch_dim, num_inputs, 1)
            predicate_outputs = tf.matmul(predicate_inputs, self.U[p.name])
            if self.activation is not None:
                predicate_outputs = self.activation(predicate_outputs)
                #predicate_outputs = self.U[p.name](predicate_inputs)
            # Shape of predicates_outputs num_predicates tensors:
            # [batch_dim, num_inputs, 1]
            predicates_outputs.append(predicate_outputs)

        # Concat the per-predicate outputs using the same order defined by
        # the ontology.
        # Shape of predicates_outputs (batch_dim, o.linear_size, 1)
        predicates_outputs = tf.concat(predicates_outputs, axis=1)
        # Remove last dim resulting shape (batch_dim, o.linear_size)
        predicates_outputs = tf.squeeze(predicates_outputs, axis=-1)
        return predicates_outputs


############################################
class MLPEmbedder(tf.keras.layers.Layer):
    def __init__(self, embedding_sizes, activation=tf.nn.relu, reg_weight=0.0,
                 **kwargs):
        super().__init__()
        if not isinstance(embedding_sizes, list):
            embedding_sizes = [embedding_sizes]
        self.embedding_sizes = embedding_sizes

        if activation is not None:
            assert len(embedding_sizes) == 1, (
                'Multiple linear layers, reduce this to a single layer')
            self.activation = activation

        self.regularizer = None
        if reg_weight > 0.0:
            self.regularizer = tf.keras.regularizers.l2(reg_weight)

        self.embedder = tf.keras.Sequential()
        for j, embedding_size in enumerate(self.embedding_sizes):
            self.embedder.add(Dense(
                embedding_size,
                use_bias=False, kernel_regularizer=self.regularizer,
                activation=self.activation))


    def call(self, inputs, **kwargs):
        embedding_list = []
        X = tf.concat(inputs, axis=-1)
        embeddings = self.embedder(X)
        return embeddings


# Input
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim
# Output
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class MLPAtomOutputLayer(tf.keras.layers.Layer):
    def __init__(self, reg_weight=0.0, activation=tf.nn.sigmoid):
        super().__init__()
        self.activation = activation_factory(activation)
        if activation is None:
            assert len(embedding_sizes) == 1, (
                'Multiple linear layers, reduce this to a single layer')
            self.activation = activation

        self.regularizer = None
        if reg_weight > 0.0:
            self.regularizer = tf.keras.regularizers.l2(reg_weight)

        self.layer = Dense(1, kernel_regularizer=self.regularizer, activation=self.activation)

    # Inputs is (num_atoms_in_formula, o.linear_size. atom_embedding_size)
    def call(self, inputs, **kwargs):
        return self.layer(inputs)

############################################
class MLPCossimEmbedder(tf.keras.layers.Layer):
    def __init__(self, arity, embedding_sizes, activation=tf.nn.relu, reg_weight=0.0,
                 **kwargs):
        super().__init__()
        self.arity = arity
        if not isinstance(embedding_sizes, list):
            embedding_sizes = [embedding_sizes]
        self.embedding_sizes = embedding_sizes

        if activation is not None:
            assert len(embedding_sizes) == 1, (
                'Multiple linear layers, reduce this to a single layer')
            self.activation = activation

        self.regularizer = None
        if reg_weight > 0.0:
            self.regularizer = tf.keras.regularizers.l2(reg_weight)

        self.embedders = []
        for i in range(self.arity):
            embedder = tf.keras.Sequential()
            for j, embedding_size in enumerate(self.embedding_sizes):
                embedder.add(Dense(embedding_size,
                                   use_bias=False,
                                   kernel_regularizer=self.regularizer,
                                   activation=self.activation))
                self.embedders.append(embedder)


    def call(self, inputs, **kwargs):
        assert len(inputs) == len(self.embedders)
        embeddings = []
        for embedder, X in zip(self.embedders, inputs):
            embeddings.append(embedder(X))
        embeddings = tf.concat(embeddings, axis=-1)
        return embeddings

# Input
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim
# Output
# batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...)
class CosineSimAtomOutputLayer(tf.keras.layers.Layer):
    def __init__(self, atom_embedding_size):
        super().__init__()
        self.atom_embedding_size = atom_embedding_size

    def call(self, inputs, **kwargs):
        assert int(inputs.shape[-1] % self.atom_embedding_size) == 0
        X = []
        start = 0
        end = 0
        arity = int(inputs.shape[-1] / self.atom_embedding_size)
        for i in range(arity):
            end += self.atom_embedding_size
            X.append(inputs[:,:,start:end])
            start = end
        return tf.reduce_sum(tf.reduce_prod(X, axis=0), axis=-1)
