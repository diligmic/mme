import tensorflow as tf
import numpy as np
import abc

class Potential(tf.Module):

    def __init__(self):
        super(Potential, self).__init__()
        self.beta = tf.Variable(initial_value=tf.zeros(shape=()))

    @property
    @abc.abstractmethod
    def cardinality(self):
        return 1

    def compute_all_states(self):
        if self.cardinality not in Potential.states:
            Potential.states[self.cardinality] = np.array([[bool(i & (1<<k)) for k in range(self.cardinality)] for i in range(2**self.cardinality)])
        return Potential.states[self.cardinality]

    def __call__(self, y, x=None):
        pass


class NeuralPotential(Potential):

    def __init__(self, model):
        super(NeuralPotential, self).__init__()
        self.model = model

    def __call__(self, y, x=None):
        if x is not None:
            y = tf.concat([y,x], axis=-1)
        return self.model(y)


class FragmentedPotential(Potential):

    def __init__(self, base_potential):
        super(FragmentedPotential, self).__init__()
        self.base_potential = base_potential

    @abc.abstractmethod
    def aggregate(self, phi):
        pass

    @abc.abstractmethod
    def fragment(self, y, x=None):
        return None, None

    def call(self, y, x=None):

        gamma_y, gamma_x = self.fragment(y,x)
        phi = self.base_potential(gamma_y, gamma_x)
        phi = tf.squeeze(phi, axis=-1)
        Phi = self.aggregate(phi)
        return Phi


class GlobalPotential(tf.Module):

    def __init__(self, potentials=()):

        self.potentials = list(potentials)

    def add(self, potential):
        self.potentials.append(potential)

    def __call__(self, y, x=None):
        res = tf.constant(0.)
        for Phi in self.potentials:
            res += Phi.beta * Phi(y,x)
        return res


class LogicPotential(Potential):

    def __init__(self, constraint):
        super(LogicPotential, self).__init__()
        self.constraint = constraint

    def cardinality(self):
        return len([0 for i in self.constraint.atoms if not i.predicate.given])

    def __call__(self, y, x=None):
        y = tf.cast(y, dtype=tf.bool)
        t = self.constraint.compile(herbrand_interpretation=y)
        return tf.math.count_nonzero(t, axis=-1, dtype=tf.float32)

class DotProductPotential(Potential):

    def __init__(self, model):
        super(DotProductPotential, self).__init__()
        self.model = model


    def __call__(self, y, x=None):
        y = tf.cast(y, tf.float32)
        o = self.model(x)
        o =  tf.reshape(o, [tf.shape(x)[0], 1, -1])
        t = tf.reduce_sum(o*y, -1)
        return t



