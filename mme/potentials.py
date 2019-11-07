import tensorflow as tf
import numpy as np
import abc

class Potential(tf.keras.layers.Layer):

    id = -1
    states = {}

    @staticmethod
    def __newid__():
        Potential.id+=1
        return Potential.id

    def __init__(self):
        super(Potential, self).__init__()
        self.beta = tf.Variable("beta_%d"%Potential.__newid__(), shape=(), initializer=tf.zeros_initializer)


    @property
    def vars(self):
        return []

    @property
    @abc.abstractmethod
    def cardinality(self):
        return 1

    def compute_all_states(self):
        if self.cardinality not in Potential.states:
            Potential.states[self.cardinality] = np.array([[bool(i & (1<<k)) for k in range(self.cardinality)] for i in range(2**self.cardinality)])
        return Potential.states[self.cardinality]

    def __call__(self,y, x=None):
        return self.call(y,x)

    def variables(self):
        return [self.beta] + self.vars



class NeuralPotential(Potential):


    def __init__(self, model):
        super(NeuralPotential, self).__init__()
        self.model = model

    def call(self, y, x=None):
        if x is not None:
            y = tf.concat([y,x], axis=-1)
        return self.model(y)

    @property
    def vars(self):
        return self.model.variables


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

    @property
    def vars(self):
        return self.base_potential.vars


    def call(self, y, x=None):

        gamma_y, gamma_x = self.fragment(y,x)
        phi = self.base_potential(gamma_y, gamma_x)
        phi = tf.squeeze(phi, axis=-1)
        Phi = self.aggregate(phi)
        return Phi


class GlobalPotential():

    def __init__(self, potentials=()):

        self.potentials = list(potentials)

    def add(self, potential):
        self.potentials.append(potential)

    def __call__(self, y, x=None):
        res = tf.constant(0.)
        for Phi in self.potentials:
            res += Phi.beta * Phi(y,x)
        return res

    @property
    def variables(self):
        vars = []
        for Phi in self.potentials:
            vars.extend(Phi.variables())
        return vars


class LogicPotential(Potential):

    def __init__(self, constraint):
        super(LogicPotential, self).__init__()
        self.constraint = constraint

    def cardinality(self):
        return len([0 for i in self.constraint.atoms if not i.predicate.given])

    def __call__(self, y, x=None):
        y = tf.cast(y, dtype=tf.bool)
        t = self.constraint.compile(herbrand_interpretation=y)
        return tf.count_nonzero(t, axis=-1, dtype=tf.float32)





