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

    def __init__(self, constraint, logic):
        super(LogicPotential, self).__init__()
        self.constraint = constraint
        self.logic = logic

    @property
    def cardinality(self):
        return len([0 for i in self.constraint.atoms if not i.predicate.given])


    def __call__(self, y, x=None):
        t = self.constraint.compile(herbrand_interpretation=y, logic=self.logic)
        t = tf.cast(t, tf.float32)
        return tf.reduce_sum(t, axis=-1)


class SupervisionLogicalPotential(Potential):

    def __init__(self, model, predicates, ontology):
        super(SupervisionLogicalPotential, self).__init__()
        self.model = model
        model.add(tf.keras.layers.Dense(len(predicates), activation=None, use_bias=False))
        self.beta = tf.Variable(initial_value=tf.ones(shape=()))

        indices = []
        for p in predicates:
            fr = ontology.predicate_range[p.name][0]
            to = ontology.predicate_range[p.name][1]
            r = list(range(fr, to))
            indices.append(r)

        self.indices = np.stack(indices, axis=1)

    def _reshape_y(self,y ):
        y = tf.gather(y, self.indices, axis=1)
        return y

    def __call__(self, y, x=None):
        y = tf.cast(y, tf.float32)
        y = self._reshape_y(y)
        o = self.model(x)
        o =  tf.reshape(o, [tf.shape(x)[0], 1, -1])
        t = tf.reduce_sum(o*y, -1)
        # print(t)
        return t


