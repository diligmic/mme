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
        res = 0.
        for Phi in self.potentials:
            n = Phi.beta * Phi(y,x)
            res = res + n
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
        # y.shape = [num_examples, num_variables]
        if y is not None:
            groundings = self.constraint.ground(herbrand_interpretation=y) # num_examples, num_groundings, 1, num_variables_in_grounding
        else:
            groundings = self.constraint.all_grounding_assignments()  # num_examples, num_groundings, num_possible_assignment_to_groundings, num_variables_in_grounding, num_groundings should be one if no evidence
        t = self.constraint.compile(groundings=groundings, logic=self.logic) # num_examples, num_groundings, num_possible_assignment_to_groundings
        t = tf.cast(t, tf.float32)
        #todo: handle aggregation like in supervision. in principle we could use only an optional (automatically grodunded and reduced) dimension for groundings assignments
        return tf.reduce_sum(tf.reduce_sum(t, axis=-1), axis=-1) # [num_examples]


class SupervisionLogicalPotential(Potential):

    def __init__(self, model, indices):
        super(SupervisionLogicalPotential, self).__init__()
        self.model = model
        self.beta = tf.Variable(initial_value=tf.ones(shape=()))
        self.indices = indices

    def _reshape_y(self,y ):
        y = tf.gather(y, self.indices, axis=-1)
        return y

    def __call__(self, y, x=None):
        y = tf.cast(y, tf.float32) # num_examples x num_variables
        n = len(y.shape)
        y = self._reshape_y(y) # num_examples x num_groundings x num_variable_in_grounding
        o = self.model(x)
        o =  tf.reshape(o, [y.shape[0], x.shape[0], -1])
        ax = tf.range(len(y.shape))[-(n-1):]
        t = tf.reduce_sum(o*y, axis=ax)
        return t







