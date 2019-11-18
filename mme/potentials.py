import tensorflow as tf
import numpy as np
import abc



class Potential(tf.Module):

    def __init__(self):
        tf.Module.__init__(self)
        self.beta = tf.Variable(initial_value=tf.zeros(shape=()))

    @property
    @abc.abstractmethod
    def cardinality(self):
        return 1

    def __call__(self, y, x=None):
        pass

class CountableGroundingPotential(Potential):

    def __init__(self):
        super(CountableGroundingPotential, self).__init__()

    def ground(self, y, x=None):
        pass

    def call_on_groundings(self, y, x=None):
        pass

    def reduce_groundings(self, y):
        return tf.reduce_mean(y, axis=-1)

    def __call__(self, y, x=None):
        g,x_g = self.ground(y,x)
        g = self.call_on_groundings(g,x_g)
        r = self.reduce_groundings(g)
        return r

    @property
    def num_groundings(self):
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
        super(GlobalPotential, self).__init__()
        self.potentials = list(potentials)

    def add(self, potential):
        self.potentials.append(potential)

    def __call__(self, y, x=None):
        res = 0.
        for Phi in self.potentials:
            n = Phi.beta * Phi(y,x)
            res = res + n
        return res


    def save(self, path):
        print(self.variables)
        ckpt = tf.train.Checkpoint(obj = self)
        ckpt.save(path)

    def restore(self, path):

        ckpt = tf.train.Checkpoint(obj = self)
        ckpt.restore(path)


class LogicPotential(CountableGroundingPotential):

    def __init__(self, formula, logic):
        super(LogicPotential, self).__init__()
        self.formula = formula
        self.logic = logic

    @property
    def cardinality(self):
        return len([0 for i in self.formula.atoms if not i.predicate.given])

    @property
    def num_grounding(self):
        return self.formula.num_groundings

    def ground(self, y, x=None):
        if y is not None:
            groundings = self.formula.ground(herbrand_interpretation=y) # num_examples, num_groundings, 1, num_variables_in_grounding
        else:
            groundings = self.formula.all_grounding_assignments()
        return groundings, x

    def call_on_groundings(self, y, x=None):
        t = self.formula.compile(groundings=y, logic=self.logic) # num_examples, num_groundings, num_possible_assignment_to_groundings
        t = tf.cast(t, tf.float32)
        return tf.reduce_sum(t, axis=-1)

    @property
    def num_groundings(self):
        return self.formula.num_groundings



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
        y = self._reshape_y(y) # num_examples x num_groundings x num_variable_in_grounding
        o = self.model(x)
        o =  tf.reshape(o, [y.shape[0], x.shape[-2], -1])
        t = tf.reduce_mean(o*y, axis=tf.range(len(y.shape))[1:])
        return t



class MutualExclusivityPotential(CountableGroundingPotential):

    def __init__(self, indices):
        super(MutualExclusivityPotential, self).__init__()
        self.indices = indices

    @property
    def cardinality(self):
        return len(self.indices[0])

    @property
    def num_groundings(self):
        return len(self.indices)

    def ground(self, y, x=None):
        if y is not None:
            g = tf.gather(y, self.indices, axis=-1)
        else:
            g = None
        return g,x

    def call_on_groundings(self, y, x=None):
        if y is None:
            return self.cardinality * tf.ones([1, self.num_groundings])
        else:
            y = tf.cast(y, tf.float32)
        n = len(y.shape)-1
        # o_m_y = 1 - y
        y_exp = tf.expand_dims(1 - y, axis=-2) * (1 - tf.eye(self.cardinality))
        y_exp_p_1 = y_exp + tf.eye(self.cardinality)
        ya = tf.reduce_prod(y_exp_p_1, axis=-1)
        yya = y*ya
        t = 1 - yya
        y = 1 - tf.reduce_prod(t, axis=-1)
        if len(y.shape)>2:
            ax = tf.range(len(y.shape))[-(n-2):]
            return tf.reduce_sum(y, axis=ax)
        else:
            return y


class EvidenceLogicPotential(CountableGroundingPotential):

    def __init__(self, formula, logic, evidence, evidence_mask):
        super(EvidenceLogicPotential, self).__init__()
        self.formula = formula
        self.logic = logic
        self.evidence = evidence
        self.evidence_mask = evidence_mask

    @property
    def cardinality(self):
        return len([0 for i in self.formula.atoms if not i.predicate.given])

    @property
    def num_groundings(self):
        return self.formula.num_groundings

    def ground(self, y, x=None):
        if y is not None:
            groundings = self.formula.ground(herbrand_interpretation=y) # num_examples, num_groundings, 1, num_variables_in_grounding
        else:
            groundings = self.formula.all_sample_groundings_given_evidence(evidence=self.evidence, evidence_mask=self.evidence_mask)
        return groundings, x

    def call_on_groundings(self, y, x=None):
        y = self.logic.cast(y)
        t = self.formula.compile(groundings=y, logic=self.logic) # num_examples, num_groundings, num_possible_assignment_to_groundings
        t = tf.cast(t, tf.float32)
        return tf.reduce_sum(t, axis=-1)




