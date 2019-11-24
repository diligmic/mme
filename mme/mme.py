from .parser import Formula
from collections import OrderedDict
import tensorflow as tf
from .potentials import LogicPotential, SupervisionLogicalPotential, CountableGroundingPotential
import numpy as np
from collections.abc import Iterable



class Domain():

    def __init__(self, name, data):

        if name is not None:
            self.name = str(name)
        else:
            raise Exception("Attribute 'name' is None.")
        self.data = data
        self.num_constants = len(data) #TODO check iterable


class Predicate():


    def __init__(self, name, domains, given=False):
        self.name = name

        self.domains = []
        groundings_number = 1
        for domain in domains:
            if not isinstance(domain, Domain):
                raise Exception(str(domain) + " is not an instance of " + str(Domain))
            self.domains.append(domain)
            groundings_number*=domain.num_constants
        self.groundings_number = groundings_number
        self.given = given


class Ontology():


    def __init__(self):

        self.domains = {}
        self.predicates = OrderedDict()
        self.herbrand_base_size = 0
        self.predicate_range = {}
        self.finalized = False
        self.constraints = []

    def add_domain(self, d):
        self.finalized = False
        if not isinstance(d, Iterable):
            D = [d]
        else:
            D = d
        for d in D:
            if d.name in self.domains:
                raise Exception("Domain %s already exists" % d.name)
            self.domains[d.name] = d

    def add_predicate(self, p):
        self.finalized = False
        if not isinstance(p, Iterable):
            P = [p]
        else:
            P = p
        for p in P:
            if p.name in self.predicates:
                raise Exception("Predicate %s already exists" % p.name)
            self.predicates[p.name] = p
            self.predicate_range[p.name] = (self.herbrand_base_size,self.herbrand_base_size+p.groundings_number)
            self.herbrand_base_size += p.groundings_number


    def get_constraint(self,formula):

        return Formula(self, formula)


class MonteCarloTraining():

    def __init__(self, global_potential, sampler, learning_rate=0.001, p_noise=0, num_samples=1, minibatch = None):
        self.p_noise = p_noise
        self.num_samples = num_samples
        self.global_potential = global_potential
        self.sampler = sampler
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.minibatch = minibatch # list of indices to gather from data


    def maximize_likelihood_step(self, y, x=None):
        """The method returns a training operation for maximizing the likelihood of the model."""

        samples = self.samples = self.sampler.sample(x, self.num_samples, minibatch=self.minibatch)

        if self.p_noise > 0:
            noise = tf.random_uniform(shape=y.shape)
            y = tf.where(noise > self.p_noise, y, 1 - y)

        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            if x is not None:
                x = tf.gather(x, self.minibatch)

        with tf.GradientTape(persistent=True) as tape:



            potentials_data = self.global_potential(y, x)

            potentials_samples = self.potentials_samples =  self.global_potential(samples, x)



        # Compute Gradients
        vars = self.global_potential.variables

        gradient_potential_data = [tf.convert_to_tensor(a) / tf.cast(tf.shape(y)[0], tf.float32) for a in
                                   tape.gradient(target=potentials_data, sources=vars)]

        E_gradient_potential = [tf.convert_to_tensor(a) / self.num_samples for a in
                                tape.gradient(target=potentials_samples, sources=vars)]

        w_gradients = [b - a for a, b in zip(gradient_potential_data, E_gradient_potential)]

        # Apply Gradients by means of Optimizer
        grad_vars = zip(w_gradients, vars)
        self.optimizer.apply_gradients(grad_vars)



class PieceWiseTraining():

    def __init__(self, global_potential, y=None, learning_rate=0.001, minibatch = None):
        self.global_potential = global_potential
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.minibatch = minibatch # list of indices to gather from data
        self.y = y


    def compute_beta_logical_potentials(self, y=None, x=None):

        if y is None:
            y=self.y
        for p in self.global_potential.potentials:

            if isinstance(p, CountableGroundingPotential):

                ntrue = p(y=None)
                # nfalse = (2**p.cardinality)*p.num_groundings - ntrue
                nfalse = (2**p.cardinality) - ntrue

                g,x = p.ground(y=y, x=x)
                phi_on_groundings = p.call_on_groundings(g,x)
                avg_data = tf.reduce_mean(tf.cast(phi_on_groundings, tf.float32),axis=-1)
                # avg_data = tf.abs(avg_data -  1e-7)
                p.beta = tf.math.log(ntrue/nfalse) + tf.math.log(avg_data/(1 - avg_data))
                if p.beta == np.inf:
                    p.beta = tf.Variable(100.)


    def maximize_likelihood_step(self, y, x=None, soft_xent = False):
        """The method returns a training operation for maximizing the likelihood of the model."""

        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            if x is not None:
                x = tf.gather(x, self.minibatch)


        for p in self.global_potential.potentials:

            if isinstance(p, SupervisionLogicalPotential):

                with tf.GradientTape(persistent=True) as tape:

                    y = p._reshape_y(y)
                    o = p.model(x)
                    if not soft_xent:
                        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = o, labels=y))
                    else:
                        xent = tf.reduce_mean(- y * tf.log(tf.nn.softmax(o)))
                    xent += tf.reduce_sum(p.model.losses)


                grad = tape.gradient(target=xent, sources=p.model.variables)

                # Apply Gradients by means of Optimizer
                grad_vars = zip(grad, p.model.variables)
                self.optimizer.apply_gradients(grad_vars)

