from .parser import Constraint
from collections import OrderedDict
import tensorflow as tf

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
        if d.name in self.domains:
            raise Exception("Domain %s already exists" % d.name)
        self.domains[d.name] = d

    def add_predicate(self, p):
        self.finalized = False
        if p.name in self.predicates:
            raise Exception("Predicate %s already exists" % p.name)
        self.predicates[p.name] = p
        self.predicate_range[p.name] = (self.herbrand_base_size,self.herbrand_base_size+p.groundings_number)
        self.herbrand_base_size += p.groundings_number


    def get_constraint(self,formula, logic=None):

        return Constraint(self, formula, logic)


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



class PiecewiseLearningModel():

    def __init__(self, global_potential, learning_rate=0.001, p_noise=0, num_samples=1, minibatch = None):
        self.p_noise = p_noise
        self.num_samples = num_samples
        self.global_potential = global_potential
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.minibatch = minibatch # list of indices to gather from data


    def maximize(self, y, x=None):
        """The method returns a training operation for maximizing the likelihood of the model."""

        if self.p_noise > 0:
            noise = tf.random_uniform(shape=y.shape)
            y = tf.where(noise > self.p_noise, y, 1 - y)

        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            if x is not None:
                x = tf.gather(x, self.minibatch)

        vars = self.global_potential.variables

        potentials_data = self.global_potential(y, x)
        avg_gradient_potential_data = [tf.convert_to_tensor(a) / tf.cast(tf.shape(y)[0], tf.float32) for a in
                                   tf.gradients(ys=potentials_data, xs=vars)]


        tf.gradients(ys=potentials_samples, xs=vars)

        # Compute Gradients
        vars = self.global_potential.variables



        E_gradient_potential = [tf.convert_to_tensor(a) / self.num_samples for a in
                                tf.gradients(ys=potentials_samples, xs=vars)]

        w_gradients = [b - a for a, b in zip(gradient_potential_data, E_gradient_potential)]
        # Apply Gradients by means of Optimizer
        grad_vars = zip(w_gradients, vars)
        train_op = self.optimizer.apply_gradients(grad_vars)

        return train_op




