import tensorflow as tf
from potentials import LogicPotential, SupervisionPotential, CountableGroundingPotential
import abc
import numpy as np

eps = 1e-12


class Train():

    def __init__(self, global_potential, parameters=None):
        self.global_potential = global_potential
        self.parameters = parameters

    @abc.abstractmethod
    def train(self, y, x=None):
        pass


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

        samples = self.samples = self.sampler.sample(x, minibatch=self.minibatch)

        if self.p_noise > 0:
            noise = tf.random_uniform(shape=y.shape)
            y = tf.where(noise > self.p_noise, y, 1 - y)

        if self.minibatch is not None:
            y = tf.gather(y, self.minibatch)
            if x is not None:
                x = tf.gather(x, self.minibatch)

        with tf.GradientTape(persistent=True) as tape:



            potentials_data = self.global_potential(y, x, training=True)

            potentials_samples = self.potentials_samples =  self.global_potential(samples, x, training=False)



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

            if isinstance(p, SupervisionPotential):

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


class PieceWiseTrainingv2(Train):

    def __init__(self, global_potential, parameters = None):
        super(PieceWiseTrainingv2, self).__init__(global_potential, parameters)
        self.global_potential = global_potential
        self.optimizer_nn = self.parameters["adam_nn"]
        self.optimizer_logic = self.parameters["adam_logic"]

    def compute_beta_logical_potentials(self, y, x=None, y_mask=None, evidence_mask= None, ):

        for p in self.global_potential.potentials:

            if isinstance(p, LogicPotential):

                if np.all(np.logical_or(y_mask, evidence_mask)):
                    ntrue, nfalse = p.true_false_assignments(x=x, evidence=y.astype(np.bool), evidence_mask=evidence_mask.astype(np.bool))
                    ntrue, nfalse = tf.reduce_sum(ntrue), tf.reduce_sum(nfalse)
                    g = p.ground(y=y, x=x)
                    phi_on_groundings = p.call_on_groundings(g,x)
                    avg_data = tf.reduce_mean(tf.cast(phi_on_groundings, tf.float32),axis=-1)
                    p.beta = tf.math.log(ntrue/nfalse) + tf.math.log(avg_data/(1 - avg_data))
                    if p.beta == np.inf:
                        p.beta = tf.Variable(100.)
                else:
                    ntrue_u, nfalse_u = p.true_false_assignments(evidence=y, evidence_mask=tf.logical_or(evidence_mask, y_mask))
                    ntrue_uo, nfalse_uo = p.true_false_assignments(x=None, evidence=y, evidence_mask=evidence_mask)

                    for i in range(10):
                        e_beta = tf.math.exp(p.beta)
                        grad = - tf.reduce_sum(((ntrue_u / (ntrue_u * e_beta + nfalse_u + eps)) - (
                                    ntrue_uo / (ntrue_uo * e_beta + nfalse_uo + eps)))*e_beta)
                        grad_var = [(grad, p.beta)]
                        self.optimizer_logic.apply_gradients(grad_var)

    def nn_w_update(self, y, x=None, soft_xent = False):
        """The method returns a training operation for maximizing the likelihood of the model."""

        for p in self.global_potential.potentials:

            if isinstance(p, SupervisionPotential):

                with tf.GradientTape(persistent=True) as tape:

                    y = p._reshape_y(y)
                    x = p._reshape_x(x)
                    o = p.model(x)
                    if not soft_xent:
                        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = o, labels=y))
                    else:
                        xent = tf.reduce_mean(- y * tf.log(tf.nn.softmax(o)))
                    xent += tf.reduce_sum(p.model.losses)

                grad = tape.gradient(target=xent, sources=p.model.variables)

                # Apply Gradients by means of Optimizer
                grad_vars = zip(grad, p.model.variables)
                self.optimizer_nn.apply_gradients(grad_vars)

    def train(self, y, x=None):

        # Train beta
        evidence_mask = self.parameters["evidence_mask"]
        train_mask = self.parameters["train_mask"]
        train_and_evidence_mask = np.logical_or(evidence_mask, train_mask)

        self.compute_beta_logical_potentials(y=np.array(y*train_and_evidence_mask).astype(np.bool), x=None, evidence_mask=np.array(evidence_mask).astype(np.bool), y_mask=np.array(train_mask).astype(np.bool))

        # Train w
        epochs = 10
        y_for_argmax = tf.cast(y, tf.int32)
        for e in range(epochs):
            self.nn_w_update(y, x=x)
            if self.parameters["evaluate"] and  e % 50 == 0:
                    nn = self.parameters["nn"]
                    x_test = self.parameters["x_test"]
                    indices_to_test = self.parameters["indices_to_test"]
                    y_test = tf.gather(y_for_argmax[0], indices_to_test)
                    y_nn = nn(x_test)
                    acc_nn = tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
                    print("Accuracy NN at epoch %d" % e, acc_nn)




PIECEWISE = "P13C3W153"



def create_training(id, P, parameters):
    if id == PIECEWISE:
        return PieceWiseTrainingv2(P, parameters)
    else:
        raise Exception("Training algorithm %s is not known." % str(id))




class EMTraining():
    def __init__(self, training, inference):

        self.training = training
        self.inference = inference

    def em(self, y, x=None, num_cycles=10):

        for em_step in range(num_cycles):
            self.training.train(y,x)
            y = self.inference.infer(x)


