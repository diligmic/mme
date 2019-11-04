import tensorflow as tf
import tensorflow_probability as tfp
from .constants import *
import numpy as np


class Sampler(object):

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):
        pass

class SamplerAggregation(Sampler):


    def __init__(self, samplers):

        self.samplers = []
        for s in samplers:
            if isinstance(s, Sampler):
                self.samplers.append(s)

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        S = []
        for s in self.samplers:
            S.append(s.sample(conditional_data, num_samples, minibatch))

        return tf.stack(S, axis=0)

class NegativeSampler(object):

    def __init__(self, y):
        self.y = y

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        self.current_state = tf.cast(tf.random_uniform(shape=tf.shape(self.y), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        return self.current_state


class VariationalMAPSampler(Sampler):

    def __init__(self, name, global_potential, var_shape, convergence_threshold = 0.00001, max_num_iter = 1000, learning_rate = 0.1):
        self.global_potential = global_potential
        self.var_shape = var_shape
        self.var_ = tf.get_variable("MAP_state_{}".format(name), initializer=tf.random_normal(shape=var_shape, mean=0.0, stddev=0.1))
        # self.var_ = tf.get_variable("MAP_state_{}".format(name), initializer=0.5 * tf.ones(shape=var_shape))
        self.convergence_threshold = convergence_threshold
        self.max_num_iter = max_num_iter

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    def _internal_state(self):
        return tf.sigmoid(self.var_)

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        not_converged = lambda i:  tf.less(i, self.max_num_iter)
        def body(i):
            current_state = self._internal_state()
            if minibatch is not None:
                current_state = tf.gather(current_state, minibatch)
            update_map = self.optimizer.minimize(- self.global_potential(current_state, conditional_data), var_list=[self.var_])
            with tf.control_dependencies([update_map]):
                i = tf.add(i, 1)
                return i

        with tf.control_dependencies([tf.variables_initializer([self.var_])]):
            loop = tf.while_loop(not_converged, body, [tf.constant(0.)])
            with tf.control_dependencies([loop]):
                map = self.current_state = tf.stop_gradient(self._internal_state())
        if minibatch is not None:
            map = tf.gather(map, minibatch)
        return map



class VariationalMAPSamplerSigmoid(Sampler):

    def __init__(self, name, global_potential, var_shape, convergence_threshold = 0.00001, max_num_iter = 1000, learning_rate = 0.1):
        self.global_potential = global_potential
        self.var_shape = var_shape
        # self.var_ = tf.get_variable("MAP_state_{}".format(name), initializer=tf.random_normal(shape=var_shape, mean=0.0, stddev=0.1))
        self.var_ = tf.get_variable("MAP_state_{}".format(name), initializer=0.5 * tf.ones(shape=var_shape))
        self.convergence_threshold = convergence_threshold
        self.max_num_iter = max_num_iter

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    def _internal_state(self):
        # return tf.sigmoid(self.var_)
        return tf.identity(self.var_)

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        not_converged = lambda i:  tf.less(i, self.max_num_iter)
        def body(i):
            current_state = self._internal_state()
            if minibatch is not None:
                current_state = tf.gather(current_state, minibatch)
            update_map = self.optimizer.minimize(- self.global_potential(current_state, conditional_data), var_list=[self.var_])
            with tf.control_dependencies([update_map]):
                with tf.control_dependencies([tf.assign(self.var_, tf.clip_by_value(self.var_, clip_value_min=0, clip_value_max=1))]):
                    i = tf.add(i, 1)
                    return i



        # with tf.control_dependencies([tf.variables_initializer([self.var_])]):
        loop = tf.while_loop(not_converged, body, [tf.constant(0.)])
        with tf.control_dependencies([loop]):
            map = self.current_state = tf.stop_gradient(self._internal_state())
        if minibatch is not None:
            map = tf.gather(map, minibatch)
        return map


class VariationalBernoulli(Sampler):

    def __init__(self, name, global_potential, var_shape, convergence_threshold = 0.00001, max_num_iter = 1000, learning_rate = 0.1):
        self.global_potential = global_potential
        self.var_shape = var_shape
        self.var_ = tf.get_variable("state_{}".format(name), initializer=tf.random_normal(shape=var_shape, mean=0.0, stddev=0.1))
        self.convergence_threshold = convergence_threshold
        self.max_num_iter = max_num_iter

        self.optimizer = tf.train.AdamOptimizer(learning_rate)

    def _internal_state(self):
        #reparametrization trick
        eps = tf.random_uniform(shape=self.var_shape, minval=0, maxval=1, dtype=tf.float32)
        return eps * self.var_ / (eps * self.var_ + (1 - eps)*(1- self.var_))

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        i = tf.constant(0)

        converged = lambda i:  i < self.max_num_iter

        def body(i):

            current_state = self._internal_state()
            if minibatch is not None:
                current_state = tf.gather(current_state, minibatch)
            update = self.optimizer.minimize(- self.global_potential(current_state, conditional_data), var_list=[self.var_])
            with tf.control_dependencies([update]):
                i = i + 1
            return i

        loop = tf.while_loop(converged, body, (i))
        with tf.control_dependencies(loop):
            eps = tf.random_uniform(shape=self.var_shape, minval=0, maxval=1, dtype=tf.float32)
            sample = eps < tf.sigmoid(self.var_)
        if minibatch is not None:
            sample = tf.gather(sample, minibatch)
        return sample

class GenerativeNetworkSampler(Sampler):


    def __init__(self, name, global_potential, num_samples, num_variables, generative_model, learning_rate = 0.001, noise_dimensions = 10, is_test = False):
        self.global_potential = global_potential
        self.num_samples = num_samples
        self.model = generative_model
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.name = name
        self.noise_dimensions = noise_dimensions
        self.is_test = is_test

        self.num_chains = num_samples
        self.num_variables = num_variables

        self.current_state = tf.get_variable(name,
                                             initializer=tf.where(
                                                 tf.random_uniform(shape=[self.num_chains, num_variables], minval=0,
                                                                   maxval=1) > 0.5,
                                                 tf.ones([self.num_chains, num_variables]),
                                                 tf.zeros([self.num_chains, num_variables])),
                                             trainable=False
                                             )

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        rand = tf.random_uniform(shape=[self.num_samples])
        on_state = self.model(self.current_state)
        off_state = self.current_state

        potential_on = self.global_potential(on_state)
        potential_off = self.global_potential(self.current_state)
        p = tf.sigmoid(potential_on - potential_off)

        cond = rand < p
        new_state = tf.where(cond, on_state, off_state)


        ass = tf.assign(self.current_state, new_state)
        if not self.is_test:
            update_map = self.optimizer.minimize(- self.global_potential(new_state, conditional_data), var_list=self.model.variables)
            with tf.control_dependencies([ass, update_map]):
                new_state  = tf.stop_gradient(new_state)
        return new_state


class GPUGibbsSampler(Sampler):

    count = 0
    @staticmethod
    def _get_id():
        temp = GPUGibbsSampler.count
        GPUGibbsSampler.count +=1
        return temp


    def __init__(self, potential, num_variables, inter_sample_burn=0, num_chains = 10, initial_state=None, evidence = None, flips=None):

        self.potential = potential
        self.num_chains = num_chains
        self.inter_sample_burn = inter_sample_burn
        self.num_variables = num_variables
        name = "init_state_global_"+str(GPUGibbsSampler._get_id())
        if initial_state is None:
            self.current_state = tf.get_variable(name,
                                                     initializer=tf.where(
                                                         tf.random_uniform(shape=[self.num_chains, num_variables], minval=0,
                                                                           maxval=1) > 0.5,
                                                         tf.ones([self.num_chains, num_variables]),
                                                         tf.zeros([self.num_chains, num_variables])),
                                                     trainable=False
                                                     )
        else:
            self.current_state = tf.get_variable(name,
                                                 initializer=initial_state,
                                                 trainable=False
                                                 )
        self.evidence = evidence
        self.flips = flips

    def sample(self, conditional_data=None, num_samples=None, minibatch = None):

        if num_samples is not None:
            num_chain_temp = ((num_samples-1) // self.num_chains) + 1
        else:
            num_chain_temp = self.num_chains

        if self.flips is None:
            kernel = GPUGibbsKernel(self.potential, conditional_data, evidence_mask=self.evidence, inter_sample_burn=self.inter_sample_burn)

        else:
            kernel = PartialGPUGibbsKernel(self.potential, conditional_data, evidence_mask=self.evidence, flips=self.flips)

        current_state = self.current_state if minibatch is None else tf.gather(self.current_state, minibatch)

        mcmc_res = tfp.mcmc.sample_chain(num_chain_temp,
                                        current_state,
                                        num_burnin_steps=self.inter_sample_burn,
                                        kernel=kernel)
        samples = mcmc_res[0]
        if minibatch is None:
            ass = tf.assign(self.current_state, samples[-1])
        else:
            ass = tf.scatter_update(self.current_state, minibatch, samples[-1])
        with tf.control_dependencies([ass]):
            samples = tf.stop_gradient(samples)
        samples = tf.reshape(samples,[-1, self.num_variables])
        # return samples
        return tf.concat(samples,axis=0)[:num_samples]


class GPUGibbsKernel(tfp.mcmc.TransitionKernel):

    def __init__(self, potential, conditional_data, evidence_mask=None, inter_sample_burn=0):
        super(GPUGibbsKernel, self).__init__()
        self.potential = potential
        self.masks = []
        self.evidence_mask = evidence_mask if evidence_mask is not None else None
        self.conditional_data = conditional_data



    def one_step(self, current_state, previous_kernel_results):

        num_examples = current_state.get_shape()[0].value
        num_variables = current_state.get_shape()[1].value

        K = tf.random.shuffle(tf.range(num_variables, dtype=tf.int32))
        i = [tf.constant(0, dtype=tf.int32), current_state]
        c = lambda i, s: i < num_variables

        def body(i,s):
            k = K[i]
            mask = tf.one_hot(k, depth=num_variables)
            off_state = s * (1 - mask)
            on_state = off_state + mask
            rand = tf.random_uniform(shape=[num_examples])

            potential_on = self.potential(on_state, self.conditional_data)
            potential_off = self.potential(off_state, self.conditional_data)
            p = tf.sigmoid(potential_on - potential_off)
            # p = tf.Print(p, [p, rand])

            cond = rand < p
            current_state = tf.where(cond, on_state, off_state)
            if self.evidence_mask is not None:
                current_state = tf.cond(tf.equal(self.evidence_mask[0, k], 1), lambda: s, lambda: current_state)
            i = i + 1
            return i, current_state

        i, r= tf.while_loop(c, body, i)
        return r, []

    def is_calibrated(self):
        return True

    def bootstrap_results(self, init_state):
        return []


class PartialGPUGibbsKernel(tfp.mcmc.TransitionKernel):

    def __init__(self, potential, conditional_data, flips, evidence_mask=None):
        super(PartialGPUGibbsKernel, self).__init__()
        self.potential = potential
        self.masks = []
        self.evidence_mask = evidence_mask if evidence_mask is not None else None
        self.flips = flips
        self.conditional_data = conditional_data

    def one_step(self, current_state, previous_kernel_results):

        num_examples = current_state.get_shape()[0].value
        num_variables = current_state.get_shape()[1].value

        P = tf.zeros([num_examples])
        OFF = tf.constant(0.)
        ON = tf.constant(0.)
        DELTA = tf.constant(0.)
        i = [tf.constant(0, dtype=tf.int32),current_state,P, OFF, ON, DELTA]
        c = lambda i,s,p,off,on,delta: i < self.flips

        def body(i,s,P,OFF, ON, DELTA):
            if self.evidence_mask is not None:
                k = tf.squeeze(tf.multinomial(tf.log(1 - tf.cast(self.evidence_mask, tf.float32)), 1, output_dtype=tf.int32))
            else:
                k = tf.random_uniform(minval=0,
                                      maxval=num_variables,
                                      dtype=tf.int32,
                                      shape=())
            mask = tf.one_hot(k, depth=num_variables)
            off_state = s * (1 - mask)
            on_state = off_state + mask
            rand = tf.random_uniform(shape=[num_examples])

            #Time efficient
            potential_on = self.potential(on_state, x=self.conditional_data)
            potential_off = self.potential(off_state, x=self.conditional_data)
            delta_potential = tf.abs(potential_on - potential_off)
            p = tf.reshape(tf.sigmoid(potential_on - potential_off), [-1])
            #Memory Efficient
            # energy_on = self.model.compute_energy(on_state)
            # energy_off = self.model.compute_energy(off_state)
            # p = tf.reshape(tf.sigmoid(- energy_on + energy_off), [-1])

            cond = rand < p
            current_state = tf.where(cond, on_state, off_state)
            if self.evidence_mask is not None:
                current_state= tf.cond(tf.equal(self.evidence_mask[0, i],1), lambda: s, lambda: current_state)
            i = i+1
            OFF = OFF + tf.reduce_sum(potential_off)
            ON = ON + tf.reduce_sum(potential_on)
            DELTA = DELTA + tf.reduce_sum(delta_potential)
            return i,current_state, p, OFF, ON, DELTA

        i,r,p, OFF,ON, DELTA = tf.while_loop(c, body, i)
        return r, [p, OFF/(num_examples*self.flips),ON/(num_examples*self.flips), DELTA/(num_examples*self.flips)]

    def is_calibrated(self):
        return True

    def bootstrap_results(self, init_state):
        num_examples = init_state.get_shape()[0].value
        return [tf.zeros([num_examples]), tf.constant(0.), tf.constant(0.), tf.constant(0.)]