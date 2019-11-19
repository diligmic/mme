import unittest
from mme import Ontology, Domain, Predicate
import mme
from itertools import product
import numpy as np
import tensorflow as tf
import datasets


num_examples = 10
rng = np.arange(num_examples)
x_ids = np.reshape(np.tile(rng, [num_examples, 1]).T, [-1])
y_ids = np.tile(rng, [num_examples])

print(x_ids)
print(y_ids)
