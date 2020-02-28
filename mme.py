#creating nested workspace
import logic
import potentials
import inference
import training
import utils
import ranking
from mme_parser import Formula
import tensorflow as tf
import numpy as np
from collections.abc import Iterable
from sortedcontainers import SortedDict
from itertools import permutations, combinations
import time

eps=1e-10
class Domain():

    def __init__(self, name, data):

        if name is not None:
            self.name = str(name)
        else:
            raise Exception("Attribute 'name' is None.")
        self.data = data
        self.constants = data #TODO clean this data vs constants. Data is a refuse
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
        self.arity = len(self.domains)


class Ontology():



    def __init__(self, domains, predicates):

        self.domains = {}
        self.predicates = SortedDict()
        self.herbrand_base_size = 0
        self.predicate_range = {}
        self.finalized = False
        self.constraints = []


        for d in domains:
            self.__add_domain(d)

        for p in predicates:
            self.__add_predicate(p)

        self.__create_indexing_scheme()

    def __check_multidomain(self):
        if len(self.domains)>1:
            raise Exception("This operation does not allow multi domains")

    def __add_domain(self, d):
        if not isinstance(d, Iterable):
            D = [d]
        else:
            D = d
        for d in D:
            if d.name in self.domains:
                raise Exception("Domain %s already exists" % d.name)
            self.domains[d.name] = d

    def __add_predicate(self, p):
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

    def __create_indexing_scheme(self):
        # Managing a linearized version of this logic
        self._up_to_idx = 0  # linear max indices
        self._dict_indices = {}  # map potentials id to correspondent multidimensional indices tensor

        self.finalized = False
        self._linear = None
        self._linear_evidence = None

        self._linear_size = 0
        for p in self.predicates.values():
            shape = [d.num_constants for d in p.domains]
            length = np.prod(shape)
            fr = self._up_to_idx
            to = fr + length
            self._up_to_idx = to
            self._dict_indices[p.name] = np.reshape(np.arange(fr, to), shape)
            self._linear_size += length
        self.finalized=True

    def get_constraint(self,formula):
        return Formula(self, formula)

    def FOL2LinearState(self, file):
        self.__check_multidomain()
        #just converting APIs from old NMLN
        pp = SortedDict({p.name: p.arity for p in self.predicates.values()})
        constants, predicates, evidences = utils.read_file_fixed_world(file, list(self.domains.values())[0].constants, pp)
        linear = []
        for p,v in predicates.items():
            linear.extend(np.reshape(v, [-1]))
        linear = np.reshape(linear, [1, -1])
        return linear

    def linear2Dict(self, linear_state):
        d = SortedDict()
        for p in self.predicates.values():
            d[p.name] = np.take(linear_state, self._dict_indices[p.name])
        return d

    def prettyPrintFromLinear(self, linear_state):
        for p in self.predicates.values():
            print(p)
            print(np.take(linear_state, self._dict_indices[p.name]))
            print()

    def linear_size(self):
        return self._linear_size

    def sample_fragments_idx(self, k, num=100, get_ids = False):
        self.__check_multidomain()
        ii = []
        all_ids = []
        for _ in range(num):
            i=[]
            num_constants = list(self.domains.values())[0].num_constants
            idx = np.random.choice(num_constants, size=k, replace=False)
            idx = np.random.permutation(idx)
            all_ids.append(idx)
            for p in self.predicates.values():
                a = p.arity
                f_idx = self._dict_indices[p.name]
                for j in range(a):
                    f_idx = np.take(f_idx, idx, axis=j)
                f_idx = np.reshape(f_idx, [-1])
                i.extend(f_idx)
            ii.append(i)
        res = np.stack(ii, axis=0)
        if not get_ids:
            return res
        else:
            return res, np.stack(all_ids, axis=0)

    def all_fragments_idx(self, k, get_ids = False, get_atom_to_fragments_mask=False):
        self.__check_multidomain()
        ii = []
        all_ids = []
        num_constants = list(self.domains.values())[0].num_constants
        for idx in permutations(range(num_constants), k):
            all_ids.append(idx)
            i = []
            for p in self.predicates.values():
                a = p.arity
                f_idx = self._dict_indices[p.name]
                for j in range(a):
                    f_idx = np.take(f_idx, idx, axis=j)
                f_idx = np.reshape(f_idx, [-1])
                i.extend(f_idx)
            ii.append(i)
        res = np.stack(ii, axis=0)

        atom_to_fragments_mask = np.zeros([self.linear_size(), len(res)])
        for i in range(len(res)):
            for j in range(len(res[0])):
                atom_id = res[i,j]
                atom_to_fragments_mask[atom_id, i] = 1

        to_return = res
        if get_ids:
            to_return = [res,np.stack(all_ids, axis=0)]
        if get_atom_to_fragments_mask:
            to_return = to_return+[atom_to_fragments_mask]
        return to_return

    def size_of_fragment_state(self, k):
        self.__check_multidomain()
        size = 0
        for p in self.predicates.values():
            size += k**p.arity
        return size


