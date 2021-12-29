import numpy as np
import tensorflow as tf
import collections
import pyparsing
import bisect
import itertools

import r2n.logic as logic


pyparsing.ParserElement.enablePackrat()


class Domain:
    def __init__(
        self,
        name: str,
        num_constants: int,
        constants: list = None,
        features: np.array = None,
    ):
        """Domain of constants in a FOL theory.

        TODO [extended_summary]

        Args:
            name (str):
                Name of the domain.
            num_constants (int):
                Number of domain's constants.
            constants (list, optional):
                List of domain's constants identifiers (str or int). Defaults to None.
            features (np.array, optional):
                Matrix of shape (num_constants, feature_size), where the i-th row
                represents the features of the i-th constant. Defaults to None.

        Raises:
            Exception:
                `name` should not be `None`.
        """

        if name is not None:
            self.name = str(name)
        else:
            raise Exception("Attribute 'name' is None.")
        self.num_constants = num_constants
        self.constants = constants

        # Map constants names to id (row index)
        self.constant_name_to_id = {c: i for i, c in enumerate(self.constants)}
        if constants is not None:
            assert self.num_constants == len(constants)
        if features is not None:
            assert features.shape[0] == len(constants)
            self.features = np.expand_dims(features, axis=0)
        else:
            self.features = np.expand_dims(
                np.eye(num_constants, dtype=np.float32), axis=0
            )

    def __hash__(self):
        return str(self.name).__hash__()


class Predicate:
    def __init__(self, name: str, domains: "list[Domain]", given: bool = False):
        """Relations in a FOL theory.

        TODO [extended_summary]

        Args:
            name (str):
                (Unique) name of the predicate.
            domains (list[Domain]):
                Positional list of domains.
            given (bool, optional): TODO. Defaults to False.

        Raises:
            Exception:
                Elements in `domains` must be of instance `Domain`.
        """
        self.name = name
        self.given = given

        self.domains = []
        groundings_number = 1
        for domain in domains:
            if not isinstance(domain, Domain):
                raise Exception(str(domain) + " is not an instance of " + str(Domain))
            self.domains.append(domain)
            groundings_number *= domain.num_constants
        self.groundings_number = groundings_number
        self.arity = len(self.domains)

    def __lt__(self, other):
        return self.name < other.name


class Ontology:
    def __init__(self, domains: "list[Domain]", predicates: "list[Predicate]"):
        """Multi-sorted FOL language.

        TODO [extended_summary]

        Args:
            domains (list[Domain]):
                List of constant domains of the ontology.
            predicates (list[Predicate]):
                List of predicates of the ontology.
        """
        self.domains = {}
        self._domain_list = []
        self.predicates = collections.OrderedDict()
        self.herbrand_base_size = 0
        self._predicate_range = collections.OrderedDict()
        self._range_to_predicate = RangeBisection()
        self.finalized = False
        self.constraints = []
        # Overall, number of elements in the assignment vector.
        self._linear_size = 0

        for d in domains:
            self.__add_domain(d)

        if len(domains) == 1:
            self.num_constants = domains[0].num_constants

        self.tuple_indices = {}
        for p in predicates:
            self.__add_predicate(p)

        self.__create_indexing_scheme()

        #  For some datasets, computing the indices of fragments is heavy. We store them.
        self.all_fragments_cache = {}

    def __str__(self):
        s = ""
        s += (
            "Domains (%d): " % len(self.domains)
            + ", ".join(
                [
                    "%s (%d)" % (name, domain.num_constants)
                    for name, domain in self.domains.items()
                ]
            )
            + "\n"
        )
        s += (
            "Predicates (%d):" % len(self.predicates)
            + ", ".join(self.predicates.keys())
            + "\n"
        )
        return s

    @staticmethod
    def read_ontology_from_file(file):
        import problog

        program = problog.program.PrologFile(file)
        predicates = {}
        constants = set()
        for fact in program:
            p, arity = str(fact.functor), len(fact.args)
            if p not in predicates:
                predicates[p] = arity
            else:
                assert (
                    predicates[p] == arity
                ), "Predicate {} arity inconsistency.".format(p)
            args = set([str(a.functor) for a in fact.args])
            if "-" in args:
                print()
            constants.update(args)
        return sorted(constants), predicates

    @staticmethod
    def from_file(file: str):
        """Instantiate a new ontology by reading from a file.

        TODO [extended_summary]

        Args:
            file (str):
                Filepath containing the knowledge base to be converted into an ontology.

        Returns:
            (Ontology):
                The `Ontology` object.
        """
        constants, predicates = Ontology.read_ontology_from_file(file)
        d = Domain(name="domain", constants=constants, num_constants=len(constants))
        predicates = [
            Predicate(p, domains=[d for _ in range(a)]) for p, a in predicates.items()
        ]
        return Ontology(domains=[d], predicates=predicates)

    def __check_multidomain(self):
        """
        Internal function to check if the FOL language is multi-sorted (i.e. multiple domains)
        """
        if len(self.domains) > 1:
            raise Exception("This operation does not allow multi domains")

    def __add_domain(self, d):
        if not isinstance(d, collections.abc.Iterable):
            D = [d]
        else:
            D = d
        for d in D:
            if d.name in self.domains:
                raise Exception("Domain %s already exists" % d.name)
            self.domains[d.name] = d
            self._domain_list.append(d)

    def __add_predicate(self, p):
        if not isinstance(p, collections.abc.Iterable):
            P = [p]
        else:
            P = p
        for p in P:
            if p.name in self.predicates:
                raise Exception("Predicate %s already exists" % p.name)
            self.predicates[p.name] = p
            self._predicate_range[p.name] = (
                self.herbrand_base_size,
                self.herbrand_base_size + p.groundings_number,
            )
            self._range_to_predicate[
                (
                    self.herbrand_base_size,
                    self.herbrand_base_size + p.groundings_number - 1,
                )
            ] = p.name
            self.herbrand_base_size += p.groundings_number
            k = tuple([d.name for d in p.domains])
            if k not in self.tuple_indices:
                # Cartesian product of the domains.
                ids = np.array(
                    [
                        i
                        for i in itertools.product(
                            *[range(self.domains[d].num_constants) for d in k]
                        )
                    ]
                )
                self.tuple_indices[k] = ids

    def __create_indexing_scheme(self):
        """
        Creates the indexing scheme used by the Ontology object for all the logic to tensor operations.
        """
        # Managing a linearized version of this logic
        self._up_to_idx = 0  # linear max indices
        self._dict_indices = (
            {}
        )  # mapping potentials id to correspondent multidimensional indices tensor

        self.finalized = False
        self._linear = None
        self._linear_evidence = None

        # Overall, number of elements in the assignment vector.
        self._linear_size = 0
        for p in self.predicates.values():
            # For unary predicates, this is just the domain size as [size]
            # For n-ary predicates, this is just the tensor of domain sizes [d1_size, d2_size, ...]
            shape = [d.num_constants for d in p.domains]
            # Overall domain size.
            predicate_domain_size = np.prod(shape)
            start_idx = self._up_to_idx
            end_idx = start_idx + predicate_domain_size
            self._up_to_idx = end_idx
            # print('Dict Indices', start_idx, end_idx)
            self._dict_indices[p.name] = np.reshape(
                np.arange(start_idx, end_idx), shape
            )
            self._linear_size += predicate_domain_size
        self.finalized = True

    def mask_by_file(self, file: str, dtype: tf.dtypes = tf.float32):
        mask = tf.zeros(self.linear_size(), dtype=dtype)
        ids = []
        with open(file) as f:
            for line in f:
                ids.append(self.atom_string_to_id(line))
        ids = tf.reshape(tf.constant(ids), shape=(len(ids), 1))
        mask = tf.tensor_scatter_nd_update(
            mask, tf.constant(ids), tf.ones(ids.shape[0])
        )
        return mask

    def mask_by_atom_strings(self, atom_strings: list, dtype: tf.dtypes = tf.float32):
        """Creates a mask from a list of atom strings.

        TODO [extended_summary]

        Args:
            atom_strings (list):
                List of atom strings.
            dtype (tf.Dtype):
                Output tensor dtype. Defaults to tf.float32.

        Returns:
            mask (tf.Tensor):
                Array of shape (self.linear_size()) having 1s mathing the indices of the
                atoms.
        """
        mask = tf.zeros(self.linear_size(), dtype=dtype)
        ids = []
        for atom in atom_strings:
            ids.append(self.atom_string_to_id(atom))
        ids = tf.reshape(tf.constant(ids), shape=(len(ids), 1))
        mask = tf.tensor_scatter_nd_update(
            mask, tf.constant(ids), tf.ones(ids.shape[0])
        )
        return mask

    def mask_by_constant(self, constants: list, negate: bool = False):
        """Creates a mask from a list of constants.

        Args:
            constants (list):
                List of constants ids.
            negate (bool):
                If True, the mask has 1s when not matching the constants.
                Defaults to False.

        Returns:
            mask (np.array):
                Array of shape (self.linear_size()) having 1s mathing the indices of the
                constants.
        """
        constant_set = frozenset(constants)
        if negate is False:
            mask = np.zeros(self.linear_size())
            non_default_value = 1
        else:
            mask = np.ones(self.linear_size())
            non_default_value = 0

        for i in range(self.linear_size()):
            data = self.id_to_predicate_constant_strings(i)
            atom_constants = data[1:]
            for c in atom_constants:
                if c in constant_set:
                    mask[i] = non_default_value
                    break
        return mask

    def mask_by_predicates(self, predicates: list):
        """Creates a mask from a list of constants and predicates.

        TODO [extended_summary]

        Args:
            predicates (list):
                List of `Predicates`.

        Returns:
            mask (np.array):
                Array of shape (self.linear_size()) having 1s mathing the indices of the
                combination of all the constants and the predicate.
        """
        mask = np.zeros([self.linear_size()])
        for p in predicates:
            a, b = self._predicate_range[p]
            mask[a:b] = 1.0
        mask = np.expand_dims(mask, 0)
        return mask

    def mask_by_constant_and_predicate(
        self, constants: list, predicates: list, negate_constants: bool = False
    ):
        """Creates a mask from a list of constants and predicates.

        TODO [extended_summary]

        Args:
            constants (list):
                List of constants ids.
            predicates (list):
                List of `Predicates`.
            negate_constants (bool, optional):
                If True, the mask has 1s when not matching the constants.
                Defaults to False.

        Returns:
            mask (np.array):
                Array of shape (self.linear_size()) having 1s mathing the indices of the
                combination of the constants and the predicate.
        """
        if predicates is None or not predicates:
            return self.mask_by_constant(constants, negate=negate_constants)

        constant_set = frozenset(constants)
        mask = np.zeros(self.linear_size())

        if negate_constants is False:
            non_default_value = 1
        else:
            non_default_value = 0

        for p in predicates:
            a, b = self._predicate_range[p]
            if negate_constants is True:
                mask[a:b] = 1
            for i in range(a, b):
                data = self.id_to_predicate_constant_strings(i)
                atom_constants = data[1:]
                for c in atom_constants:
                    if c in constant_set:
                        mask[i] = non_default_value
                        break
        return mask

    def ids_by_constant_and_predicate(self, constants, predicates):
        """

            Get all ids for a predicate of entries containing one of the constants.

        Args:
            constants:
            predicates:

        Returns:
            list of atom ids containing one of the constants

        """
        ids = []
        constant_set = frozenset(constants)
        for p in predicates:
            a, b = self._predicate_range[p]
            for i in range(a, b):
                data = self.id_to_predicate_constant_strings(i)
                atom_constants = data[1:]
                for c in atom_constants:
                    if c in constant_set:
                        ids.append(i)
                        break
        return ids

    def id_range_by_predicate(self, name):
        """
           Return (a,b) tuple, where [a,b[ is the interval of indices of atoms of predicate "name"
           in the linearized indexing.

        Args:
            name: predicate name

        Returns:
            a tuple (a,b) of indices

        """
        a, b = self._predicate_range[name]
        assert b - a > 0, "Can not find predicate %s" % name
        return a, b

    def linear_to_fol_dictionary(self, linear_state):
        """
            Create a dictionary mapping predicate names to np.array. For each key-value pair, the "value" of the
            dictionary array is the adiacency matrix of the predicate with name "key".

        Args:
            linear_state: a np.array with shape [self.linear_size()]

        Returns:
            a dictionary mapping predicate names to np.array

        """
        d = collections.OrderedDict()
        for p in self.predicates.values():
            d[p.name] = np.take(linear_state, self._dict_indices[p.name])
        return d

    def fol_dictionary_to_linear(self, dictionary):
        """

            Gets an input dictionary, mapping names to np.array. Return a concatenated linear version of all the values
            of the dictionary. This function is the inverse of Ontology.linear_to_fol_dictionary.

        Args:
            dictionary: a dictionary mapping predicate names to np.array

        Returns:
            a np.array

        """
        hb = np.zeros([self.linear_size()])
        for name in self.predicates:
            if name not in dictionary:
                raise Exception(
                    "%s predicate array is not provided in the dictionary." % name
                )
            array = dictionary[name]
            a, b = self._predicate_range[name]
            try:
                hb[a:b] = np.reshape(array, [-1])
            except:
                array = array.todense()
                hb[a:b] = np.reshape(array, [-1])
        return np.reshape(hb, [1, self.linear_size()])

    def fol_dictionary_to_linear_tf(self, dictionary, axis=1):
        """

            Return a concatenation of the keys of the dictionary along a specified "axis.

        Args:
            dictionary: a dictionary mapping predicate names to arrays
            axis: the axis of the concatenation. Defaults to 1.

        Returns:
            a concatenation of the keys of the dictionary along a specified "axis". The order of the concatenation is the
            iteration order of Ontology.predicates.

        """
        import tensorflow as tf

        # todo merge with the previous (numpy) one in some way.
        hb = []
        for name in self.predicates:
            if name not in dictionary:
                raise Exception(
                    "%s predicate array not provided in the dictionary." % name
                )
            array = dictionary[name]
            hb.append(array)
        res = tf.concat(hb, axis=axis)
        # assert res.shape[0] == ontology.linear_size()
        return res

    def id_to_atom(self, id_atom):
        predicate_name = self._range_to_predicate[id_atom]
        shape = self._dict_indices[predicate_name].shape
        ids = np.unravel_index(
            id_atom - self._predicate_range[predicate_name][0], shape
        )
        return predicate_name, ids

    def id_to_atom_string(self, id_atom):
        p_name, cs = self.id_to_atom(id_atom)
        p = self.predicates[p_name]
        return p_name + "(%s)" % ",".join(
            [p.domains[i].constants[c] for i, c in enumerate(cs)]
        )

    def id_to_predicate_constant_strings(self, id_atom):
        p_name, cs = self.id_to_atom(id_atom)
        p = self.predicates[p_name]
        return [p_name] + [p.domains[i].constants[c] for i, c in enumerate(cs)]

    def atom_string_to_id(self, atom):
        predicate, constants = atom_parser(atom)
        p = self.predicates[predicate]
        constants_ids = tuple(
            p.domains[i].constant_name_to_id[c] for i, c in enumerate(constants)
        )
        return self.atom_to_id(predicate, constants_ids)

    def atom_to_id(self, predicate_name, constant_ids):
        return self._dict_indices[predicate_name][tuple(constant_ids)]

    def linear_size(self):
        return self._linear_size

    def sample_fragments_idx(self, k, num=100, get_ids=False):
        self.__check_multidomain()
        ii = []
        all_ids = []
        for _ in range(num):
            i = []
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

    def all_fragments_idx_wrong(
        self, k, get_ids=False, get_atom_to_fragments_mask=False
    ):
        self.__check_multidomain()
        if k in self.all_fragments_cache is not None:
            groundings_hb_indices, indices = self.all_fragments_cache[k]
        else:
            num_constants = list(self.domains.values())[0].num_constants
            indices = np.array(list(itertools.permutations(range(num_constants), r=k)))
            groundings_hb_indices = []
            for i, (name, predicate) in enumerate(self.predicates.items()):
                predicate_range = self._predicate_range[name]
                size = predicate.domains[0].num_constants
                for j in range(k):
                    groundings_hb_indices.append(
                        predicate_range[0] + size * indices[:, j : j + 1] + indices
                    )

            groundings_hb_indices = np.concatenate(groundings_hb_indices, axis=1)
            self.all_fragments_cache[k] = groundings_hb_indices, indices
        if get_ids:
            to_return = groundings_hb_indices, indices
        else:
            to_return = groundings_hb_indices
        return to_return

    def all_fragments_idx(self, k, get_ids=False, get_atom_to_fragments_mask=False):

        ii = []
        all_ids = []
        num_constants = list(self.domains.values())[0].num_constants
        for idx in itertools.permutations(range(num_constants), k):
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
        to_return = res
        if get_ids:
            to_return = [res, np.stack(all_ids, axis=0)]

        if get_atom_to_fragments_mask:
            atom_to_fragments_mask = np.zeros([self.linear_size(), len(res)])
            for i in range(len(res)):
                for j in range(len(res[0])):
                    atom_id = res[i, j]
                    atom_to_fragments_mask[atom_id, i] = 1
            to_return = to_return + [atom_to_fragments_mask]

        return to_return

    def one_factors(self, k=3, return_pairs=False):

        if self.num_constants % 2 == 0:
            n = self.num_constants
            odd = False
        else:
            n = self.num_constants + 1
            odd = True

        """Creating the indices for the one-factors"""
        A = []
        r = np.arange(n // 2)
        r2 = np.arange(n // 2, n - 1)
        r2 = r2[::-1]
        for i in range(n - 1):
            rr = np.mod(r + i, n - 1)
            rr2 = np.mod(r2 + i, n - 1)
            rr2 = np.concatenate(([n - 1], rr2), axis=0)
            a = np.stack((rr, rr2), axis=1)
            A.append(a)
        A = np.stack(A, axis=0)

        """Now I create a map between a pair of indices in a factorization and its correspondent k-factor """
        idx, ids = self.all_fragments_idx(k, get_ids=True)
        d = collections.OrderedDict()
        for j, id in enumerate(ids):
            for l, k in itertools.permutations(id, 2):
                if (l, k) not in d:
                    d[(l, k)] = []
                d[(l, k)].append(j)

        # Now I create a vector like the map
        B = []
        for of in A:
            C = []
            for a, b in of:
                C.append(d[tuple([a, b])])
            B.append(C)
        B = np.array(B)

        """Now I need the indices of the interpretations of each of the factorizations"""
        idx, ids = self.all_fragments_idx(2, get_ids=True)
        d = collections.OrderedDict()
        for j, (l, k) in enumerate(ids):
            d[(l, k)] = idx[j]

        # Now I create a vector like the map
        D = []
        for of in A:
            C = []
            for a, b in of:
                C.append(d[tuple([a, b])])
            D.append(C)
        D = np.array(D)

        if not return_pairs:
            return D, B
        else:
            return A, D, B

    def size_of_fragment_state(self, k):
        self.__check_multidomain()
        size = 0
        for p in self.predicates.values():
            size += k ** p.arity
        return size


class Node:
    def __init__(self):
        self.name = "node"
        self.args = []


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class Variable(Node):
    def __init__(self, name, domain):
        super().__init__()
        self.name = name
        self.domain = domain

    def set_indices(self, indices):
        self.indices = indices


class Atom(Node):
    def __init__(self, predicate, args, idx):
        super().__init__()
        self.predicate = predicate
        self.name = predicate.name
        self.args = args
        self.idx = idx

    def set_indices(self, offset_range):
        base = offset_range
        for i, v in enumerate(self.args):
            next_domain_size = (
                self.args[i + 1].domain.num_constants if i < (len(self.args) - 1) else 1
            )
            base = base + v.indices * next_domain_size
        self.indices = base
        self.num_groundings = len(base)

    def ground(self, herbrand_interpretation, formula_filter=None):
        ground_indices = self.indices
        if ground_indices is None:
            raise Exception(f"Atom indices not set: {self.name}")

        if formula_filter is not None:
            if not isinstance(formula_filter, Formula):
                raise Exception(
                    f"Formula filter must be of instance Formula: {formula_filter}"
                )
            groundings = formula_filter.ground(herbrand_interpretation)
            filter_indices = np.squeeze(
                formula_filter.compile(groundings, logic.BooleanLogic).numpy()
            )
            ground_indices = self.indices[filter_indices > 0]

        if isinstance(herbrand_interpretation, np.ndarray):
            e = np.expand_dims(
                np.take(herbrand_interpretation, ground_indices, axis=-1), axis=-1
            )
            return e
        # TODO(giuseppe): This is the only point linked to tensorflow.
        # If we want to make the compilation dynamic we just provide the
        # compilation function as parameter of the atom and to the
        # constraint in turn.
        return tf.expand_dims(
            tf.gather(herbrand_interpretation, ground_indices, axis=-1), axis=-1
        )

    def compile(self, groundings):
        n = len(groundings.shape)
        start = np.zeros([n], dtype=np.int32)
        size = -1 * np.ones([n], dtype=np.int32)
        start[-1] = int(self.idx)
        size[-1] = 1
        sl = tf.squeeze(tf.slice(groundings, start, size), axis=-1)
        # sl2 = groundings[:, :, :, ontology.idx]
        return sl

    def evaluate(self, values):
        return tf.squeeze(values[:, :, self.idx], axis=0)


class Operator(Node):
    def __init__(self, f, args, name):
        super().__init__()
        self.f = f
        self.name = name
        self.args = args

    def compile(self, groundings):
        targs = []
        for a in self.args:
            cc = a.compile(groundings)
            targs.append(cc)
        return self.f(targs)

    def evaluate(self, values):
        targs = []
        for a in self.args:
            targs.append(a.evaluate(values))
        return self.f(targs)


class Formula(object):
    def __init__(self, ontology, definition, variables=None, hard: bool = None):
        self.ontology = ontology
        self.variables = collections.OrderedDict()
        self.atoms = []
        self.logic = None
        if variables is not None:
            self.variables = variables
        self.expression_tree = self.parse(definition)
        self.definition = definition
        self.hard = hard

        # Computing Variable indices
        if variables is None:
            sizes = []
            for i, (k, v) in enumerate(self.variables.items()):
                sizes.append(np.arange(v.domain.num_constants))

            # Cartesian Product
            # indices = [i for i in itertools.product(*sizes)] #TODO consider using tiling + stacking to see if it improves performances
            # indices = np.array(indices)
            indices = cartesian_product(*sizes)

            for i, (k, v) in enumerate(self.variables.items()):
                v.set_indices(indices[:, i])

        # Computing Atom indices
        for i, a in enumerate(self.atoms):
            a.set_indices(self.ontology._predicate_range[a.predicate.name][0])

        # Num groundings of the formula = num grounding of a generic atom
        self.num_groundings = self.atoms[0].num_groundings

        self.num_given = sum([1 for a in self.atoms if a.predicate.given])

    def arity(self):
        if self.variables is not None:
            return len(self.variables)
        return 0

    def num_atoms(self):
        return len(self.atoms)

    def is_hard(self):
        return self.hard

    def grounding_indices(self, filter=None, herbrand_interpretation=None):
        indices = tf.stack([a.indices for a in self.atoms], axis=-1)
        if filter is not None and herbrand_interpretation is not None:
            formula_filter = Formula(self.ontology, filter, variables=self.variables)
            groundings = formula_filter.ground(herbrand_interpretation)
            filter = tf.squeeze(formula_filter.compile(groundings, logic.BooleanLogic))
            indices = tf.boolean_mask(indices, filter)

        return indices

    def all_assignments_to_a_grounding(self):
        # this corresponds to 1 sample, 1 grounding, 2^n possible assignments, n values of a single assignment [1,1, 2^n, n]
        n = len(self.atoms)
        l = list(itertools.product([True, False], repeat=n))
        return np.array(l)

    def all_sample_groundings_given_evidence(self, evidence, evidence_mask):

        y_e = self.ground(herbrand_interpretation=evidence)
        y_e = tf.squeeze(
            y_e, axis=-2
        )  # removing the grounding assignments dimension since we will add it here
        m_e = self.ground(herbrand_interpretation=evidence_mask)
        m_e = tf.squeeze(
            m_e, axis=-2
        )  # removing the grounding assignments dimension since we will add it here

        n_examples = len(y_e)
        n_groundings = len(y_e[0])
        n_variables = len(self.atoms)
        k = n_variables - self.num_given
        n_assignments = 2 ** k

        shape = [n_variables, n_examples, n_groundings, 2 ** k]

        indices = tf.where(m_e[0][0] > 0)
        given = tf.gather(y_e, tf.reshape(indices, [-1]), axis=-1)
        given = tf.transpose(given, [2, 1, 0])
        given = tf.reshape(given, [self.num_given, n_examples, n_groundings, 1])
        given = tf.cast(tf.tile(given, [1, 1, 1, 2 ** k]), tf.float32)
        first = tf.scatter_nd(shape=shape, indices=indices, updates=given)

        indices = tf.where(m_e[0][0] < 1)
        l = list(itertools.product([False, True], repeat=k))
        comb = np.stack(l, axis=1).astype(np.float32)
        assignments = np.tile(
            np.reshape(comb, [-1, 1, 1, 2 ** k]), [1, n_examples, n_groundings, 1]
        )
        second = tf.scatter_nd(shape=shape, indices=indices, updates=assignments)

        final = tf.transpose(first + second, [1, 2, 3, 0])
        return final

    def _create_or_get_variable(self, id, domain):
        if id in self.variables:
            assert (
                self.variables[id].domain == domain
            ), "Inconsistent domains for variables and predicates"
        else:
            v = Variable(id, domain)
            self.variables[id] = v
        return self.variables[id]

    def _parse_action(self, class_name):
        def _create(tokens):

            if class_name == "Atomic":
                predicate_name = tokens[0]
                predicate = self.ontology.predicates[predicate_name]
                args = []
                for i, t in enumerate(tokens[1:]):
                    args.append(self._create_or_get_variable(t, predicate.domains[i]))
                a = Atom(predicate, args, len(self.atoms))
                self.atoms.append(a)
                return a
            elif class_name == "NOT":
                args = tokens[0][1:]
                return Operator(lambda x: self.logic._not(x), args, name="NOT")
            elif class_name == "and":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._and(x), args, name="and")
            elif class_name == "OR":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._or(x), args, name="OR")
            elif class_name == "XOR":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._xor(x), args, name="XOR")
            elif class_name == "IMPLIES":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._implies(x), args, name="IMPLIES")
            elif class_name == "IFF":
                args = tokens[0][::2]
                return Operator(lambda x: self.logic._iff(x), args, name="IFF")

        return _create

    def parse(self, definition):

        left_parenthesis, right_parenthesis, colon, left_square, right_square = map(
            pyparsing.Suppress, "():[]"
        )
        symbol = pyparsing.Word(pyparsing.alphas)

        """ TERMS """
        var = symbol
        # var.setParseAction(ontology._createParseAction("Variable"))

        """ FORMULAS """
        formula = pyparsing.Forward()
        not_ = pyparsing.Keyword("not")
        and_ = pyparsing.Keyword("and")
        or_ = pyparsing.Keyword("or")
        xor = pyparsing.Keyword("xor")
        implies = pyparsing.Keyword("->")
        iff = pyparsing.Keyword("<->")

        forall = pyparsing.Keyword("forall")
        exists = pyparsing.Keyword("exists")
        forall_expression = forall + symbol + colon + pyparsing.Group(formula)
        forall_expression.setParseAction(self._parse_action("FORALL"))
        exists_expression = exists + symbol + colon + pyparsing.Group(formula)
        exists_expression.setParseAction(self._parse_action("EXISTS"))

        relation = pyparsing.oneOf(list(self.ontology.predicates.keys()))
        atomic_formula = (
            relation
            + left_parenthesis
            + pyparsing.delimitedList(var)
            + right_parenthesis
        )
        atomic_formula.setParseAction(self._parse_action("Atomic"))
        espression = forall_expression | exists_expression | atomic_formula
        formula << pyparsing.infixNotation(
            espression,
            [
                (not_, 1, pyparsing.opAssoc.RIGHT, self._parse_action("NOT")),
                (and_, 2, pyparsing.opAssoc.LEFT, self._parse_action("and")),
                (or_, 2, pyparsing.opAssoc.LEFT, self._parse_action("OR")),
                (xor, 2, pyparsing.opAssoc.LEFT, self._parse_action("XOR")),
                (implies, 2, pyparsing.opAssoc.RIGHT, self._parse_action("IMPLIES")),
                (iff, 2, pyparsing.opAssoc.RIGHT, self._parse_action("IFF")),
            ],
        )

        constraint = var ^ formula
        tree = constraint.parseString(definition, parseAll=True)
        return tree[0]

    def compile(self, groundings, logic=logic.BooleanLogic):
        self.logic = logic
        t = self.expression_tree.compile(groundings)
        self.logic = None
        return t

    def ground(self, herbrand_interpretation, filter=None):
        formula_filter = (
            Formula(self.ontology, filter, variables=self.variables)
            if filter is not None
            else None
        )

        if isinstance(herbrand_interpretation, np.ndarray):
            return np.stack(
                [a.ground(herbrand_interpretation, formula_filter) for a in self.atoms],
                axis=-1,
            )
        else:
            return tf.stack(
                [a.ground(herbrand_interpretation, formula_filter) for a in self.atoms],
                axis=-1,
            )

    def evaluate(self, values, logic=logic.LukasiewiczLogic):
        self.logic = logic
        t = self.expression_tree.evaluate(values)
        self.logic = None
        return t


class GroundedFormula(object):
    """Struct to collect info about a formula."""

    def __init__(self, formula, filter, grounding_indices, evaluation=None):
        self.formula = formula
        self.filter = filter
        self.grounding_indices = grounding_indices
        self.evaluation = evaluation

    def size(self):
        return len(self.grounding_indices)

    def num_atoms(self):
        return self.formula.num_atoms()

    def is_evaluated(self):
        return self.evaluation is not None

    def get_cliques_masks_and_labels(
        self, train_mask: tf.Tensor, herbrand_interpretation: tf.Tensor
    ):
        """Generate the mask and the labels for the cliques.

        Given the training mask and the supervision of the atoms which are known to be
        true, for each formula under consideration we can build the cliques which are
        composed by atoms belonging to the training mask.
        For hard rules (those that are known to be always valid (true or false)), we can
        use also atoms that are not known (not belonging to knowledge of task) and that
        are properly filtered to respect the "type" of the constants.
        For soft rules (those that are not always valid, or uncertain), we must always
        use atoms that are supervised in order to retrieve the proper clique
        supervision (0/1).

        TODO: if works only for hard rules.

        Args:
            train_mask (tf.Tensor):
                Tensor of shape (1, ontology.linear_size()) having 1s in correspondence of
                the indices of the atoms used to select the cliques in which they appear.
            herbrand_interpretation (tf.Tensor):
                Tensor of shape (1, ontology.linear_size()) having 1s in correspondence of
                the indices of the atoms which have positive (1) supervision.

        Returns:
            cliques_mask (tf.Tensor):
                Mask for the cliques of the grounded formula.
                None if formula is not hard.
            cliques_labels (tf.Tensor):
                Labels for the selected cliques of the grounded formula.
                None if formula is not hard.
        """
        assert self.formula.is_hard(), f"Formula {self.formula.definition} is not hard."

        # Assign to each atom of the groundings the value True if it belongs to the train_mask
        train_groundings_mask = tf.cast(
            tf.gather(params=train_mask, indices=self.grounding_indices, axis=-1),
            tf.bool,
        )
        # Assign to each atom of the groundings its truth value, based on the hb
        groundings = tf.gather(
            params=herbrand_interpretation,
            indices=self.grounding_indices,
            axis=-1,
        )
        # Compute the cliques mask, by selecting the cliques having all atoms in train_mask
        formula_cliques_mask = tf.cast(
            tf.reduce_all(train_groundings_mask, axis=-1),
            tf.float32,
        )
        # Compute the cliques truth values (valid for hard rules)
        formula_cliques_labels = tf.cast(
            self.formula.compile(groundings, logic.BooleanLogic),
            tf.float32,
        )

        return formula_cliques_mask, formula_cliques_labels


def atom_parser(atom_string):
    symbol = pyparsing.Word(pyparsing.alphanums + "_")
    left_parenthesis, right_parenthesis, colon, left_square, right_square, dot = map(
        pyparsing.Suppress, "():[]."
    )
    parser_atom = (
        symbol
        + left_parenthesis
        + pyparsing.delimitedList(symbol)
        + right_parenthesis
        + pyparsing.Optional(dot)
    )
    tokens = parser_atom.parseString(atom_string)
    return tokens[0], tokens[1:]


class RangeBisection(collections.abc.MutableMapping):
    """Map ranges to values

    Lookups are done in O(logN) time. There are no limits set on the upper or
    lower bounds of the ranges, but ranges must not overlap.

    """

    def __init__(self, map=None):
        self._upper = []
        self._lower = []
        self._values = []
        if map is not None:
            self.update(map)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, point_or_range):
        if isinstance(point_or_range, tuple):
            low, high = point_or_range
            i = bisect.bisect_left(self._upper, high)
            point = low
        else:
            point = point_or_range
            i = bisect.bisect_left(self._upper, point)
        if i >= len(self._values) or self._lower[i] > point:
            raise IndexError(point_or_range)
        return self._values[i]

    def __setitem__(self, r, value):
        lower, upper = r
        i = bisect.bisect_left(self._upper, upper)
        if i < len(self._values) and self._lower[i] < upper:
            raise IndexError("No overlaps permitted")
        self._upper.insert(i, upper)
        self._lower.insert(i, lower)
        self._values.insert(i, value)

    def __delitem__(self, r):
        lower, upper = r
        i = bisect.bisect_left(self._upper, upper)
        if self._upper[i] != upper or self._lower[i] != lower:
            raise IndexError("Range not in map")
        del self._upper[i]
        del self._lower[i]
        del self._values[i]

    def __iter__(self):
        yield from zip(self._lower, self._upper)
