from pyparsing import *
ParserElement.enablePackrat()
from collections import OrderedDict
from itertools import product
import numpy as np
import tensorflow as tf
class Variable():

    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

    def set_indices(self, indices):
        self.indices = indices

class Atom():

    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = args

    def set_indices(self, offset_range):
        base = offset_range
        for i,v in enumerate(self.args):
            next_domain_size = self.args[i+1].domain.num_constants if i<(len(self.args)-1) else 1
            base = base + v.indices * next_domain_size
        self.indices = base
        self.num_groundings = len(base)
        # self.indices = base if len(base.shape)>1 else np.reshape(base, [1, -1])

    def compile(self, herbrand_interpretation):
        if self.indices is None:
            raise Exception("Atom indices not set")
        t = tf.gather(herbrand_interpretation, self.indices, axis=-1) # todo this is the only point linked to tensorflow. If we want to make the compialtion dynamic we just provide the compilation function as parameter of the atom and to the constraint in turn

        return t


class Operator():

    def __init__(self, f, args):
        self.f = f
        self.args = args

    def compile(self, herbrand_interpretation):
        targs = []
        for a in self.args:
            cc = a.compile(herbrand_interpretation)
            targs.append(cc)
        return self.f(targs)


class Constraint(object):

    def __init__(self, ontology, formula, logic):
        self.ontology = ontology
        self.variables = OrderedDict()
        self.atoms = []
        self.logic = logic

        self.expression_tree = self.parse(formula)


        #Computing Variable indices
        sizes = []
        for i, (k, v) in enumerate(self.variables.items()):
            sizes.append(range(v.domain.num_constants))

        indices = [i for i in product(*sizes)]
        indices = np.array(indices)
        for i, (k, v) in enumerate(self.variables.items()):
            v.set_indices(indices[:,i])


        #Computing Atom indices
        for a in self.atoms:
            a.set_indices(self.ontology.predicate_range[a.predicate.name][0])

        #Num groundings of the formula = num grounding of a generic atom
        self.num_groundings = self.atoms[0].num_groundings


    def _create_or_get_variable(self, id, domain):
        if id in self.variables:
            assert self.variables[id].domain == domain, "Inconsistent domains for variables and predicates"
        else:
            v = Variable(id, domain)
            self.variables[id] = v
        return self.variables[id]

    def _createParseAction(self, class_name):
        def _create(tokens):

            if class_name == "Atomic":
                predicate_name = tokens[0]
                predicate = self.ontology.predicates[predicate_name]
                args = []
                for i, t in enumerate(tokens[1:]):
                    args.append(self._create_or_get_variable(t, predicate.domains[i]))
                a = Atom(predicate, args)
                self.atoms.append(a)
                return a
            elif class_name == "NOT":
                args = tokens[0][1:]
                return Operator(self.logic._not, args)
            elif class_name == "AND":
                args = tokens[0][::2]
                return Operator(self.logic._and, args)
            elif class_name == "OR":
                args = tokens[0][::2]
                return Operator(self.logic._or, args)
            elif class_name == "XOR":
                args = tokens[0][::2]
                return Operator(self.logic._xor, args)
            elif class_name == "IMPLIES":
                args = tokens[0][::2]
                return Operator(self.logic._implies, args)
            elif class_name == "IFF":
                args = tokens[0][::2]
                return Operator(self.logic._iff, args)
            # elif class_name == "FORALL":
            #     return forall(self.variables[tokens[1]], tokens[2][0])
            # elif class_name == "EXISTS":
            #     return Exists(constraint, tokens, self.world)
            # elif class_name == "EXISTN":
            #     return Exists_n(constraint, tokens, self.world)
            # elif class_name == "ARITHM_REL":
            #     # TODO
            #     raise NotImplementedError("Arithmetic Relations not already implemented")
            # elif class_name == "FILTER":
            #     parse_and_filter(constraint, tokens)

        return _create

    def parse(self, definition):

        left_parenthesis, right_parenthesis, colon, left_square, right_square = map(Suppress, "():[]")
        symbol = Word(alphas)

        ''' TERMS '''
        var = symbol
        # var.setParseAction(self._createParseAction("Variable"))

        ''' FORMULAS '''
        formula = Forward()
        not_ = Keyword("not")
        and_ = Keyword("and")
        or_ = Keyword("or")
        xor = Keyword("xor")
        implies = Keyword("->")
        iff = Keyword("<->")

        forall = Keyword("forall")
        exists = Keyword("exists")
        forall_expression = forall + symbol + colon + Group(formula)
        forall_expression.setParseAction(self._createParseAction("FORALL"))
        exists_expression = exists + symbol + colon + Group(formula)
        exists_expression.setParseAction(self._createParseAction("EXISTS"))

        relation = oneOf(list(self.ontology.predicates.keys()))
        atomic_formula = relation + left_parenthesis + delimitedList(var) + right_parenthesis
        atomic_formula.setParseAction(self._createParseAction("Atomic"))
        espression = forall_expression | exists_expression | atomic_formula
        formula << infixNotation(espression,
                                 [
                                     (not_, 1, opAssoc.RIGHT, self._createParseAction("NOT")),
                                     (and_, 2, opAssoc.LEFT, self._createParseAction("AND")),
                                     (or_, 2, opAssoc.LEFT, self._createParseAction("OR")),
                                     (xor, 2, opAssoc.LEFT, self._createParseAction("XOR")),
                                     (implies, 2, opAssoc.RIGHT, self._createParseAction("IMPLIES")),
                                     (iff, 2, opAssoc.RIGHT, self._createParseAction("IFF"))
                                 ])

        constraint = var ^ formula
        tree = constraint.parseString(definition, parseAll=True)
        return tree[0]

    def compile(self, herbrand_interpretation):
        if self.logic is None:
            raise Exception("Compilation is not allowed without any provided logic.")
        return self.expression_tree.compile(herbrand_interpretation)