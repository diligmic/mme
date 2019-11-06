import unittest
from mme import Ontology, Domain, Predicate
import mme
from itertools import product
import numpy as np
import tensorflow as tf


class Test(unittest.TestCase):

    def test_new_domain(self):

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

    def test_duplicate_domain(self):
        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("People", data_universities)
        o.add_domain(people)
        try:
            o.add_domain(universities)
        except Exception as e:
            assert(str(e) == "Domain People already exists")

    def test_new_predicate(self):

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

        student = Predicate("student", domains=[people])
        advised_by = Predicate("advisorOf", domains=[people, people])
        member_of = Predicate("memberOf", domains=[people, universities])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(advised_by)
        o.add_predicate(member_of)
        o.add_predicate(married_with)

    def test_herbrand_base_dimensions(self):

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

        student = Predicate("student", domains=[people])
        advised_by = Predicate("advisorOf", domains=[people, people])
        member_of = Predicate("memberOf", domains=[people, universities])
        married_with = Predicate("marriedWith", domains=[people, people])

        """Herbrand Base size after adding student predicate"""
        o.add_predicate(student)
        assert(o.herbrand_base_size == len(data_people))

        """Herbrand Base size after adding advised"""
        o.add_predicate(advised_by)
        assert (o.herbrand_base_size == len(data_people)+len(data_people)*len(data_people))

        """Herbrand Base size after adding member of"""
        o.add_predicate(member_of)
        assert (o.herbrand_base_size == len(data_people)+len(data_people)*len(data_people) + len(data_people)*len(data_universities))

        """Herbrand Base size after adding married with"""
        o.add_predicate(married_with)
        assert (o.herbrand_base_size == len(data_people) + len(data_people) * len(data_people) + len(data_people) * len(
            data_universities) + len(data_people)*len(data_people))

    def test_variables_and_atom_indices(self):

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

        student = Predicate("student", domains=[people])
        advised_by = Predicate("advisorOf", domains=[people, people])
        member_of = Predicate("memberOf", domains=[people, universities])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(advised_by)
        o.add_predicate(member_of)
        o.add_predicate(married_with)


        c = o.get_constraint("memberOf(x,y)")

        indices = [i for i in product(range(len(data_people)), range(len(data_universities)))]
        indices = np.array(indices)
        atom = c.expression_tree
        vars = atom.args
        assert np.all(vars[0].indices == indices[:,0])
        assert np.all(vars[1].indices == indices[:,1])


        assert np.any(atom.indices == range(*o.predicate_range["memberOf"]))

    def test_variables_and_atom_indices_v2(self):
        #Here we test non-contiguos atom indices due to different ways variables are used (for example in simmetry check)

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

        student = Predicate("student", domains=[people])
        advised_by = Predicate("advisorOf", domains=[people, people])
        member_of = Predicate("memberOf", domains=[people, universities])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(advised_by)
        o.add_predicate(member_of)
        o.add_predicate(married_with)


        c = o.get_constraint("marriedWith(x,y) and marriedWith(y,x)", mme.logic.TFLogic)

        rp_1 = np.reshape(a=c.expression_tree.args[0].indices, newshape=[len(data_people), len(data_people)])
        rp_2 = np.reshape(a=c.expression_tree.args[1].indices, newshape=[len(data_people), len(data_people)])

        assert np.all(rp_1 == rp_2.T)

    def test_formula_parsing(self):
        #here we start testing connectives parsing

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

        student = Predicate("student", domains=[people])
        advised_by = Predicate("advisorOf", domains=[people, people])
        member_of = Predicate("memberOf", domains=[people, universities])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(advised_by)
        o.add_predicate(member_of)
        o.add_predicate(married_with)


        c = o.get_constraint("marriedWith(x,y) and marriedWith(y,x)", mme.logic.TFLogic)

    def test_compilation(self):
        """here we test compilation of formulas"""

        o = Ontology()
        data_people = ["Michelangelo", "Giuseppe", "Maria"]
        people = Domain("People", data_people)
        o.add_domain(people)

        student = Predicate("student", domains=[people])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(married_with)

        herbrand_interpretation = [0, 1, 1, #student interpretation

                                   0, 0, 0,#marriedWith interpretation
                                   0, 0, 1,
                                   0, 1, 0
                                   ]

        c = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)", mme.logic.TFLogic)
        t = c.compile(herbrand_interpretation=np.array(herbrand_interpretation, dtype=np.bool))

        sess = tf.Session()
        assert np.all(sess.run(t)) == True


    """Here we should add much more test cases for all the things imported from NMLNs"""


    def test_simple_learning_problem(self):
        """here we test a simple learning problem with gibbs sampling and monte carlo"""

        """Ontology instantiation """
        o = Ontology()

        """Domains definition"""
        data_people = ["Michelangelo", "Giuseppe", "Maria"] #this should be substituted with actual features
        people = Domain("People", data_people)
        o.add_domain(people)

        """Predicates definition"""
        student = Predicate("student", domains=[people])
        professor = Predicate("professor", domains=[people])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(professor)
        o.add_predicate(married_with)

        """These is a single interpretation we want to learn from (alias labels)"""
        herbrand_interpretation = np.array([[0, 1, 1,  # student interpretation
                                   1, 0, 0, # professor interpretation
                                   0, 0, 0,  # marriedWith interpretation
                                   0, 0, 1,
                                   0, 1, 0
                                   ]], dtype=np.bool)


        """Potentials definition"""

        """Logical Contraints definition"""
        c1 = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)", mme.logic.TFLogic)
        c2 = o.get_constraint("student(x) and not professor(x)", mme.logic.TFLogic)
        c3 = o.get_constraint("student(x) and professor(x)", mme.logic.TFLogic)

        """Creating the correspondent potentials"""
        p1 = mme.potentials.LogicPotential(c1)
        p2 = mme.potentials.LogicPotential(c2)
        p3 = mme.potentials.LogicPotential(c3)

        '''Instantiating the global potential and adding single potentials'''
        P = mme.potentials.GlobalPotential()
        P.add(p1)
        P.add(p2)
        P.add(p3)

        """Instantiating a sampling algorithm """
        sampler = mme.sampling.GPUGibbsSampler(potential=P, num_variables=o.herbrand_base_size,
                                               num_chains=10)

        """Instantiating training object using the previous sampler and MonteCarlo to compute expecations"""
        mct = mme.MonteCarloTraining(global_potential=P, sampler=sampler, p_noise=0, num_samples=10,
                                       learning_rate=0.1)

        """Training operation asks for the maximization of the likelihood of the given interpretation"""
        train_op = mct.maximize_likelihood_step(herbrand_interpretation)


        """Tensorflow training routine"""
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(train_op)



        #  Here we check that betas sign is meaningful
        assert sess.run(P.variables[0])>0
        assert sess.run(P.variables[1])>0
        assert sess.run(P.variables[2])<0




if __name__ == '__main__':
    unittest.main()