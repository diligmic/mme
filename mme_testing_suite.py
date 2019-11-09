import unittest
from mme import Ontology, Domain, Predicate
import mme
from itertools import product
import numpy as np
import tensorflow as tf


mode = "TEST"

def exit_early_develop():
    if mode == "DEVELOPMENT":
        return True
    else:
        False


class Test(unittest.TestCase):

    def test_new_domain(self):

        if exit_early_develop():
            return

        o = Ontology()
        data_people = ["Michelangelo", "Francesco", "Giuseppe", "Maria"]
        data_universities = ["Siena", "Leuven", "Mediterranea"]
        people = Domain("People", data_people)
        universities = Domain("Universities", data_universities)
        o.add_domain(people)
        o.add_domain(universities)

    def test_duplicate_domain(self):
        if exit_early_develop():
            return
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
        if exit_early_develop():
            return

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
        if exit_early_develop():
            return

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
        if exit_early_develop():
            return

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

        if exit_early_develop():
            return
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


        c = o.get_constraint("marriedWith(x,y) and marriedWith(y,x)")

        rp_1 = np.reshape(a=c.expression_tree.args[0].indices, newshape=[len(data_people), len(data_people)])
        rp_2 = np.reshape(a=c.expression_tree.args[1].indices, newshape=[len(data_people), len(data_people)])

        assert np.all(rp_1 == rp_2.T)

    def test_formula_parsing(self):
        #here we start testing connectives parsing

        if exit_early_develop():
            return
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


        c = o.get_constraint("marriedWith(x,y) and marriedWith(y,x)")

    def test_compilation(self):
        """here we test compilation of formulas"""

        if exit_early_develop():
            return
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

        c = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)")
        t = c.compile(herbrand_interpretation=np.array(herbrand_interpretation, dtype=np.bool))

        assert np.all(t) == True


    """Here we should add much more test cases for all the things imported from NMLNs"""


    def test_simple_learning_problem(self):
        """here we test a simple learning problem with gibbs sampling and monte carlo"""

        if exit_early_develop():
            return
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
                                   ]], dtype=np.float32)


        """Potentials definition"""

        """Logical Contraints definition"""
        c1 = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)")
        c2 = o.get_constraint("student(x) and not professor(x)")
        c3 = o.get_constraint("student(x) and professor(x)")

        """Creating the correspondent potentials"""
        p1 = mme.potentials.LogicPotential(c1, mme.logic.BooleanLogic)
        p2 = mme.potentials.LogicPotential(c2, mme.logic.BooleanLogic)
        p3 = mme.potentials.LogicPotential(c3, mme.logic.BooleanLogic)

        '''Instantiating the global potential and adding single potentials'''
        P = mme.potentials.GlobalPotential()
        P.add(p1)
        P.add(p2)
        P.add(p3)

        """Instantiating a sampling algorithm """
        sampler = mme.inference.GPUGibbsSampler(potential=P, num_variables=o.herbrand_base_size,
                                                num_chains=10)

        """Instantiating training object using the previous sampler and MonteCarlo to compute expecations"""
        mct = mme.MonteCarloTraining(global_potential=P, sampler=sampler, p_noise=0, num_samples=10,
                                       learning_rate=0.1)

        """Training operation asks for the maximization of the likelihood of the given interpretation"""


        """Tensorflow training routine"""
        for i in range(10):
            mct.maximize_likelihood_step(herbrand_interpretation)

        #  Here we check that betas sign is meaningful
        assert P.variables[0]>0
        assert P.variables[1]>0
        assert P.variables[2]<0

    def test_piecewise_with_logical_supervision(self):
        """Here we test piecewise supervised learning with a logic description"""

        if exit_early_develop():
            return
        """Loading Data"""
        num_examples = 100
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train[:num_examples]
        y_train = y_train[:num_examples]

        x_train = np.reshape(x_train, [-1, 784])
        y_train = np.eye(10)[y_train]
        herbrand_interpretation = np.reshape(y_train.T, [1, -1])

        """Logic description"""
        o = Ontology()

        d = mme.Domain("Images", data=x_train)
        o.add_domain(d)

        predicates_to_supervise = []
        for i in range(10):
            pred = Predicate("%d" % i, domains=[d])
            o.add_predicate(pred)
            predicates_to_supervise.append(pred)

        """Defining a neural model on which to condition our distribution"""
        nn = tf.keras.Sequential()
        nn.add(tf.keras.layers.Input(shape=(784,)))
        nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  # up to the last hidden layer

        """Instantiating the supervised potential"""
        p1 = mme.potentials.SupervisionLogicalPotential(model=nn, predicates=predicates_to_supervise, ontology=o)

        """Instantiating the Global Potential"""
        P = mme.potentials.GlobalPotential()
        P.add(p1)

        """Instantiating training object using the previous sampler and MonteCarlo to compute expecations"""
        pwt = mme.PieceWiseTraining(global_potential=P, learning_rate=0.001, y=herbrand_interpretation)

        """Tensorflow training routine"""

        for i in range(100):
            pwt.maximize_likelihood_step(y=herbrand_interpretation, x=x_train)

            # if i%10==0:
            #
            #     print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p1.model(x_train), axis=1),tf.argmax(y_train, axis=1)), tf.float32)))

        assert (tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(p1.model(x_train), axis=1), tf.argmax(y_train, axis=1)), tf.float32)) > 0.7)

    def testing_piecewise_model_counting(self):

        if exit_early_develop():
            return
        """Ontology instantiation """
        o = Ontology()

        """Domains definition"""
        data_people = ["Michelangelo", "Giuseppe", "Maria"]  # this should be substituted with actual features
        people = Domain("People", data_people)
        o.add_domain(people)

        """Predicates definition"""
        student = Predicate("student", domains=[people])
        professor = Predicate("professor", domains=[people])
        married_with = Predicate("marriedWith", domains=[people, people])
        o.add_predicate(student)
        o.add_predicate(professor)
        o.add_predicate(married_with)

        """Logical Contraints definition"""
        c1 = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)")
        assert tf.math.count_nonzero(c1.compile(None,mme.logic.BooleanLogic)) == 3

        c2 = o.get_constraint("marriedWith(x,y) and marriedWith(y,x)")
        assert tf.math.count_nonzero(c2.compile(None,mme.logic.BooleanLogic)) == 1

        c3 = o.get_constraint("(student(x) and marriedWith(x,y)) -> marriedWith(y,x)")
        assert tf.math.count_nonzero(c3.compile(None, mme.logic.BooleanLogic)) == 7

        c4 = o.get_constraint("student(x) and marriedWith(x,y) and marriedWith(y,x)")
        assert tf.math.count_nonzero(c4.compile(None, mme.logic.BooleanLogic)) == 1


    def test_logic_problem_with_peacewise(self):



        if exit_early_develop():
            return

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
                                   ]], dtype=np.float32)


        """Potentials definition"""

        """Logical Contraints definition"""
        c1 = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)")
        c2 = o.get_constraint("student(x) -> not professor(x)")
        c3 = o.get_constraint("student(x) and professor(x)")

        """Creating the correspondent potentials"""
        p1 = mme.potentials.LogicPotential(c1,mme.logic.BooleanLogic)
        p2 = mme.potentials.LogicPotential(c2, mme.logic.BooleanLogic)
        p3 = mme.potentials.LogicPotential(c3, mme.logic.BooleanLogic)

        '''Instantiating the global potential and adding single potentials'''
        P = mme.potentials.GlobalPotential()
        P.add(p1)
        P.add(p2)
        P.add(p3)

        """Instantiating training object using the previous sampler and MonteCarlo to compute expecations"""
        pwt = mme.PieceWiseTraining(global_potential=P, learning_rate=0.1, y=herbrand_interpretation)
        pwt.compute_beta_logical_potentials()

        """Training operation asks for the maximization of the likelihood of the given interpretation"""


        """Tensorflow training routine""" #---> no training routing is necessary for logical constraints


        print([p.beta for p in P.potentials])

        #  Here we check that betas sign is meaningful
        assert P.potentials[0].beta>0
        assert P.potentials[1].beta>0
        assert P.potentials[2].beta<0


    def test_fuzzy_map_inference(self):


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
                                   ]], dtype=np.float32)


        """Potentials definition"""

        """Logical Contraints definition"""
        c1 = o.get_constraint("marriedWith(x,y) -> marriedWith(y,x)")
        c2 = o.get_constraint("student(x) -> not professor(x)")
        c3 = o.get_constraint("student(x) -> professor(x)")

        """Creating the correspondent potentials"""
        p1 = mme.potentials.LogicPotential(c1,mme.logic.BooleanLogic)
        p2 = mme.potentials.LogicPotential(c2, mme.logic.BooleanLogic)
        p3 = mme.potentials.LogicPotential(c3, mme.logic.BooleanLogic)

        '''Instantiating the global potential and adding single potentials'''
        P = mme.potentials.GlobalPotential()
        P.add(p1)
        P.add(p2)
        P.add(p3)

        """Instantiating training object using the previous sampler and MonteCarlo to compute expecations"""
        pwt = mme.PieceWiseTraining(global_potential=P, learning_rate=0.1, y=herbrand_interpretation)
        pwt.compute_beta_logical_potentials()




        """Inference"""
        evidence = np.zeros(herbrand_interpretation.shape)
        evidence[0,11] = 1 # marriedWith(Giuseppe,Maria)
        evidence[0,1] = 1 # student(Giuseppe)
        evidence_mask = np.array(evidence)>0
        map_inference = mme.inference.FuzzyMAPInference(y_shape=herbrand_interpretation.shape,
                                        potential=P,
                                        logic = mme.logic.LukasiewiczLogic,
                                        evidence=evidence,
                                        evidence_mask=evidence_mask)

        for i in range(20):
            map_inference.infer_step()

        print(map_inference.map())
        assert map_inference.map()[0,13]>0.7 # marriedWith(Maria,Giuseppe)
        assert map_inference.map()[0,4]<0.3 # professor(Giuseppe)




if __name__ == '__main__':
    # mode = "DEVELOPMENT"
    mode = "TEST"
    unittest.main()