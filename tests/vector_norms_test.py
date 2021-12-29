import numpy as np


def random_vec_norm_1(size: tuple = (10,)):
    generator = np.random.Generator(np.random.MT19937())
    v = generator.uniform(low=-1, high=1, size=size)
    return v / np.linalg.norm(v)


def score(val: float):
    return np.exp(-val)


def _and(a: float, b: float):
    return np.minimum(a, b)


def MLP(eA: np.array, eB: np.array):
    assert (
        len(eA.shape) == 1 and len(eB.shape) == 1
    ), "Inputs should be one dimensional vectors"
    assert (
        eA.shape[0] == eB.shape[0]
    ), f"Wrong shapes: eA.shape[0] = {eA.shape[0]} vs eB.shape[0] = {eB.shape[0]}"
    input = np.concatenate((eA, eB), axis=-1)
    generator = np.random.Generator(np.random.MT19937())
    W = generator.uniform(low=-1, high=1, size=(eA.shape[0], input.shape[0]))
    b = generator.uniform(low=-1, high=1, size=(eA.shape[0],))
    output = W @ input + b
    return output


def head_embedding(eA: np.array, eB: np.array):

    # norm of embedding of atom should be almost 0 if correct
    # in general norm(atom_embedding) in [0, +inf)
    norm_eA = np.linalg.norm(eA)
    norm_eB = np.linalg.norm(eB)

    # [0, +inf) -> [1, 0)
    # norm == 0 <-> score == 1
    # norm -> +inf <-> score -> 0
    score_eA = score(norm_eA)
    score_eB = score(norm_eB)

    # semantics injection
    # given the scores of the atoms composing the head of a formula, the representation
    # of the head itself is the truth value between the atoms. The result, between 0 and 1
    # is then mapped back to [0, + inf), thus obtaining the norm of the representation.
    # norm(eAB) = -ln(score_eA and score_eB)
    norm_eAB = -np.math.log(_and(score_eA, score_eB))

    # the norm of the representation of the head of the formula defines a hypersphere in a
    # N dimensional space, thus infinite vectors that could play the role of the
    # representation of the head of the formula. In order to select the "right" vector
    # we use an MLP that will provide the direction of the vector itself.
    eAB_1 = MLP(eA, eB)
    eAB = eAB_1 / np.linalg.norm(eAB_1) * norm_eAB
    return eAB


if __name__ == "__main__":

    # embedding size
    N = 10

    # constants embeddings
    e_italy = random_vec_norm_1(size=(N,))
    e_south_europe = random_vec_norm_1(size=(N,))
    e_europe = random_vec_norm_1(size=(N,))
    e_locatedIn = random_vec_norm_1(size=(N,))

    # A and B -> C
    # A = locatedIn(italy, south_europe)
    # B = locatedIn(south_europe, europe)
    # C = locatedIn(italy, europe)

    # embedding of atom R(a,b): e = e_a + e_R - e_b
    eA = e_italy + e_locatedIn - e_south_europe
    eB = e_south_europe + e_locatedIn - e_europe
    eC = e_italy + e_locatedIn - e_europe

    eAB = head_embedding(eA, eB)
