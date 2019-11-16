import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')


def main(lr,seed,perc_soft,l2w):



    (x_train, hb_train), (x_test, hb_test) = datasets.citeseer()
    num_examples = len(x_train)
    num_classes = 6


    #I set the seed after since i want the dataset to be always the same
    np.random.seed(seed)
    tf.random.set_seed(seed)

    m_e = np.zeros_like(hb_train)
    m_e[:, num_examples*num_classes:] = 1

    y_e_train = hb_train * m_e
    y_e_test = hb_test * m_e

    """Logic Program Definition"""
    o = mme.Ontology()

    #Domains
    docs = mme.Domain("Documents", data=x_train)
    o.add_domain([docs])

    # Predicates

    preds = ["ag","ai", "db","ir","ml","hci"]
    for name in preds:
        p = mme.Predicate(name, domains=[docs])
        o.add_predicate(p)

    cite = mme.Predicate("cite", domains=[docs,docs], given=True)
    o.add_predicate(cite)

    """MME definition"""
    potentials = []
    #Supervision
    indices = np.reshape(np.arange(num_classes * docs.num_constants),
                         [num_classes, docs.num_constants]).T # T because we made classes as unary potentials
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))
    nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(num_classes,use_bias=False))
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)
    potentials.append(p1)

    #Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    p2 = mme.potentials.MutualExclusivityPotential(indices=indices)
    potentials.append(p2)

    #Logical
    logical_preds = []
    for name in preds:
        c = mme.Formula(definition="%s(x) and cite(x,y) -> %s(y)" % (name,name), ontology=o)
        p3 = mme.potentials.EvidenceLogicPotential(formula=c,logic=mme.logic.BooleanLogic, evidence=y_e_train, evidence_mask=m_e)
        potentials.append(p3)


    P = mme.potentials.GlobalPotential(potentials)


    pwt = mme.PieceWiseTraining(global_potential=P, y=hb_train)
    pwt.compute_beta_logical_potentials()
    for p in potentials:
        print(p, p.beta)
    P.save_weights("citeseer_pretrain")



    epochs = 150
    y_test = tf.gather(hb_test[0], indices, axis=1)
    for _ in range(epochs):
        pwt.maximize_likelihood_step(hb_train, x=x_train)
        y_nn = tf.nn.softmax(nn(x_test))
        acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        print(acc_nn)
    P.save_weights("citeseer_post_train")


    #Test accuracy after supervised step
    y_nn = tf.nn.softmax(nn(x_test))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
    print(acc_nn)
    exit()


    """Inference"""
    steps_map = 500
    hb = hb_test
    x = x_test
    evidence = y_e_test
    evidence_mask = m_e>0


    map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                    potential=P,
                                                    logic=mme.logic.LukasiewiczLogic,
                                                    evidence=evidence,
                                                    evidence_mask=evidence_mask,
                                                    learning_rate= lr) #tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps_map, decay_rate=0.96, staircase=True))

    y_test = tf.reshape(hb[0, :num_examples * num_classes], [num_examples, num_classes])
    for i in range(steps_map):
        map_inference.infer_step(x)
        if i % 10 == 0:
            y_map = tf.reshape(map_inference.map()[0, :num_examples * num_classes], [num_examples, num_classes])
            acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
            print("Accuracy MAP", acc_map.numpy())

        if mme.utils.heardEnter():
            break

    y_map = tf.reshape(map_inference.map()[0, :num_examples * num_classes], [num_examples, num_classes])
    y_nn = p1.model(x)

    acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))

    return [acc_map, acc_nn]



if __name__ == "__main__":
    seed = 0

    res = []
    for a  in product([0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1], [1, 0.3, 0.1, 0.06, 0.03, 0.01]):
        perc, lr = a
        acc_map, acc_nn = main(lr=lr, seed=seed, perc_soft=perc, l2w=0.01)
        acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
        res.append("\t".join([str(a) for a in [perc, lr, acc_map, str(acc_nn)+"\n"]]))
        for i in res:
            print(i)

    with open("res_dlm_%d"%seed, "w") as file:
        file.write("perc, lr, acc_map, acc_nn\n")
        file.writelines(res)





