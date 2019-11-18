import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')

base_savings = os.path.join("savings", "citeseer")
pretrain_path = os.path.join(base_savings,"pretrain")
posttrain_path = os.path.join(base_savings,"posttrain")

def main(lr,seed,perc_soft=0,l2w=0.1, w_rule=0.01):



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
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.sigmoid))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(num_classes,use_bias=False))
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)
    potentials.append(p1)

    #Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    p2 = mme.potentials.MutualExclusivityPotential(indices=indices)
    potentials.append(p2)

    #Logical
    logical_preds = []
    constraints = []
    for name in preds:
        c = mme.Formula(definition="%s(x) and cite(x,y) -> %s(y)" % (name,name), ontology=o)
        p3 = mme.potentials.EvidenceLogicPotential(formula=c,logic=mme.logic.BooleanLogic, evidence=y_e_train, evidence_mask=m_e)
        potentials.append(p3)
        constraints.append(c)

    adam = tf.keras.optimizers.Adam(lr=0.001)




    def make_hb_with_model(neural_softmax, hb_all):
        new_hb = tf.concat(
            (tf.reshape(tf.transpose(neural_softmax, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)
        return new_hb


    def training_step(logic=False):
        with tf.GradientTape() as tape:
            neural_logits = nn(x_train)


            total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.gather(hb_train[0], indices),
                                                        logits=neural_logits)) + tf.reduce_sum(nn.losses)

            if logic:
                neural_softmax = tf.nn.softmax(nn(x_train))
                hb_model_train = make_hb_with_model(neural_softmax, hb_train)

                logical_loss = 0

                for c in constraints:
                    groundings = c.ground(herbrand_interpretation=hb_model_train)
                    logical_loss += tf.reduce_mean(- c.compile(groundings, mme.logic.LukasiewiczLogic))

                total_loss += w_rule*logical_loss

        grads = tape.gradient(target=total_loss, sources=nn.variables)
        grad_vars = zip(grads, nn.variables)
        adam.apply_gradients(grad_vars)



    logic = False
    epochs = 150
    y_test = tf.gather(hb_test[0], indices)
    for e in range(epochs):
        training_step(logic)
        y_nn = nn(x_test)
        acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        print(acc_nn)



    """Inference"""
    adam2 = tf.keras.optimizers.Adam(lr=lr)

    steps_map = 250
    hb = hb_test
    x = x_test
    evidence = y_e_test
    evidence_mask = m_e>0

    # Recreating an mme scenario
    mutual = mme.potentials.MutualExclusivityPotential(indices)

    prior = tf.nn.softmax(nn(x_test))
    y_bb = tf.Variable(initial_value=prior)
    max_beta_me = 2

    lambda_0 = 0.05

    def map_inference_step(i):

        with tf.GradientTape() as tape:
            y_map = tf.sigmoid(10 * (y_bb - 0.5))
            hb_model_test = make_hb_with_model(y_map, hb_test)
            groundings = c.ground(herbrand_interpretation=hb_model_test)
            inference_loss = w_rule * tf.reduce_mean(- c.compile(groundings, mme.logic.LukasiewiczLogic)) \
                             + lambda_0 * tf.reduce_mean(tf.square(prior - y_bb)) \
                             +(max_beta_me - max_beta_me*(steps_map - i)*steps_map)* tf.reduce_sum(- mutual.call_on_groundings(y_map))

        grads = tape.gradient(target=inference_loss, sources=y_bb)
        grad_vars = [(grads, y_bb)]
        adam2.apply_gradients(grad_vars)

    for e in range(steps_map):
        map_inference_step(e)
        acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_bb, axis=1)), tf.float32))
        print(acc_map)
        if mme.utils.heardEnter(): break
    acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_bb, axis=1)), tf.float32))

    return acc_nn, acc_map


if __name__ == "__main__":
    seed = 0

    res = []
    for a  in product([5,20], [0.01]):
        w_rule, lr = a
        acc_map, acc_nn = main(lr=lr, seed=seed, w_rule=w_rule, l2w=0.1)
        acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
        res.append("\t".join([str(a) for a in [w_rule, lr, acc_map, str(acc_nn)+"\n"]]))
        for i in res:
            print(i)

    with open("res_dlm_%d"%seed, "w") as file:
        file.write("perc, lr, acc_map, acc_nn\n")
        file.writelines(res)





