import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


tf.get_logger().setLevel('ERROR')

base_savings = os.path.join("savings", "citeseer")
pretrain_path = os.path.join(base_savings,"pretrain")
posttrain_path = os.path.join(base_savings,"posttrain")

def main(lr,seed,lambda_0,l2w, test_size, valid_size, run_on_test=False, map_steps = 20, em_cycles=4):




    (x_train, hb_train), (x_valid, hb_valid), (x_test, hb_test), (x_all, hb_all), labels, mask_train_labels, trid, vaid, teid= datasets.citeseer_em(test_size, valid_size, seed)
    num_examples = len(x_all)
    num_classes = 6

    np.random.seed(seed)
    tf.random.set_seed(seed)

    indices = np.reshape(np.arange(num_classes * len(x_all)),
                         [num_classes, len(x_all)]).T  # T because we made classes as unary potentials

    indices_train = indices[trid]
    if run_on_test:
        x_to_test = x_test
        hb_to_test = hb_test
        num_examples_to_test = len(x_test)
        indices_to_test = indices[teid]

    else:
        x_to_test = x_valid
        hb_to_test = hb_valid
        num_examples_to_test = len(x_valid)
        indices_valid = np.reshape(np.arange(num_classes * len(x_valid)),
                             [num_classes, len(x_valid)]).T  # T because we made classes as unary potentials
        indices_to_test = indices[vaid]

    y_to_test = tf.gather(hb_all[0], indices_to_test)

    """Logic Program Definition"""
    o = mme.Ontology()

    # Domains
    docs = mme.Domain("Documents", data=x_all)
    o.add_domain([docs])

    # Predicates

    preds = ["ag", "ai", "db", "ir", "ml", "hci"]
    for name in preds:
        p = mme.Predicate(name, domains=[docs])
        o.add_predicate(p)

    cite = mme.Predicate("cite", domains=[docs, docs], given=True)
    o.add_predicate(cite)

    """MME definition"""
    potentials = []
    # Supervision
    indices = np.reshape(np.arange(num_classes * docs.num_constants),
                         [num_classes, docs.num_constants]).T  # T because we made classes as unary potentials

    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(num_classes, use_bias=False))
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)
    potentials.append(p1)

    # Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    # p2 = mme.potentials.MutualExclusivityPotential(indices=indices)
    # potentials.append(p2)

    # Logical
    np.ones_like(hb_all)
    evidence_mask = np.zeros_like(hb_all)
    evidence_mask[:, num_examples * num_classes:]=1
    for name in preds:
        c = mme.Formula(definition="%s(x) and cite(x,y) -> %s(y)" % (name, name), ontology=o)
        p3 = mme.potentials.EvidenceLogicPotential(formula=c, logic=mme.logic.BooleanLogic, evidence=hb_all,
                                                   evidence_mask=evidence_mask)
        potentials.append(p3)

    P = mme.potentials.GlobalPotential(potentials)

    def pretrain_step():
        """pretrain rete"""
        y_train = tf.gather(hb_all[0], indices_train)

        adam = tf.keras.optimizers.Adam(lr=0.001)

        def training_step():
            with tf.GradientTape() as tape:
                neural_logits = nn(x_train)

                total_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_train,
                                                            logits=neural_logits)) + tf.reduce_sum(nn.losses)

            grads = tape.gradient(target=total_loss, sources=nn.variables)
            grad_vars = zip(grads, nn.variables)
            adam.apply_gradients(grad_vars)

        epochs_pretrain = 200
        for e in range(epochs_pretrain):
            training_step()
            y_nn = nn(x_to_test)
            acc_nn = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
            print(acc_nn)

        y_new = tf.gather(tf.eye(num_classes), tf.argmax(nn(x_all), axis=1), axis=0)

        new_labels = tf.where(mask_train_labels > 0, labels, y_new)
        new_hb = tf.concat(
            (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]),axis=1)

        return new_hb


    def em_step(new_hb):

        pwt = mme.PieceWiseTraining(global_potential=P)
        hb = new_hb

        """BETA TRAINING"""
        pwt.compute_beta_logical_potentials(y=hb)
        for p in potentials:
            print(p, p.beta)

        """NN TRAINING"""
        epochs = 50

        for _ in range(epochs):
            pwt.maximize_likelihood_step(new_hb, x=x_all)
            y_nn = nn(x_to_test)
            acc_nn = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
            print("TRAINING", acc_nn.numpy())


        """Fix the training hb for inference"""
        new_labels = tf.where(mask_train_labels > 0, labels, 0.5*tf.ones_like(labels))
        evidence = tf.concat(
            (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)
        evidence = tf.cast(evidence, tf.float32)
        evidence_mask = tf.concat(
            (tf.reshape(tf.transpose(mask_train_labels.astype(np.float32), [1, 0]), [1, -1]), tf.ones_like(hb_all[:, num_examples * num_classes:])), axis=1)>0

        """MAP Inference"""
        dict = {"var": tf.Variable(initial_value=nn(x_all)),
                "labels": labels,
                "mask_train_labels": mask_train_labels,
                "hb_all": hb_all,
                "num_examples": num_examples,
                "num_classes": num_classes}

        steps_map = map_steps
        map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                        potential=P,
                                                        logic=mme.logic.LukasiewiczLogic,
                                                        evidence=evidence,
                                                        evidence_mask=evidence_mask,
                                                        learning_rate=lr,
                                                        external_map = dict)

        P.potentials[0].beta = lambda_0
        y_map = tf.gather(map_inference.map()[0], indices_to_test)
        acc_map = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
        print("MAP", acc_map.numpy())
        for i in range(steps_map):
            map_inference.infer_step(x_all)
            y_map = tf.gather(map_inference.map()[0], indices_to_test)
            acc_map = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
            print("MAP", i, acc_map.numpy())
            if mme.utils.heardEnter():
                break


        new_hb = tf.cast(map_inference.map()>0.5, tf.float32)

        y_new_fuzzy = tf.gather(map_inference.map()[0], indices)
        new_labels = tf.where(mask_train_labels > 0, labels, y_new_fuzzy)
        new_hb_fuzzy = tf.concat(
            (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)

        return new_hb, new_hb_fuzzy


    em_cycles = em_cycles
    for i in range(em_cycles):
        if i == 0:
            new_hb = hb_pretrain = pretrain_step()
        else:
            old_hb = new_hb
            new_hb, new_hb_fuzzy = em_step(old_hb)

    y_map = tf.gather(new_hb[0], indices_to_test)
    y_pretrain = tf.gather(hb_pretrain[0], indices_to_test)
    acc_pretrain = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_pretrain, axis=1)), tf.float32))
    acc_map = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
    return acc_pretrain, acc_map

















if __name__ == "__main__":
    seed = 0
    with open("res_dlm_10_splits", "w") as file:
        file.write("seed, test, acc_map, acc_nn\n")
    res = []
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.random.seed(0)
    seeds=np.random.choice(np.arange(1000), [10], replace=False)
    # seeds=[0]
    for a  in product( seeds,[(0.1,0.,4,30),(0.25,0.001,4,30),(0.5,0.001,4,30),(0.75,0.001,4,30),(0.9,0.001,4,30)]):

        seed,(test_size,lambda_0,em_cycles, steps) = a
        acc_map, acc_nn = main(lr=1., seed=seed, lambda_0 =lambda_0, l2w=0.006, test_size=test_size, valid_size=0., run_on_test=True, map_steps=steps, em_cycles=em_cycles)
        acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
        res.append("\t".join([str(a) for a in  [seed, test_size, acc_map, str(acc_nn)+"\n"]]))
        for i in res:
            print(i)
        with open("res_dlm_10_splits", "a") as file:
            file.write(res[-1])





