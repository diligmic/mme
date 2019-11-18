import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')

base_savings = os.path.join("savings", "citeseer")
pretrain_path = os.path.join(base_savings,"pretrain")
posttrain_path = os.path.join(base_savings,"posttrain")

def main(lr,seed,perc_soft,l2w):



    documents, trid, teid, hb_all, me_per_inference, me_per_training,labels,mask_on_labels = datasets.citeseer_em()
    num_examples = len(documents)
    num_classes = 6


    #I set the seed after since i want the dataset to be always the same
    np.random.seed(seed)
    tf.random.set_seed(seed)

    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(documents.shape[1],)))
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.sigmoid))
    nn.add(tf.keras.layers.Dense(num_classes, use_bias=False, activation=None))

    def pretrain():
        """pretrain rete"""
        x_train = documents[trid]
        y_train = labels[trid]

        adam = tf.keras.optimizers.Adam(lr=0.001)

        def training_step():
            with tf.GradientTape() as tape:
                neural_logits = nn(x_train)

                total_loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels= y_train,
                                                            logits=neural_logits)) + tf.reduce_sum(nn.losses)

            grads = tape.gradient(target=total_loss, sources=nn.variables)
            grad_vars = zip(grads, nn.variables)
            adam.apply_gradients(grad_vars)

        epochs_pretrain = 150
        for e in range(epochs_pretrain):
            training_step()
            y_nn = tf.nn.softmax(nn(documents))
            acc_nn = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(tf.gather(labels, teid), axis=1), tf.argmax(tf.gather(y_nn, teid), axis=1)), tf.float32))
            print(acc_nn)


    pretrain()

    y_nn = tf.gather(tf.eye(num_classes), tf.argmax(nn(documents), axis=1), axis=0)
    new_labels = tf.where(mask_on_labels>0, labels, y_nn)
    new_hb = tf.concat((tf.reshape(tf.transpose(new_labels, [1,0]), [1, -1]), hb_all[:,num_examples*num_classes:]), axis=1)

    """Logic Program Definition"""
    o = mme.Ontology()

    # Domains
    docs = mme.Domain("Documents", data=documents)
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
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)
    potentials.append(p1)

    # Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    p2 = mme.potentials.MutualExclusivityPotential(indices=indices)
    potentials.append(p2)

    # Logical
    logical_preds = []
    for name in preds:
        c = mme.Formula(definition="%s(x) and cite(x,y) -> %s(y)" % (name, name), ontology=o)
        p3 = mme.potentials.EvidenceLogicPotential(formula=c, logic=mme.logic.BooleanLogic, evidence=me_per_training*hb_all,
                                                   evidence_mask=me_per_training)
        potentials.append(p3)

    P = mme.potentials.GlobalPotential(potentials)
    pwt = mme.PieceWiseTraining(global_potential=P)


    def em_cycle(hb, documents):


        """BETA TRAINING"""
        pwt.compute_beta_logical_potentials(y=hb)
        for p in potentials:
            print(p, p.beta)

        """NN TRAINING"""
        epochs = 50
        for _ in range(epochs):
            pwt.maximize_likelihood_step(hb, x=documents)
            y_nn = tf.nn.softmax(nn(documents))
            acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.gather(labels, teid), axis=1), tf.argmax(tf.gather(y_nn, teid), axis=1)), tf.float32))
            print(acc_nn)



        y_nn = tf.nn.softmax(nn(documents))
        new_labels = tf.where(mask_on_labels > 0, labels, y_nn)
        evidence = tf.concat((tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)

        map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                        potential=P,
                                                        logic=mme.logic.LukasiewiczLogic,
                                                        evidence=evidence,
                                                        evidence_mask=me_per_inference,
                                                        learning_rate= lr)
                                                        # initial_value=evidence) #tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps_map, decay_rate=0.96, staircase=True))

        steps_map = 100
        print(tf.gather(map_inference.map()[0], indices)[:10])
        input()

        for i in range(steps_map):
            map_inference.infer_step(documents)
            y_map = tf.gather(map_inference.map()[0], indices)
            acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.gather(labels, teid), axis=1), tf.argmax(tf.gather(y_map, teid), axis=1)), tf.float32))
            print("Accuracy MAP", acc_map.numpy())
            print(y_map[:10])

            if mme.utils.heardEnter():
                break

        y_map = tf.gather(tf.eye(num_classes), tf.argmax(tf.gather(map_inference.map()[0], indices), axis=1), axis=0)
        new_labels = tf.where(mask_on_labels > 0, labels, y_map)
        new_hb = tf.concat( (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)

        return new_hb


    em_steps = 150
    for _ in range(em_steps):
        new_hb = em_cycle(new_hb, documents)


if __name__ == "__main__":
    seed = 0

    res = []
    for a  in product([0], [0.06]):
        perc, lr = a
        acc_map, acc_nn = main(lr=lr, seed=seed, perc_soft=perc, l2w=0.1)
        acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
        res.append("\t".join([str(a) for a in [perc, lr, acc_map, str(acc_nn)+"\n"]]))
        for i in res:
            print(i)

    with open("res_dlm_citeceer_em"%seed, "w") as file:
        file.write("perc, lr, acc_map, acc_nn\n")
        file.writelines(res)






