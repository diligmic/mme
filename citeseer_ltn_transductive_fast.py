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

def main(lr,seed,test_size, valid_size=0.,l2w=0.006, w_rule=1000., ):
    (x_train, hb_train), (x_valid, hb_valid), (x_test, hb_test), (x_all, hb_all), labels, mask_train_labels, trid, vaid, teid = datasets.citeseer_em(test_size,valid_size)
    num_examples = len(x_all)
    num_classes = 6

    #I set the seed after since i want the dataset to be always the same
    np.random.seed(seed)
    tf.random.set_seed(seed)



    indices = np.reshape(np.arange(num_classes * num_examples),
                         [num_classes, num_examples]).T # T because we made classes as unary potentials
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(x_all.shape[1],)))
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(num_classes,use_bias=False))

    rng = np.arange(num_examples)
    x_ids = np.reshape(np.tile(rng, [num_examples, 1]).T, [-1])
    y_ids = np.tile(rng, [num_examples])

    l = mme.logic.LukasiewiczLogic

    cite = np.tile(np.expand_dims(hb_all[0,num_examples*num_classes:],1), [1, num_classes])





    adam = tf.keras.optimizers.Adam(lr=0.001)



    def training_step(logic=False):
        with tf.GradientTape() as tape:
            neural_logits = nn(x_all[trid])


            total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels[trid],
                                                        logits=neural_logits)) + tf.reduce_sum(nn.losses)
            if logic:

                preds = tf.nn.softmax(nn(x_all))

                px = tf.gather(preds, x_ids)
                py = tf.gather(preds, y_ids)

                rule = l._implies([l._and([px, cite]), py])
                logical_loss = - tf.reduce_mean(tf.reduce_sum(rule, axis=1))

                total_loss += w_rule*logical_loss

        grads = tape.gradient(target=total_loss, sources=nn.variables)
        grad_vars = zip(grads, nn.variables)
        adam.apply_gradients(grad_vars)



    logic = False
    epochs = 300
    y_test = labels[teid]
    for e in range(epochs):
        if e>=200:
            pretrain_acc=acc_nn
            logic=True
        training_step(logic)
        y_nn = tf.nn.softmax(nn(x_test))
        acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        print(acc_nn)

    return pretrain_acc, acc_nn

    #Test accuracy after supervised step
    y_nn = tf.nn.softmax(nn(x_test))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))



    """Inference"""
    steps_map = 500
    hb = hb_test
    x = x_test
    evidence = y_e_test
    evidence_mask = m_e>0


    initial_nn = tf.concat((tf.reshape(tf.transpose(tf.nn.softmax(nn(x_test)),[1,0]), [1, -1]), hb_test[:,num_examples*num_classes:]), axis=1)

    map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                    potential=P,
                                                    logic=mme.logic.LukasiewiczLogic,
                                                    evidence=evidence,
                                                    evidence_mask=evidence_mask,
                                                    learning_rate= lr,
                                                    initial_value=initial_nn) #tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps_map, decay_rate=0.96, staircase=True))

    y_test = tf.gather(hb[0], indices)
    for i in range(steps_map):
        map_inference.infer_step(x)
        y_map = tf.gather(map_inference.map()[0], indices)
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
    for a  in product( [0.25, 0.1],[0.01]):
        test_size,lr = a
        acc_map, acc_nn = main(lr=lr, seed=seed, l2w=0.006, test_size=test_size, valid_size=0. )
        acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
        res.append("\t".join([str(a) for a in [ test_size, acc_map, str(acc_nn)+"\n"]]))
        for i in res:
            print(i)

    with open("res_dlm_%d"%seed, "w") as file:
        file.write("perc, lr, acc_map, acc_nn\n")
        file.writelines(res)





