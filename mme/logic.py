import tensorflow as tf

class TFLogic():

    @staticmethod
    def _not(args):
        assert len(args)==1, "N-Ary negation not defined"
        return tf.logical_not(args[0])

    @staticmethod
    def _and(args):
        t = tf.stack(args, axis=-1)
        return tf.reduce_all(t, axis=-1)

    @staticmethod
    def _or(args):
        t = tf.stack(args, axis=-1)
        return tf.reduce_any(t, axis=-1)

    @staticmethod
    def _implies(args):
        assert len(args)==2, "N-Ary implies not defined"
        t = tf.logical_or(tf.logical_not(args[0]), args[1])
        return t

    @staticmethod
    def _iff(args):
        assert len(args) == 2, "N-Ary iff not defined"
        t = tf.equal(args[0], args[1])
        return t

    @staticmethod
    def _xor(args):
        assert len(args) == 2, "N-Ary xor not defined"
        t = tf.logical_xor(args[0], args[1])
        return t