import tensorflow as tf


def supervised_loss(train_mask: tf.Tensor, cliques_mask: tf.Tensor, params):
    """Supervised loss.

    Args:
        train_mask (tf.Tensor):
            Tensor of shape (1, o.linear_size()), having 1s in correspondence of those
            atoms which are to be considered as train examples.
        cliques_mask (tf.Tensor):
            Tensor resulting from the concatenation of the masks for each grounded formula
            with shape (1, sum(grounded_formulas_size)), having 1s in correspondence of
            the train atoms for the cliques. For the semantic case.
        params (utils.Params):
            Command line parameters passed as Params object.
    """

    def _loss(targets, predictions):
        supervised_loss = tf.keras.backend.binary_crossentropy(targets, predictions)

        # Negative sampling
        atoms_weigths = tf.where(
            tf.logical_and(
                train_mask == 0.0,
                tf.random.uniform(shape=train_mask.shape)
                < 1.0 - params.prob_neg_sampling,
            ),
            tf.ones_like(train_mask),
            train_mask,
        )

        if cliques_mask is not None:
            cliques_weights = params.semantic_loss_weight * cliques_mask
            weights = tf.concat([atoms_weigths, cliques_weights], axis=-1)
        else:
            weights = atoms_weigths

        supervised_loss = tf.reduce_sum(supervised_loss * weights) / tf.reduce_sum(
            weights
        )

        return supervised_loss

    return _loss
