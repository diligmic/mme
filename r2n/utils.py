import sys
import re
import os
import getopt
import inspect
import random
import time
import pickle
import collections
import sortedcontainers
import argparse

import sklearn.metrics
import tensorflow as tf
import numpy as np

import r2n.knowledge as knowledge
import r2n.logic as logic


class Params:
    def __init__(self):
        self.seed = random.randint(0, 1000)
        # constant embedding parameters
        self.constant_embedding_sizes = []
        self.constant_embedder_regularization = 0.0
        self.constant_embedding_normalization = False
        self.constant_embedder_activation = None
        self.per_domain_constant_embedding_sizes = {}
        self.per_domain_constant_activation = {}
        # atom embedding parameters
        self.atom_embedder = "TransE"
        self.atom_embedder_regularization = 0.0
        self.atom_embedding_size = [30]
        self.atom_embedding_normalization = False
        self.atom_embedding_activation = "linear"
        self.per_predicate_atom_embedder = {}
        # formula embedding and chained transformers parameters
        self.formula_embedder = "Transformers"
        self.num_transformers = 3
        self.share_network_transformers = False
        self.transformer_embedding_size = 30
        self.transformer_masked_predictions = True
        self.transformer_embedder_regularization = 0.0
        self.transformer_input_dropout = 0.0
        # pretrain parameters
        self.pretrain = False
        self.pretrain_epochs = 50
        self.pretrain_learning_rate = 0.03
        # train parameters
        self.epochs = 300
        self.learning_rate = 0.01
        self.reg_weight = 0.0
        self.optimizer = "adam"
        self.prob_neg_sampling = 1.0
        self.num_stages = 10
        self.first_stage_prop = 1.0  # first_stage_prop=1 disable training stages
        # loss weights
        self.positive_example_weight = 1.0
        self.reasoning_loss_weight = 1.0
        self.no_reasoning_loss_weight = 1.0
        self.no_reasoning_min_loss_weight = 1.0
        # other parameters
        self.debug = False
        self.boolean_values_init_value_for_test_data = 0.0
        self.concat_atom_and_reasoning_embeddings = False
        self.reasoning_on_embeddings = True

        # model specific parameters
        self.model = "R2NC"
        self.task = "S1"
        self.formula_file = r"data/countries/formulas.txt"

        # Specilizes the domain embedders by domain. This is useful for
        # symbolic-subsymbolic tasks where it is required to differenciate
        # the constant embedders.
        self.per_domain_embedding_sizes = {}
        self.per_domain_activation = {}

        # semantic parameters
        self.attention_formulas = False
        self.formula_loss_weight = 1.0
        self.transductive_loss = False
        self.semantic_loss_weight = 0.0
        self.formula_balance_weight = 0.1

        # Formulas params
        self.formula_hidden_neurons = 0
        self.formula_dropout_prob = 0.0

    def __repr__(self):
        vars_list = [f"{key}={value!r}" for key, value in vars(self).items()]
        vars_str = "\n  ".join(vars_list)
        return f"\n[\n  {vars_str}\n]"


def args_parser(params):
    parser = argparse.ArgumentParser(
        description="R2N - Run the relational reasoning model for the countries dataset."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="run the model(s) (R2NC, R2NS, R2NSO) [R2NC R2NS R2NSO]",
        metavar="MODEL",
        default="R2NC",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="task for the countries dataset [S1 S2 S3]",
        metavar="TASK",
        default="S1",
    )
    parser.add_argument(
        "-p",
        "--pretrain",
        help="set pretrain to true",
        action="store_true",
    )
    parser.add_argument(
        "-E",
        "--pretrain_epochs",
        type=int,
        help="number of pretrain epochs [50]",
        metavar="PRETRAIN_EPOCHS",
        default=50,
    )
    parser.add_argument(
        "-L",
        "--pretrain_learning_rate",
        type=float,
        help="pretrain learning rate [0.03]",
        metavar="PRETRAIN_LR",
        default=0.03,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="number of epochs [200]",
        metavar="EPOCHS",
        default=200,
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="learning rate [0.01]",
        metavar="LR",
        default=0.01,
    )
    parser.add_argument(
        "-a",
        "--atom_embedding_size",
        type=int,
        help="atom embedding size [30]",
        metavar="ATOM_EMBEDDING_SIZE",
        default=30,
    )
    parser.add_argument(
        "--transformer_embedding_size",
        type=int,
        help="transformer embedding size [30]",
        metavar="TRANSFORMER_EMBEDDING_SIZE",
        default=30,
    )
    parser.add_argument(
        "-T",
        "--num_transformers",
        type=int,
        help="number of chained transformers (multi-hops) [3]",
        metavar="NUM_TRANSFORMERS",
        default=3,
    )
    parser.add_argument(
        "-ae",
        "--atom_embedder",
        type=str,
        help="the type of used atom embedder",
        metavar="ATOM_EMBEDDER",
        default="TransE",
    )
    parser.add_argument(
        "-fe",
        "--formula_embedder",
        type=str,
        help="the type of used formula embedder",
        metavar="FORMULA_EMBEDDER",
        default="Transformers",
    )
    parser.add_argument(
        "--semantic_loss_weight",
        type=float,
        help="weight for the semantic loss [0.8]",
        metavar="SEMANTIC_LOSS_WEIGHT",
        default=0.8,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="the exp seed",
        metavar="SEED",
        default="0",
    )

    parser.add_argument(
        "--formula_file",
        type=str,
        help='File containing the logical formulas ["data/countries/formulas.txt"]',
        metavar="FORMULA_FILE",
        default=r"data/countries/formulas.txt",
    )

    args = parser.parse_args()
    for arg in vars(args):
        val = getattr(args, arg)
        if string_equals_case_insensitive(arg, "atom_embedding_size"):
            setattr(params, arg, [val])
            assert params.atom_embedding_size and params.atom_embedding_size[-1] >= 0
        else:
            setattr(params, arg, val)

    return params


class MMapModelCheckpoint(tf.keras.callbacks.Callback):
  """Save models to Memory as a Keras callback."""

  def __init__(self, model: tf.keras.Model,
               monitor: str='val_loss',
               maximize: bool=True):

    self.model = model
    self.best_val = -sys.float_info.max if maximize else sys.float_info.max
    self.monitor = monitor
    self.best_weights = None
    self.best_epoch = None
    self.maximize = maximize

  def restore_weights(self, model=None):
    print('Restoring weights from epoch', self.best_epoch)
    if model is None:
      self.model.set_weights(self.best_weights)
    else:
      model.set_weights(self.best_weights)

  def on_epoch_end(self, epoch, logs):
    assert self.monitor in logs, 'Unknown metric %s in %s' % (
        self.monitor, str(logs))
    val = logs[self.monitor]
    if (self.maximize and val >= self.best_val) or (
        not self.maximize and val <= self.best_val):
      self.best_val = val
      self.best_weights = self.model.get_weights()
      self.best_epoch = epoch
      print('\nNew best val (%.3f)' % val, flush=True)

class BestMetricEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics, prefix=None):
        super().__init__()
        if prefix:
            self.metrics = []
            for m in metrics:
                self.metrics.append(prefix + m)
        else:
            self.metrics = metrics

        self.best_score = {}
        self.best_score_iteration = {}

    def on_epoch_end(self, epoch, logs=None):
        for m in self.metrics:
            assert m in logs, "Metric:%s not available in logs %s" % (m, logs)
            if m not in self.best_score or logs[m] > self.best_score[m]:
                self.best_score[m] = logs[m]
                self.best_score_iteration[m] = epoch
        s = "\nMetric Best"
        for m, iter in self.best_score_iteration.items():
            s += " %s:%.3f @iter:%d|" % (m, self.best_score[m], iter)
        print(s)


class RecomputeBestMetricEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, metrics, targets):
        super().__init__()
        self.inputs = inputs
        self.metrics = metrics
        self.targets = targets

        self.best_score = {}
        self.best_score_iteration = sortedcontainers.SortedDict()  # to print sorted
        self.best_score_time = collections.defaultdict(lambda: 0)
        self._start_time = None

    def on_epoch_end(self, epoch, logs=None):
        if self._start_time is None:
            self._start_time = time.time()
        outputs = self.model.predict(self.inputs)
        outputs = outputs[:2]  # TODO: clean this code, this removes the formula output
        targets = self.targets[
            :2
        ]  # TODO: clean this code, this removes the formula output
        # Work around to make this work with single or multi-output models.
        if len(targets) == 1:
            outputs = [outputs]
        for metric in self.metrics:
            base_metric_name = metric._name
            for i, (t, output) in enumerate(zip(targets, outputs)):
                base_name = "%d_%s" % (i, base_metric_name)
                mdict = metric(t, output)[-1]
                for metric_name, value in mdict.items():
                    name = "%s_%s" % (metric_name, base_name)
                    if name not in self.best_score or value > self.best_score[name]:
                        self.best_score[name] = value
                        self.best_score_iteration[name] = epoch
                        self.best_score_time[name] = time.time() - self._start_time

        s = "\nMetric Best"
        for name, iter in self.best_score_iteration.items():
            s += " %s:%.3f @iter:%d|" % (name, self.best_score[name], iter)
        print(s)


def get_accuracy_with_ids(y_true, y_pred, ids):
    targets = tf.gather(tf.squeeze(y_true), ids)
    targets = tf.argmax(targets, axis=1)
    predictions = tf.gather(tf.squeeze(y_pred), ids)
    predictions = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), tf.float32))
    return accuracy.numpy() if tf.executing_eagerly() else accuracy


def get_AUC_with_ids(y_true, y_pred, ids):
    def RocAucWrapper(y_true, y_pred):
        auc = 0.0
        num = 0.0
        # roc_auc_score crashes when one dim has no targets
        # (like oceania in some tasks). Therefore we split the
        # computation by class and then average by hand.
        for i in range(y_true.shape[-1]):
            # Select the i-th col for targets and predictions.
            yt = y_true[:, i : i + 1]
            yp = y_pred[:, i : i + 1]
            if tf.reduce_sum(yt).numpy() != 0.0:
                auc += sklearn.metrics.roc_auc_score(yt, yp, multi_class="ovr")
                num += 1
        return auc / num if num > 0.0 else 0.0

    targets = tf.gather(tf.squeeze(y_true), ids)
    predictions = tf.gather(tf.squeeze(y_pred), ids)
    result = tf.py_function(RocAucWrapper, (targets, predictions), tf.float32)
    return result.numpy() if tf.executing_eagerly() else result


def get_metrics(hb, test_ids, val_ids):
    """This is called from the main program to get the metrics for test and val data.

    [extended_summary]

    Args:
        hb ([type]): [description]
        test_ids ([type]): [description]
        val_ids ([type]): [description]
    """

    def _test_accuracy(y_true, y_pred):
        del y_true
        return get_accuracy_with_ids(hb, y_pred, test_ids)

    def _test_AUC(y_true, y_pred):
        del y_true
        return get_AUC_with_ids(hb, y_pred, test_ids)

    def _val_accuracy(y_true, y_pred):
        del y_true
        return get_accuracy_with_ids(hb, y_pred, val_ids)

    def _val_AUC(y_true, y_pred):
        del y_true
        return get_AUC_with_ids(hb, y_pred, val_ids)

    def _val_AUC_acc_tie_breaker(y_true, y_pred):
        del y_true
        return 0.99 * get_AUC_with_ids(
            hb, y_pred, val_ids
        ) + 0.01 * get_accuracy_with_ids(hb, y_pred, val_ids)

    _test_accuracy.__name__ = "test_accuracy"
    _test_AUC.__name__ = "test_auc"
    _val_accuracy.__name__ = "val_accuracy"
    _val_AUC.__name__ = "val_auc"
    _val_AUC_acc_tie_breaker.__name__ = "val_auc_acc_tie_breaker"

    return [
        _test_accuracy,
        _test_AUC,
        _val_accuracy,
        _val_AUC,
        _val_AUC_acc_tie_breaker,
    ]  # for best model selection


def get_mask(o, constants, given_constants, predicates):
    ids = []
    mask = np.zeros([o.linear_size()])
    for p in predicates:
        for c in constants:
            for r in given_constants:
                atom_str = "%s(%s, %s)." % (p, c, r)
                id = o.atom_string_to_id(atom_str)
                ids.append(id)
    mask[ids] = 1
    mask = np.expand_dims(mask, 0)
    return mask


def get_predicate_mask(o, predicates):
    mask = np.zeros([o.linear_size()])
    for p in predicates:
        a, b = o._predicate_range[p]
        mask[a:b] = 1.0
    mask = np.expand_dims(mask, 0)
    return mask


def get_target_ids(
    ontology: knowledge.Ontology, constants_1: list, constants_2: list, predicates: list
):
    """This is very custom and works only for binary predicates.

    [extended_summary]

    Args:
        ontology (knowledge.Ontology): [description]
        constants_1 (list): [description]
        constants_2 (list): [description]
        predicates (list): [description]

    Returns:
        [type]: [description]
    """
    ids = []
    for p in predicates:
        for c1 in constants_1:
            ids_c1 = []
            for c2 in constants_2:
                atom_str = "%s(%s, %s)." % (p, c1, c2)
                id = ontology.atom_string_to_id(atom_str)
                ids_c1.append(id)
            ids.append(ids_c1)
    return ids


def get_mask_from_ids(ontology: knowledge.Ontology, ids: list):
    """Returns the mask of the 'linearized' ontology, given the list of indices.

    Args:
        ontology (mme.knowledge.Ontology):
            Dataset ontology.
        ids (list):
            List of ids to be used to build the mask.

    Returns:
        mask (tf.Tensor):
            Mask of shape (1, ontology.linear_size()), having 1s in correspondence of the
            indices.
    """
    if any(isinstance(sublist, list) for sublist in ids):
        ids = [el for sublist in ids for el in sublist]

    mask = np.zeros([ontology.linear_size()])
    mask[ids] = 1
    mask = np.expand_dims(mask, 0)
    return mask


def get_ids_and_mask(
    ontology: knowledge.Ontology,
    path_constants: str,
    path_regions: str,
    predicates: list,
):
    """Get the ids and the relative mask given a set of constants, regions and predicates.

    The constants, regions and predicates are intermixed in every possible combination,
    thus giving a list of indices which are then converted in a mask of the linearized
    version of the ontology.

    Args:
        ontology (mme.knowledge.Ontology):
            Dataset ontology.
        path_constants (str):
            File containing a list of constants.
        path_regions (str):
            File containing a list of regions.
        predicates (list):
            List of predicates.

    Returns:
        ids (list):
            Indices of the atoms built combining constants, regions and predicates.
        mask (tf.Tensor):
            Mask of shape (1, ontology.linear_size()), having 1s in correspondence of the
            ids.
    """
    constants = read_file_by_lines(path_constants)
    regions = read_file_by_lines(path_regions)
    ids = get_target_ids(ontology, constants, regions, predicates)
    mask = get_mask_from_ids(ontology, ids)

    return ids, mask


def is_bool(value):
    return value == "true" or value == "True" or value == "false" or value == "False"


def is_number(value):
    if value == "False" or value == "True":
        return False
    regnum = re.compile(r"\d+\.?\d*")
    return regnum.fullmatch(value)


def is_string(value):
    if value == "False" or value == "True":
        return False
    return not is_number(value)


def string_equals_case_insensitive(s1, s2):
    return s1.lower() == s2.lower()


def get_formulas_from_file(
    filename: str,
    ontology: knowledge.Ontology,
    herbrand_interpretation: tf.Tensor,
    evaluate: bool = False,
):
    """Read formulas from file.

    TODO [extended_summary]

    Args:
        filenames (str):
            Input(s) formula files.
        ontology (mme.knowledge.Ontology):
            Dataset ontology.
        herbrand_interpretation (tf.Tensor):
            Mask of shape (1, o.linear_size()) having 1s in correspondence of the
            indices of the atoms in the knowledge base.
        evaluate (bool, optional):
            Whether formulas should be evaluated or not. Defaults to False.
    """

    def _assemble_formula(definition: str, filter: str = None, hard: bool = None):
        """Take a formula definition and a given filter and return a GroundedFormula.

        Args:
            definition (str):
                Formula definition.
            filter (str, optional):
                Formula filter definition. Defaults to None.
            hard (bool, optional):
                Whether the formula is hard or not. Defaults to None.

        Returns:
            (GroundedFormula):
                Output GroundedFormula given the definition and the filter.
        """
        formula = knowledge.Formula(ontology=ontology, definition=definition, hard=hard)
        if not evaluate:
            return knowledge.GroundedFormula(
                formula,
                filter,
                formula.grounding_indices(filter, herbrand_interpretation),
            )
        else:
            groundings = formula.ground(herbrand_interpretation, filter)
            evaluation = formula.compile(groundings, logic.BooleanLogic)
            return knowledge.GroundedFormula(
                formula,
                filter,
                formula.grounding_indices(filter, herbrand_interpretation),
                evaluation,
            )

    grounded_formulas = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            is_comment = line.lstrip().startswith("#")
            if is_comment:
                continue
            tokens = line.split("|")
            tokens[-1] = tokens[-1].rstrip()
            assert len(tokens) < 4, f"Wrong line {line}"
            if len(tokens) == 0:
                continue
            print(f"Building formula {line.rstrip()}")
            filter = tokens[1] if len(tokens) > 1 else None
            hard_soft = tokens[2] if len(tokens) > 2 else None
            hard = (
                False
                if hard_soft is None
                else string_equals_case_insensitive(hard_soft, "hard")
            )
            grounded_formula = _assemble_formula(
                definition=tokens[0], filter=filter, hard=hard
            )
            grounded_formulas.append(grounded_formula)

    return grounded_formulas


def read_file_by_lines(file):
    try:
        with open(file, "r") as f:
            return [line.rstrip() for line in f.readlines()]
    except IOError as _:
        assert False, "Couldn't open file (%s)" % file


def save(path, o):
    with open(path, "wb") as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore(path):
    with open(path, "rb") as handle:
        o = pickle.load(handle)
    return o


def ranking(relations_train, relations_test, relations_predicted):
    MRR = 0.0
    HITS1 = 0.0
    HITS3 = 0.0
    HITS5 = 0.0
    HITS10 = 0.0
    counter = 0.0

    for relation in relations_test.keys():
        r_test = relations_test[relation]
        r_predicted = relations_predicted[relation]
        r_train = relations_train[relation]
        n = np.shape(r_test)[0]
        for i in range(n):
            for j in range(n):
                if r_test[i, j] == 1:
                    for s, k in enumerate((i, j)):

                        # s k
                        # 0 i
                        # 1 j

                        predicted_score = r_predicted[i, j]

                        # we multiply for 1 - r_train in such a way to eliminate the scores coming from train data
                        if s == 0:
                            mask = (
                                1 - r_train[i]
                            )  # ones row with 0 when the data is in the training data
                            mask[j] = 1
                            all_scores = sorted(r_predicted[i] * mask, reverse=True)
                        else:
                            mask = 1 - r_train[:, j]
                            mask[i] = 1
                            all_scores = sorted(r_predicted[:, j] * mask, reverse=True)
                        rank = all_scores.index(predicted_score) + 1

                        # if k == i:
                        #     all_scores = np.argsort(r_predicted[i])[::-1]
                        #     rank = list(all_scores).index(j) + 1
                        # else:
                        #     all_scores = np.argsort(r_predicted[:, j])[::-1]
                        #     rank = list(all_scores).index(i) + 1

                        counter += 1.0
                        if rank <= 1:
                            HITS1 += 1.0
                        if rank <= 3:
                            HITS3 += 1.0
                        if rank <= 5:
                            HITS5 += 1.0
                        if rank <= 10:
                            HITS10 += 1.0

                        MRR += 1.0 / rank

    MRR /= counter
    HITS1 /= counter
    HITS3 /= counter
    HITS5 /= counter
    HITS10 /= counter

    return (MRR, HITS1, HITS3, HITS5, HITS10)


def accuracy(y, targets):
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(targets, axis=1)), tf.float32)
    ).numpy()


def binary_accuracy(y, targets):
    return tf.reduce_mean(tf.cast(tf.equal(y > 0.5, targets > 0.5), tf.float32)).numpy()


def get_data_countries(
    ontology: knowledge.Ontology,
    path_task_kb: str,
    train_predicates: list,
    path_val_constants: str,
    val_predicates: list,
    path_val_regions: list,
    path_test_constants: str,
    test_predicates: list,
    path_test_regions: list,
    params: Params,
):
    """Generate train, val and test data for countries dataset.

    TODO [extended_summary]

    Args:
        ontology (knowledge.Ontology):
            Dataset ontology.
        path_task_knowledge_base (str):
            File containing the task specific training examples, used for labels.
        train_predicates (list):
            List of predicates used to build training atoms.
        path_val_constants (str):
            File containing a list of countries used to build validation atoms.
        val_predicates (list):
            List of predicates used to build validation atoms.
        path_val_regions (list):
            File containing a list of regions used to build validation atoms.
        path_test_constants (str):
            File containing a list of countries used to build test atoms.
        test_predicates (list):
            List of predicates used to build testatoms.
        path_test_regions (list):
            File containing a list of regions used to build test atoms.
        params (utils.Params):
            Default and commandline parameters.

    Returns:
        labels (tf.Tensor):
            Label of atoms present in the task specific knowledge base (known to be true).
        train_mask (tf.Tensor):
            Mask for the training atoms.
        val_ids (list):
            Indices of the validation atoms.
        val_mask (tf.Tensor):
            Mask for the validation atoms.
        test_ids (list):
            Indices of the test atoms.
        test_mask (tf.Tensor):
            Mask for the test atoms.
    """

    # Reading task specific training atoms
    train_atom_strings = read_file_by_lines(path_task_kb)

    # Building atoms labels (for the ones which are known)
    # NOTE: negative sampling will be necessary to create labels for unknown atoms
    labels = tf.constant(
        tf.expand_dims(ontology.mask_by_atom_strings(train_atom_strings), axis=0)
    )

    # Building validation ids and mask
    val_ids, val_mask = get_ids_and_mask(
        ontology, path_val_constants, path_val_regions, val_predicates
    )

    # Building test ids and mask
    test_ids, test_mask = get_ids_and_mask(
        ontology, path_test_constants, path_test_regions, test_predicates
    )

    # Building mask for either validation and test atoms
    test_val_mask = tf.math.logical_or(
        tf.cast(test_mask, tf.bool), tf.cast(val_mask, tf.bool)
    )

    # Building mask for learning predicates
    train_predicates_mask = get_predicate_mask(ontology, train_predicates)

    # Building mask for learning predicates to be used as training
    train_mask = tf.cast(
        tf.math.logical_and(train_predicates_mask, tf.math.logical_not(test_val_mask)),
        tf.float32,
    )

    # Building train mask
    # NOTE: not all atoms belonging to this mask are in the knowledge base.
    # Those which are in the kb can be weighted with `positive_example_weight`.
    train_mask = tf.constant(
        tf.where(
            tf.math.logical_and(train_mask > 0.0, labels > 0.0),
            params.positive_example_weight * tf.ones_like(train_mask),
            train_mask,
        )
    )

    return labels, train_mask, val_ids, val_mask, test_ids, test_mask


def get_constant_embedding_size(inputs, domain, params):
    """Layers embedding Constants.
    Get the embedding size for a given domain.
    Returns:
    - if embedding is enabled for the domain
    - the embedding size, or the input size when embedding is not enabled and the input space has features.
    """
    if domain is not None and domain in params.per_domain_constant_embedding_sizes:
        return True, params.per_domain_constant_embedding_sizes[domain]

    elif params.constant_embedding_sizes and params.constant_embedding_sizes[-1] > 0:
        # global constant
        return True, params.constant_embedding_sizes[-1]

    elif (
        domain is not None
        and inputs is not None
        and isinstance(inputs, dict)
        and domain in inputs
    ):
        # feature based, no embedding
        return False, inputs[domain].shape[-1]

    # one-hot encoding
    return False, 0


def get_constant_embedding_size_by_domain(inputs, domain_names, params):
    """Get the embedding size for all domains"""
    embedding_enabled = False
    domain2constant_size = {}
    domain2constant_enabled = {}
    for domain in domain_names:
        enable, size = get_constant_embedding_size(inputs, domain, params)
        domain2constant_enabled[domain] = enable
        domain2constant_size[domain] = size
        if enable:
            embedding_enabled = True

    return embedding_enabled, domain2constant_size, domain2constant_enabled


def get_optimizer(str):
    if not str:
        return None
    if string_equals_case_insensitive(str, "adam"):
        return tf.keras.optimizers.Adam
    if string_equals_case_insensitive(str, "adagrad"):
        return tf.keras.optimizers.Adagrad
    if string_equals_case_insensitive(str, "adadelta"):
        return tf.keras.optimizers.Adadelta
    if string_equals_case_insensitive(str, "rmsprop"):
        return tf.keras.optimizers.RMSprop
    if string_equals_case_insensitive(str, "adamax"):
        return tf.keras.optimizers.Adamax
    if string_equals_case_insensitive(str, "sgd"):
        return tf.keras.optimizers.SGD
    else:
        assert False, "Unknown optimizer %s" % str


def get_activation(string: str):
    if not string or string_equals_case_insensitive(string, "linear"):
        return None
    elif string_equals_case_insensitive(string, "relu"):
        return tf.nn.relu
    elif string_equals_case_insensitive(string, "elu"):
        return tf.nn.elu
    elif string_equals_case_insensitive(string, "sigmoid"):
        return tf.nn.sigmoid
    elif string_equals_case_insensitive(string, "tanh"):
        return tf.nn.tahn
    elif string_equals_case_insensitive(string, "softmax"):
        return tf.nn.softmax
    else:
        assert False, f"Unknown activation {string}"

def StringEqualsCaseInsensitive(s1, s2):
  return s1.lower() == s2.lower()
