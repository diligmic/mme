import copy
import os
import sys
import tempfile
import random
import tensorflow as tf
import numpy as np

from r2n import knowledge, utils, losses, factories

# Default hyperparameters
params = utils.Params()

# Command line hyperparameters
params = utils.args_parser(params)

print("PARAMS", params)

# Reproducibility.
random.seed(params.seed)
np.random.seed(params.seed)
tf.random.set_seed(params.seed)

# Task specific files and structures
task = params.task
train_predicates = ["locatedIn"]
val_predicates = test_predicates = ["locatedIn"]
formula_file = params.formula_file
path_countries = r"data/countries"
suffix = r"_with_constants"
path_kb = os.path.join(path_countries, f"knowledge{suffix}.nl")
path_task_kb = os.path.join(path_countries, f"{task}{suffix}.nl")
path_val_constants = os.path.join(path_countries, f"countries_dev.txt")
path_test_constants = os.path.join(path_countries, f"countries_test.txt")
path_val_regions = path_test_regions = os.path.join(path_countries, f"regions.txt")

# Ontology and herbrand interpretation
print("Building ontology and herbrand interpretation")
ontology = knowledge.Ontology.from_file(path_kb)
herbrand_interpretation = tf.expand_dims(ontology.mask_by_file(path_kb), axis=0)

params.constant_embedding_sizes = copy.deepcopy(params.atom_embedding_size)
if utils.StringEqualsCaseInsensitive(params.atom_embedder, 'ComplEx') == True:
    params.constant_embedding_sizes[-1] = params.atom_embedding_size[-1] * 2

############################################

# Train, validation and test data
print("Reading train, validation and test data")
(
    y,  # supervision for the atoms that are in the knowledge of the task
    train_mask,  # mask for the atoms, depending on training predicate (ex. locatedIn)
    val_ids,  # validation atoms ids
    val_mask,  # mask for the validation atoms
    test_ids,  # test atom ids
    test_mask,  # mask for the test atoms
) = utils.get_data_countries(
    ontology,
    path_task_kb,
    train_predicates,
    path_val_constants,
    val_predicates,
    path_val_regions,
    path_test_constants,
    test_predicates,
    path_test_regions,
    params,
)

# Formulas
print("Building formulas")
grounded_formulas = utils.get_formulas_from_file(
    formula_file, ontology, herbrand_interpretation
)

# if params.semantic_on_output or params.semantic_ground_formulas:
if utils.string_equals_case_insensitive(
    params.model, "R2NS"
) or utils.string_equals_case_insensitive(params.model, "R2NSO"):
    # Building targets and masks for cliques
    print("Building labels and masks for cliques")
    cliques_masks = []
    cliques_labels = []
    for grounded_formula in grounded_formulas:
        if grounded_formula.formula.is_hard():
            (
                cliques_mask_per_formula,
                cliques_labels_per_formula,
            ) = grounded_formula.get_cliques_masks_and_labels(
                train_mask, herbrand_interpretation
            )
            cliques_masks.append(cliques_mask_per_formula)
            cliques_labels.append(cliques_labels_per_formula)
    cliques_mask = tf.concat(cliques_masks, axis=1)
    cliques_labels = tf.concat(cliques_labels, axis=1)
    labels = tf.concat([y, cliques_labels], axis=-1)
else:
    cliques_mask, cliques_labels, labels = None, None, y

# Model input
print("Building model input")
input_keras_np = {name: domain.features for name, domain in ontology.domains.items()}
input_keras = {k: tf.keras.Input(f.shape[1:]) for k, f in input_keras_np.items()}


# Building model layers
print("Building model layers")
(
    constant_embedding_layer,
    atom_embedding_layer,
    no_reasoning_output_layer,
    reasoning_layer,
    reasoning_output_layer,
    cliques_layer,
) = factories.layers_factory(input_keras, ontology, grounded_formulas, params)

# Model train
print("Training model")
loss = losses.supervised_loss(train_mask, cliques_mask, params)
metrics = utils.get_metrics(herbrand_interpretation, test_ids, val_ids)

if params.pretrain:
    pretrain_loss = losses.supervised_loss(train_mask, None, params)
    model_no_reasoning = factories.no_reasoning_model_factory(
        input_keras,
        constant_embedding_layer,
        atom_embedding_layer,
        no_reasoning_output_layer,
        pretrain_loss,
        metrics,
        params,
    )

    model_no_reasoning.fit(
        input_keras_np, y=y, epochs=params.pretrain_epochs, shuffle=False
    )

model = factories.model_factory(
    input_keras,
    constant_embedding_layer,
    atom_embedding_layer,
    reasoning_layer,
    reasoning_output_layer,
    cliques_layer,
    loss,
    metrics,
    params,
)

# A faster version of ModelCheckpoint storing weights in memory without
# dumping to file.
best_model_callback = utils.MMapModelCheckpoint(
    model, "val_auc_acc_tie_breaker", maximize=True)

model.fit(
    input_keras_np,
    y=labels,
    epochs=params.epochs,
    callbacks=[best_model_callback],
    shuffle=False,
)

# print("Reload best model from", checkpoint_filename)
#model.load_weights(checkpoint_filename)
best_model_callback.restore_weights(model)

predictions = model(input_keras_np)
print("Accuracy_with_reasoning", metrics[0](None, predictions))
print("AUC_with_reasoning", metrics[1](None, predictions))

if params.pretrain:
    predictions_no_reasoning = model_no_reasoning(input_keras_np)
    print("Accuracy_no_reasoning", metrics[0](None, predictions_no_reasoning))
    print("AUC_no_reasoning", metrics[1](None, predictions_no_reasoning))
