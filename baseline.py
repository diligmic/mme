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
params.formula_embedder = None  # disable formulas

print("PARAMS", params)

# Reproducibility.
random.seed(params.seed)
np.random.seed(params.seed)
tf.random.set_seed(params.seed)

# Task specific files and structures
task = params.task
train_predicates = ["locatedIn"]
val_predicates = test_predicates = ["locatedIn"]
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
    _, _, _
) = factories.layers_factory(input_keras, ontology, None, params)

# Model train
print("Training model")
metrics = utils.get_metrics(herbrand_interpretation, test_ids, val_ids)

loss = losses.supervised_loss(train_mask, None, params)
model = factories.no_reasoning_model_factory(
    input_keras,
    constant_embedding_layer,
    atom_embedding_layer,
    no_reasoning_output_layer,
    loss,
    metrics,
    params,
)

best_model_callback = utils.MMapModelCheckpoint(
    model, "val_auc_acc_tie_breaker", maximize=True)

model.fit(input_keras_np, y=y,
          epochs=params.epochs,
          callbacks=[best_model_callback],
          shuffle=False)

best_model_callback.restore_weights(model)

predictions = model(input_keras_np)
print("Accuracy_baseline", metrics[0](None, predictions))
print("AUC_baseline", metrics[1](None, predictions))
