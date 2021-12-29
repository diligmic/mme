import os
import sys
import tempfile

import tensorflow as tf

from r2n import knowledge, utils, losses, factories, logic


# Default hyperparameters
params = utils.Params()
params.atom_embedder = "TransE"
params.formula_embedder = "Transformers"
params.constant_embedding_sizes = []
params.atom_embedding_sizes = [30]
params.pre_train_epochs = 5
params.epochs = 100
params.pretrain = True
params.debug = True  # enabling eagerly, needed by the semantic loss, check why
params.formula_weight = 1.0
params.formula_balance_weight = 0.1
params.semantic_on_output = True
params.positive_example_weight = 5

# Command line hyperparameters
opts, params = utils.parse_args(sys.argv, params)
print("PARAMS", params)
print("OPTS", opts)

# Task specific files and structures
task = opts["task"] if "task" in opts else "S1"
train_predicates = ["locatedIn"]
val_predicates = test_predicates = ["locatedIn"]
evidence_predicates = [
    "isCountry",
    "isRegion",
    "isContinent",
    "neighborOf",
    "transNeigh",
]
formula_file = (
    opts["formula_file"] if "formula_file" in opts else r"data/countries/formulas.txt"
)
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

# Train, validation and test data
print("Reading train, validation and test data")
(
    labels,  # supervision for the atoms that are in the knowledge of the task
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
grounded_formulas = utils.get_formula_from_csv(
    formula_file, ontology, herbrand_interpretation
)

formulas_cliques_masks = []
formulas_cliques_labels = []


def get_cliques_mask_and_labels(grounded_formula, train_mask, herbrand_interpretation):
    # Assign to each atom of the groundings the value True if it belongs to the train_mask
    train_groundings_mask = tf.cast(
        tf.gather(
            params=train_mask, indices=grounded_formula.grounding_indices, axis=-1
        ),
        tf.bool,
    )
    # Assign to each atom of the groundings its truth value, based on the hb
    groundings = tf.gather(
        params=herbrand_interpretation,
        indices=grounded_formula.grounding_indices,
        axis=-1,
    )
    # Compute the cliques mask, by selecting the cliques having all atoms in train_mask
    formula_cliques_mask = tf.cast(
        tf.reduce_all(train_groundings_mask, axis=-1),
        tf.float32,
    )
    # Compute the cliques truth values (valid for hard rules)
    formula_cliques_labels = tf.cast(
        grounded_formula.formula.compile(groundings, logic.BooleanLogic),
        tf.float32,
    )

    return formula_cliques_mask, formula_cliques_labels


formulas_cliques_labels = []
formulas_cliques_masks = []
for grounded_formula in grounded_formulas:
    cliques_mask, cliques_labels = get_cliques_mask_and_labels(
        grounded_formula, train_mask, herbrand_interpretation
    )
    formulas_cliques_labels.append(cliques_labels)
    formulas_cliques_masks.append(cliques_mask)


print("ciao")
